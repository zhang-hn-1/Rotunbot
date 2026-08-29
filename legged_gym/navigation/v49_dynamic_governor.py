"""Opt-in state-dependent command governor for the V49 velocity interface.

The governor is a small deterministic search over upper-level ``[v, w]``
commands.  It has no access to policy actions or simulator state beyond the
measured velocity supplied in :class:`ReachabilityState`.
"""

from dataclasses import dataclass
import math

from .v49_dynamic_reachability import DynamicReachabilityTable, ReachabilityPrediction


def _pair(value, name):
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1).tolist()
    if len(value) != 2:
        raise ValueError("%s must contain [v, w]" % name)
    result = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) for item in result):
        raise ValueError("%s must be finite" % name)
    return result


@dataclass(frozen=True)
class DynamicGovernorConfig:
    """All search, objective, projection, and safety parameters."""

    enable_dynamic_governor: bool = False
    maximum_forward_speed: float = 0.35
    maximum_yaw_rate: float = 0.10
    minimum_turn_radius: float = 3.148148148148148
    envelope_fraction: float = 0.85
    stationary_threshold: float = 0.0
    turn_authority_start_speed: float = 0.0
    turn_authority_full_speed: float = 0.0
    candidate_forward_offsets: tuple = (-0.04, -0.02, 0.0, 0.02, 0.04)
    candidate_yaw_offsets: tuple = (-0.02, -0.01, 0.0, 0.01, 0.02)
    maximum_forward_command_step: float = 0.08
    maximum_yaw_command_step: float = 0.03
    weight_forward_error: float = 1.0
    weight_yaw_error: float = 1.0
    weight_command_delta: float = 0.10
    no_direction_reversal: bool = True
    direction_epsilon: float = 1.0e-5

    def __post_init__(self):
        for name in (
            "maximum_forward_speed", "maximum_yaw_rate", "minimum_turn_radius",
            "envelope_fraction", "maximum_forward_command_step",
            "maximum_yaw_command_step", "weight_forward_error",
            "weight_yaw_error", "weight_command_delta",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError("%s must be nonnegative" % name)
        if float(self.minimum_turn_radius) <= 0.0:
            raise ValueError("minimum_turn_radius must be positive")


@dataclass(frozen=True)
class GovernorDecision:
    command: tuple
    prediction: ReachabilityPrediction
    fallback: bool
    coverage: str
    modified: bool
    forward_modified: bool
    yaw_modified: bool
    cost: float


class StateDependentReachabilityGovernor:
    """Choose the best measured-reachable command under explicit constraints."""

    def __init__(self, table: DynamicReachabilityTable, config=None):
        if not isinstance(table, DynamicReachabilityTable):
            raise TypeError("table must be a DynamicReachabilityTable")
        self.table = table
        self.config = config or DynamicGovernorConfig()

    def project_command(self, command):
        """Match ``project_velocity_commands`` for scalar physical commands."""
        v, w = _pair(command, "command")
        cfg = self.config
        v = max(-float(cfg.maximum_forward_speed), min(float(cfg.maximum_forward_speed), v))
        yaw_limit = min(
            float(cfg.maximum_yaw_rate),
            abs(v) / float(cfg.minimum_turn_radius),
        ) * float(cfg.envelope_fraction)
        start_speed = float(cfg.turn_authority_start_speed)
        full_speed = float(cfg.turn_authority_full_speed)
        if full_speed > start_speed:
            fraction = max(0.0, min(1.0, (abs(v) - start_speed) / (full_speed - start_speed)))
            authority = fraction * fraction * (3.0 - 2.0 * fraction)
            yaw_limit *= authority
        w = max(-yaw_limit, min(yaw_limit, w))
        if abs(v) < float(cfg.stationary_threshold):
            w = 0.0
        return (round(v, 12), round(w, 12))

    def _bounded(self, command, previous):
        cfg = self.config
        v = max(
            previous[0] - float(cfg.maximum_forward_command_step),
            min(previous[0] + float(cfg.maximum_forward_command_step), command[0]),
        )
        w = max(
            previous[1] - float(cfg.maximum_yaw_command_step),
            min(previous[1] + float(cfg.maximum_yaw_command_step), command[1]),
        )
        return self.project_command((v, w))

    def _respects_direction(self, command, desired, previous):
        if not self.config.no_direction_reversal:
            return True
        epsilon = float(self.config.direction_epsilon)
        for value, requested, old in zip(command, desired, previous):
            reference = old if abs(old) > epsilon else requested
            if abs(reference) > epsilon and value * reference < -(epsilon ** 2):
                return False
        return True

    def _candidates(self, desired, previous):
        centers = (desired, previous)
        candidates = set()
        for center in centers:
            for forward_offset in self.config.candidate_forward_offsets:
                for yaw_offset in self.config.candidate_yaw_offsets:
                    raw = (center[0] + float(forward_offset), center[1] + float(yaw_offset))
                    bounded = self._bounded(self.project_command(raw), previous)
                    if self._respects_direction(bounded, desired, previous):
                        candidates.add(bounded)
        for command in (self.project_command(desired), self.project_command(previous)):
            bounded = self._bounded(command, previous)
            if self._respects_direction(bounded, desired, previous):
                candidates.add(bounded)
        return tuple(sorted(candidates))

    def _cost(self, prediction, target, previous, command):
        predicted_v, predicted_w = prediction.at_horizon(200)
        return (
            float(self.config.weight_forward_error) * (predicted_v - target[0]) ** 2
            + float(self.config.weight_yaw_error) * (predicted_w - target[1]) ** 2
            + float(self.config.weight_command_delta) * (
                (command[0] - previous[0]) ** 2 + (command[1] - previous[1]) ** 2
            )
        )

    def select_command(self, current_state, desired_command, previous_command=None):
        desired = _pair(desired_command, "desired_command")
        previous = self.project_command(
            (0.0, 0.0) if previous_command is None else _pair(previous_command, "previous_command")
        )
        target = self.project_command(desired)
        candidates = self._candidates(target, previous)
        if not candidates:
            candidates = (previous,)
        scored = []
        for command in candidates:
            prediction = self.table.predict_reachable_response(current_state, command)
            if prediction.out_of_coverage:
                continue
            cost = self._cost(prediction, target, previous, command)
            # Cost first, then smallest command change, then lexicographic
            # command order makes ties independent of set/hash iteration.
            scored.append((cost, abs(command[0] - target[0]) + abs(command[1] - target[1]), command, prediction))
        if scored:
            _, _, command, prediction = min(scored, key=lambda item: (item[0], item[1], item[2]))
            fallback = False
            cost = self._cost(prediction, target, previous, command)
        else:
            command = min(
                candidates,
                key=lambda item: (
                    abs(item[0] - target[0]) + abs(item[1] - target[1]), item,
                ),
            )
            prediction = self.table.predict_reachable_response(current_state, command)
            fallback = True
            cost = self._cost(prediction, target, previous, command)
        return GovernorDecision(
            command=command,
            prediction=prediction,
            fallback=fallback,
            coverage=prediction.coverage,
            modified=command != target,
            forward_modified=command[0] != target[0],
            yaw_modified=command[1] != target[1],
            cost=cost,
        )
