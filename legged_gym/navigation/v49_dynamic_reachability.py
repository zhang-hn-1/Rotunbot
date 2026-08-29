"""Deterministic table model for the frozen V49 200 ms response.

The model is deliberately independent of Isaac Gym.  Its command is the
existing physical ``[v, w]`` pair: body-forward velocity in m/s and
gravity-aligned world-Z yaw rate in rad/s.  Runtime callers are responsible
for applying the existing hard command projection before asking the table to
score a candidate; the table then clamps only to its measured coverage and
reports that fact.
"""

from dataclasses import dataclass
import csv
import itertools
import math


HORIZONS_MS = (50, 100, 150, 200)


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("%s must be finite" % name)
    return value


@dataclass(frozen=True)
class ReachabilityState:
    """State used to query the model, with physical SI units."""

    current_forward_velocity: float
    current_yaw_rate: float = 0.0
    body_linear_velocity: tuple = None
    body_angular_velocity: tuple = None
    joint_position: tuple = None
    joint_velocity: tuple = None
    previous_command: tuple = None
    base_orientation: tuple = None

    def __post_init__(self):
        _finite(self.current_forward_velocity, "current_forward_velocity")
        _finite(self.current_yaw_rate, "current_yaw_rate")

    @property
    def current_v(self):
        return float(self.current_forward_velocity)

    @property
    def current_w(self):
        return float(self.current_yaw_rate)

    @property
    def command_units(self):
        return ("m/s", "rad/s")


@dataclass(frozen=True)
class ReachabilityPrediction:
    """Predicted terminal response and coverage metadata."""

    projected_command: tuple
    coverage: str
    out_of_coverage: bool
    predicted_forward_velocity_50ms: float
    predicted_yaw_rate_50ms: float
    predicted_forward_velocity_100ms: float
    predicted_yaw_rate_100ms: float
    predicted_forward_velocity_150ms: float
    predicted_yaw_rate_150ms: float
    predicted_forward_velocity_200ms: float
    predicted_yaw_rate_200ms: float

    def at_horizon(self, horizon_ms):
        horizon_ms = int(horizon_ms)
        if horizon_ms not in HORIZONS_MS:
            raise ValueError("unsupported horizon: %s" % horizon_ms)
        return (
            getattr(self, "predicted_forward_velocity_%dms" % horizon_ms),
            getattr(self, "predicted_yaw_rate_%dms" % horizon_ms),
        )


def _command_pair(command):
    if hasattr(command, "detach"):
        command = command.detach().cpu().reshape(-1).tolist()
    if len(command) != 2:
        raise ValueError("command must contain [forward_v, yaw_w]")
    return (_finite(command[0], "command_v"), _finite(command[1], "command_w"))


def _state_velocity(state):
    if isinstance(state, ReachabilityState):
        return state.current_v
    if isinstance(state, dict):
        for name in ("current_forward_velocity", "current_v", "initial_v"):
            if name in state:
                return _finite(state[name], name)
    for name in ("current_forward_velocity", "current_v", "initial_v"):
        if hasattr(state, name):
            return _finite(getattr(state, name), name)
    raise TypeError("state must expose current_forward_velocity/current_v")


def _row_value(row, *names):
    for name in names:
        if name in row and row[name] not in (None, "", "None"):
            return _finite(row[name], name)
    raise ValueError("row is missing one of: %s" % ", ".join(names))


class DynamicReachabilityTable:
    """Sparse-safe trilinear table indexed by current-v and command v/w."""

    def __init__(self, values, current_v_knots, command_v_knots, command_w_knots):
        self._values = values
        self.current_v_knots = tuple(sorted(current_v_knots))
        self.command_v_knots = tuple(sorted(command_v_knots))
        self.command_w_knots = tuple(sorted(command_w_knots))
        if not self._values:
            raise ValueError("reachability table cannot be empty")

    @classmethod
    def from_rows(cls, rows):
        grouped = {}
        for row in rows:
            key = (
                _row_value(row, "current_v", "initial_v"),
                _row_value(row, "projected_v", "command_v", "target_v"),
                _row_value(row, "projected_w", "command_w", "target_w"),
            )
            measurements = []
            for horizon in HORIZONS_MS:
                measurements.extend((
                    _row_value(
                        row,
                        "predicted_forward_velocity_%dms" % horizon,
                        "mean_actual_v_%dms" % horizon,
                        "actual_v_%dms" % horizon,
                    ),
                    _row_value(
                        row,
                        "predicted_yaw_rate_%dms" % horizon,
                        "mean_actual_w_%dms" % horizon,
                        "actual_w_%dms" % horizon,
                    ),
                ))
            grouped.setdefault(key, []).append(measurements)
        values = {}
        for key, samples in grouped.items():
            values[key] = tuple(
                sum(sample[index] for sample in samples) / len(samples)
                for index in range(len(samples[0]))
            )
        return cls(
            values,
            {key[0] for key in values},
            {key[1] for key in values},
            {key[2] for key in values},
        )

    @classmethod
    def from_csv(cls, path):
        with open(path, newline="", encoding="utf-8") as handle:
            return cls.from_rows(csv.DictReader(handle))

    @staticmethod
    def _bounds(knots, value):
        if value <= knots[0]:
            return knots[0], knots[0], 0.0, value < knots[0]
        if value >= knots[-1]:
            return knots[-1], knots[-1], 0.0, value > knots[-1]
        for lower, upper in zip(knots, knots[1:]):
            if lower <= value <= upper:
                if upper == lower:
                    return lower, upper, 0.0, False
                return lower, upper, (value - lower) / (upper - lower), False
        raise AssertionError("value is not bracketed")

    def _interpolate(self, query):
        axes = (
            self.current_v_knots,
            self.command_v_knots,
            self.command_w_knots,
        )
        bounds = [self._bounds(axis, value) for axis, value in zip(axes, query)]
        clamped_query = tuple(item[0] + item[2] * (item[1] - item[0]) for item in bounds)
        was_clamped = any(item[3] for item in bounds)
        lower_upper = [(item[0], item[1]) for item in bounds]
        fractions = [item[2] for item in bounds]
        corner_keys = list(itertools.product(*lower_upper))
        corner_weights = []
        for bits, key in zip(itertools.product((0, 1), repeat=3), corner_keys):
            weight = 1.0
            for bit, fraction in zip(bits, fractions):
                weight *= fraction if bit else (1.0 - fraction)
            corner_weights.append((key, weight))
        # Sparse non-rectangular sweeps legitimately omit corners whose
        # interpolation weight is exactly zero.  Requiring those corners
        # would turn a measured knot into a false out-of-coverage query.
        if all(weight <= 1.0e-12 or key in self._values for key, weight in corner_weights):
            result = []
            for index in range(len(next(iter(self._values.values())))):
                total = 0.0
                for key, weight in corner_weights:
                    if weight > 1.0e-12:
                        total += weight * self._values[key][index]
                result.append(total)
            return tuple(clamped_query), tuple(result), ("clamped" if was_clamped else "interpolated")

        # A partial sweep is still queryable, but never silently extrapolated.
        nearest = min(
            self._values,
            key=lambda key: sum((key[index] - clamped_query[index]) ** 2 for index in range(3)),
        )
        return tuple(clamped_query), self._values[nearest], "clamped" if was_clamped else "nearest"

    def predict_reachable_response(self, current_state, desired_command):
        command_v, command_w = _command_pair(desired_command)
        query = (_state_velocity(current_state), command_v, command_w)
        projected_command, values, coverage = self._interpolate(query)
        return ReachabilityPrediction(
            projected_command=(projected_command[1], projected_command[2]),
            coverage=coverage,
            out_of_coverage=coverage in ("clamped", "nearest"),
            predicted_forward_velocity_50ms=values[0],
            predicted_yaw_rate_50ms=values[1],
            predicted_forward_velocity_100ms=values[2],
            predicted_yaw_rate_100ms=values[3],
            predicted_forward_velocity_150ms=values[4],
            predicted_yaw_rate_150ms=values[5],
            predicted_forward_velocity_200ms=values[6],
            predicted_yaw_rate_200ms=values[7],
        )


def predict_reachable_response(table, current_state, desired_command):
    """Functional facade for callers that keep the table as a dependency."""
    return table.predict_reachable_response(current_state, desired_command)
