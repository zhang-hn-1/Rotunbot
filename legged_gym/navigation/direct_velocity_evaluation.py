"""Deterministic goal sets and formal gates for direct-velocity evaluation."""

import math
import hashlib
import json
import random
from pathlib import Path

from legged_gym.navigation.corridor_artifacts import EpisodeLogger, GateResult


DEFAULT_EVALUATION_SEEDS = (0, 1, 2)

_STAGE_BOUNDS = {
    "S1": ((0.5, 1.0), math.radians(10.0)),
    "S2": ((0.5, 1.5), math.radians(30.0)),
    "S2B": ((0.5, 2.0), math.radians(45.0)),
}

_STAGE_MIXTURES = {
    "S1": (("S1", 1.0),),
    "S2": (("S2", 0.70), ("S1", 0.30)),
    "S2B": (("S2B", 0.70), ("S2", 0.20), ("S1", 0.10)),
}

_ZERO_SAFETY_RULES = {
    "collision_count": ("==", 0),
    "rate_violation_count": ("==", 0),
    "feasible_domain_violation_count": ("==", 0),
    "hidden_projection_jump_count": ("==", 0),
}


class CommandDiagnostics:
    """Accumulate observable command-path and V62 safety diagnostics."""

    def __init__(
        self,
        policy_dt,
        maximum_linear_acceleration,
        maximum_yaw_acceleration,
        tolerance=3.0e-6,
    ):
        self.policy_dt = float(policy_dt)
        self.linear_step_limit = float(maximum_linear_acceleration) * self.policy_dt
        self.yaw_step_limit = float(maximum_yaw_acceleration) * self.policy_dt
        self.tolerance = float(tolerance)
        self.previous_applied = (0.0, 0.0)
        self.previous_transition_active = False
        self.counts = {
            "raw_reverse_command_count": 0,
            "requested_reverse_command_count": 0,
            "applied_reverse_command_count": 0,
            "rate_violation_count": 0,
            "feasible_domain_violation_count": 0,
            "hidden_projection_jump_count": 0,
            "transition_activation_count": 0,
            "reverse_transition_activation_count": 0,
            "transition_active_step_count": 0,
            "governor_activation_count": 0,
            "projection_activation_count": 0,
        }
        self.command_corrections = []

    @staticmethod
    def _pair(values):
        pair = tuple(float(value) for value in values)
        if len(pair) != 2:
            raise ValueError("commands must contain exactly two values")
        return pair

    def record(
        self,
        raw_command,
        requested_command,
        applied_command,
        projected_applied_command,
        transition_active,
    ):
        raw = self._pair(raw_command)
        requested = self._pair(requested_command)
        applied = self._pair(applied_command)
        projected_applied = self._pair(projected_applied_command)
        transition_active = bool(transition_active)
        delta_v = abs(applied[0] - self.previous_applied[0])
        delta_w = abs(applied[1] - self.previous_applied[1])
        rate_violation = (
            delta_v > self.linear_step_limit + self.tolerance
            or delta_w > self.yaw_step_limit + self.tolerance
        )
        domain_violation = max(
            abs(projected_applied[index] - applied[index]) for index in range(2)
        ) > self.tolerance
        transition_activation = transition_active and not self.previous_transition_active
        projection_active = max(
            abs(raw[index] - requested[index]) for index in range(2)
        ) > self.tolerance
        governor_active = max(
            abs(requested[index] - applied[index]) for index in range(2)
        ) > self.tolerance
        correction = math.sqrt(
            sum((requested[index] - applied[index]) ** 2 for index in range(2))
        )

        self.counts["raw_reverse_command_count"] += int(raw[0] < -self.tolerance)
        self.counts["requested_reverse_command_count"] += int(
            requested[0] < -self.tolerance
        )
        self.counts["applied_reverse_command_count"] += int(
            applied[0] < -self.tolerance
        )
        self.counts["rate_violation_count"] += int(rate_violation)
        self.counts["hidden_projection_jump_count"] += int(rate_violation)
        self.counts["feasible_domain_violation_count"] += int(domain_violation)
        self.counts["transition_activation_count"] += int(transition_activation)
        self.counts["reverse_transition_activation_count"] += int(
            transition_activation and requested[0] < -self.tolerance
        )
        self.counts["transition_active_step_count"] += int(transition_active)
        self.counts["governor_activation_count"] += int(governor_active)
        self.counts["projection_activation_count"] += int(projection_active)
        self.command_corrections.append(correction)
        self.previous_applied = applied
        self.previous_transition_active = transition_active
        return {
            "rate_violation": int(rate_violation),
            "feasible_domain_violation": int(domain_violation),
            "hidden_projection_jump": int(rate_violation),
            "transition_activation_event": int(transition_activation),
            "transition_active": int(transition_active),
            "governor_active": int(governor_active),
            "projection_active": int(projection_active),
            "raw_reverse_command": int(raw[0] < -self.tolerance),
            "requested_reverse_command": int(requested[0] < -self.tolerance),
            "applied_reverse_command": int(applied[0] < -self.tolerance),
            "command_correction": correction,
        }

    def summary(self):
        result = dict(self.counts)
        result["mean_command_correction"] = (
            sum(self.command_corrections) / len(self.command_corrections)
            if self.command_corrections
            else 0.0
        )
        return result


def _mixture_counts(stage, episodes):
    weighted = [
        (component, float(episodes) * fraction)
        for component, fraction in _STAGE_MIXTURES[stage]
    ]
    counts = {component: int(value) for component, value in weighted}
    remainder = episodes - sum(counts.values())
    ranked = sorted(
        enumerate(weighted),
        key=lambda item: (-(item[1][1] - int(item[1][1])), item[0]),
    )
    for index in range(remainder):
        counts[ranked[index][1][0]] += 1
    return counts


def build_fixed_goal_specs(stage, episodes, seed_list=None):
    """Build a deterministic exact-mixture formal goal set."""
    stage = str(stage).upper()
    if stage not in _STAGE_MIXTURES:
        raise ValueError("stage must be S1, S2 or S2B")
    episodes = int(episodes)
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    seeds = tuple(DEFAULT_EVALUATION_SEEDS if seed_list is None else seed_list)
    if not seeds:
        raise ValueError("seed_list must not be empty")

    components = []
    for component, count in _mixture_counts(stage, episodes).items():
        components.extend([component] * count)
    random.Random(int(seeds[0])).shuffle(components)
    generators = {int(seed): random.Random(int(seed)) for seed in seeds}
    specs = []
    for episode_index, component in enumerate(components):
        seed = int(seeds[episode_index % len(seeds)])
        generator = generators[seed]
        distance_range, bearing_max = _STAGE_BOUNDS[component]
        specs.append(
            {
                "episode_id": episode_index + 1,
                "seed": seed,
                "component": component,
                "distance_m": generator.uniform(*distance_range),
                "bearing_rad": generator.uniform(-bearing_max, bearing_max),
            }
        )
    return specs


def _stage_rules(stage, success_threshold):
    rules = dict(_ZERO_SAFETY_RULES)
    rules["success_rate"] = (">=", float(success_threshold))
    if stage == "S1":
        rules["divergence_rate"] = ("<=", 0.02)
        rules["timeout_rate"] = ("<=", 0.05)
    return rules


def evaluate_stage_gate(summary, stage):
    """Evaluate one formal B-stage set without defaulting missing metrics."""
    stage = str(stage).upper()
    if stage not in _STAGE_MIXTURES:
        raise ValueError("stage must be S1, S2 or S2B")
    threshold = 0.95 if stage == "S1" else 0.90
    result = GateResult.evaluate(summary, _stage_rules(stage, threshold), {})
    result["stage"] = stage
    return result


def evaluate_b_gate_chain(b3_summary, b2_summary, b1_summary):
    """Require the strict B3 current plus B2/B1 regression chain."""
    evaluations = (
        ("B3/S2B", b3_summary, _stage_rules("S2B", 0.90)),
        ("B2/S2", b2_summary, _stage_rules("S2", 0.90)),
        ("B1/S1", b1_summary, _stage_rules("S1", 0.93)),
    )
    failures = []
    details = {}
    for label, summary, rules in evaluations:
        result = GateResult.evaluate(summary, rules, {})
        details[label] = result
        failures.extend("%s: %s" % (label, reason) for reason in result["failures"])
    return {
        "pass": not failures,
        "current_pass": details["B3/S2B"]["pass"],
        "regression_pass": details["B2/S2"]["pass"] and details["B1/S1"]["pass"],
        "failures": failures,
        "details": details,
    }


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_identity(checkpoint, parent_checkpoint=None):
    """Resolve and verify checkpoint SHA and adjacent parent metadata."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(str(checkpoint))
    digest = _sha256(checkpoint)
    metadata_path = checkpoint.parent / "checkpoint_metadata.json"
    metadata = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        recorded_digest = metadata.get("sha256")
        if recorded_digest is not None and recorded_digest != digest:
            raise ValueError(
                "checkpoint SHA does not match %s: expected %s, got %s"
                % (metadata_path, recorded_digest, digest)
            )
    parent_value = parent_checkpoint or metadata.get("parent_checkpoint")
    parent_path = None
    parent_digest = None
    if parent_value:
        candidate = Path(parent_value).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        parent_path = candidate.resolve()
        if parent_path.is_file():
            parent_digest = _sha256(parent_path)
    return {
        "checkpoint": str(checkpoint),
        "sha256": digest,
        "parent_checkpoint": str(parent_path) if parent_path is not None else None,
        "parent_sha256": parent_digest,
        "metadata_path": str(metadata_path) if metadata_path.is_file() else None,
    }


def write_failure_artifacts(output_root, episode_record, trajectory_rows):
    """Write one replayable failed trajectory and its required plots."""
    from legged_gym.navigation.corridor_plotting import plot_corridor_artifacts

    episode_id = int(episode_record["episode_id"])
    root = Path(output_root) / "failures" / ("episode_%03d" % episode_id)
    logger = EpisodeLogger(root)
    logger.write_episode(episode_record)
    logger.write_trajectory(trajectory_rows)
    trajectory_csv = root / "trajectory.csv"
    plots = plot_corridor_artifacts(trajectory_csv, root / "plots")
    return {
        "root": str(root),
        "episodes_csv": str(root / "episodes.csv"),
        "trajectory_csv": str(trajectory_csv),
        "plots": plots,
    }


def summarize_evaluation(
    records,
    stage,
    seed_list,
    checkpoint_identity,
    wall_clock_seconds,
):
    """Aggregate formal episode records into the Task 7 summary contract."""
    records = list(records)
    if not records:
        raise ValueError("records must not be empty")
    total = len(records)
    count_keys = {
        "success_count": "success",
        "collision_count": "collision",
        "timeout_count": "timeout",
        "divergence_count": "divergent",
    }
    summary = {
        "stage": str(stage).upper(),
        "episodes": total,
        "fixed_seeds": [int(seed) for seed in seed_list],
        "wall_clock_seconds": float(wall_clock_seconds),
        "checkpoint": checkpoint_identity["checkpoint"],
        "checkpoint_sha256": checkpoint_identity["sha256"],
        "parent_checkpoint": checkpoint_identity.get("parent_checkpoint"),
        "parent_checkpoint_sha256": checkpoint_identity.get("parent_sha256"),
        "checkpoint_metadata_path": checkpoint_identity.get("metadata_path"),
        "mixture_counts": {
            component: sum(row.get("component") == component for row in records)
            for component in ("S2B", "S2", "S1")
            if any(row.get("component") == component for row in records)
        },
    }
    for output_key, record_key in count_keys.items():
        summary[output_key] = sum(int(bool(row.get(record_key, False))) for row in records)
    summary.update(
        {
            "success_rate": summary["success_count"] / total,
            "collision_rate": summary["collision_count"] / total,
            "timeout_rate": summary["timeout_count"] / total,
            "divergence_rate": summary["divergence_count"] / total,
            "mean_path_length_m": sum(float(row["path_length_m"]) for row in records) / total,
            "mean_terminal_goal_distance_m": sum(
                float(row["terminal_goal_distance_m"]) for row in records
            ) / total,
        }
    )
    counter_keys = (
        "rate_violation_count",
        "feasible_domain_violation_count",
        "hidden_projection_jump_count",
        "transition_activation_count",
        "reverse_transition_activation_count",
        "transition_active_step_count",
        "governor_activation_count",
        "projection_activation_count",
        "raw_reverse_command_count",
        "requested_reverse_command_count",
        "applied_reverse_command_count",
    )
    for key in counter_keys:
        summary[key] = sum(int(row.get(key, 0)) for row in records)
    summary["gate"] = evaluate_stage_gate(summary, stage)
    return summary
