"""Shared Phase-D runtime, sensor, and failure contracts."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PhaseDTiming:
    """Resolved timing values used by a single Phase-D rollout."""

    physics_dt_s: float
    control_decimation: int
    policy_dt_s: float
    hold_policy_steps: int
    upper_command_hz: float

    @property
    def hold_physics_steps(self):
        return int(self.hold_policy_steps) * int(self.control_decimation)

    def to_dict(self):
        return {
            "physics_dt_s": float(self.physics_dt_s),
            "control_decimation": int(self.control_decimation),
            "policy_dt_s": float(self.policy_dt_s),
            "upper_command_hz": float(self.upper_command_hz),
            "hold_policy_steps": int(self.hold_policy_steps),
            "hold_physics_steps": int(self.hold_physics_steps),
        }


def resolve_phase_d_timing(sim_dt, control_decimation, upper_command_hz):
    """Resolve and validate the physics/policy/high-level timing contract."""
    physics_dt = float(sim_dt)
    decimation = int(control_decimation)
    upper_hz = float(upper_command_hz)
    if not math.isfinite(physics_dt) or physics_dt <= 0.0:
        raise ValueError("sim_dt must be a positive finite value")
    if decimation <= 0:
        raise ValueError("control_decimation must be positive")
    if not math.isfinite(upper_hz) or upper_hz <= 0.0:
        raise ValueError("upper_command_hz must be a positive finite value")
    policy_dt = physics_dt * decimation
    raw_steps = 1.0 / (upper_hz * policy_dt)
    hold_steps = int(round(raw_steps))
    if hold_steps <= 0 or abs(raw_steps - hold_steps) > 1.0e-6:
        raise ValueError(
            "upper command period must be an integer number of policy steps: "
            "%s" % raw_steps
        )
    return PhaseDTiming(
        physics_dt_s=physics_dt,
        control_decimation=decimation,
        policy_dt_s=policy_dt,
        hold_policy_steps=hold_steps,
        upper_command_hz=1.0 / (policy_dt * hold_steps),
    )


def require_isaacgym_depth(requested, actual):
    """Fail closed unless a formal rollout used real IsaacGym depth."""
    requested = str(requested).lower() if requested is not None else "none"
    actual = str(actual).lower() if actual is not None else "none"
    if requested != "isaacgym" or actual != "isaacgym":
        raise RuntimeError(
            "formal Phase D requires depth_backend_requested=isaacgym and "
            "depth_backend_actual=isaacgym; got requested=%s actual=%s"
            % (requested, actual)
        )
    return True


FAILURE_REASONS = (
    "SUCCESS",
    "COLLISION",
    "TIMEOUT",
    "GOAL_PROGRESS_FAILURE",
    "TERMINAL_CONVERGENCE_FAILURE",
    "TRANSITION_MANAGER_STALL",
    "LOW_LEVEL_TRACKING_FAILURE",
    "PROCESS_FAILURE",
    "UNKNOWN",
)


def classify_phase_d_failure(
    *,
    success=False,
    collision=False,
    process_failure=False,
    transition_manager_stall=False,
    low_level_tracking_failure=False,
    terminal_convergence_failure=False,
    goal_progress_failure=False,
    timeout=False,
):
    """Return one primary reason with deterministic, evidence-first precedence."""
    if bool(success):
        return "SUCCESS"
    if bool(process_failure):
        return "PROCESS_FAILURE"
    if bool(collision):
        return "COLLISION"
    if bool(transition_manager_stall):
        return "TRANSITION_MANAGER_STALL"
    if bool(low_level_tracking_failure):
        return "LOW_LEVEL_TRACKING_FAILURE"
    if bool(terminal_convergence_failure):
        return "TERMINAL_CONVERGENCE_FAILURE"
    if bool(goal_progress_failure):
        return "GOAL_PROGRESS_FAILURE"
    if bool(timeout):
        return "TIMEOUT"
    return "UNKNOWN"


def terminal_convergence_evidence(rows, *, region_radius=1.5, success_radius=0.35, progress_epsilon=1.0e-4, minimum_window=25):
    """Check for sustained near-goal stagnation without collision."""
    rows = list(rows)
    if len(rows) < int(minimum_window):
        return False
    run = 0
    previous = None
    for row in rows:
        distance = float(row.get("global_goal_distance_m", row.get("goal_distance_m", float("inf"))))
        collision = bool(row.get("collision", False))
        if previous is None:
            progress = float("inf")
        else:
            progress = previous - distance
        stalled = (
            not collision
            and success_radius < distance <= region_radius
            and abs(progress) <= float(progress_epsilon)
        )
        run = run + 1 if stalled else 0
        if run >= int(minimum_window):
            return True
        previous = distance
    return False


def transition_manager_stall_evidence(rows, *, command_threshold=0.02, progress_epsilon=1.0e-4, minimum_window=5):
    """Check for a sustained command suppression window in trajectory rows."""
    rows = list(rows)
    if len(rows) < int(minimum_window):
        return False
    run = 0
    previous_goal_distance = None
    for row in rows:
        desired = math.hypot(
            float(row.get("command_target_v_mps", row.get("desired_v_scheduled_mps", 0.0))),
            float(row.get("command_target_w_rps", row.get("desired_w_scheduled_rps", 0.0))),
        )
        applied = math.hypot(
            float(row.get("applied_v_mps", 0.0)), float(row.get("applied_w_rps", 0.0))
        )
        actual = math.hypot(float(row.get("actual_v_mps", 0.0)), float(row.get("actual_w_rps", 0.0)))
        goal_distance = float(row.get("global_goal_distance_m", row.get("goal_distance_m", float("inf"))))
        transition_active = bool(row.get("transition_active", False))
        progress = 0.0 if previous_goal_distance is None else previous_goal_distance - goal_distance
        suppressed = (
            desired > float(command_threshold)
            and goal_distance > float(row.get("goal_success_radius_m", 0.35))
            and transition_active
            and applied < max(float(command_threshold) * 0.5, desired * 0.25)
            and abs(progress) <= float(progress_epsilon)
            and actual <= max(float(command_threshold), 0.02)
        )
        run = run + 1 if suppressed else 0
        if run >= int(minimum_window):
            return True
        previous_goal_distance = goal_distance
    return False
