"""Explicit contract for the frozen uniform-4150 P2P controller."""

from pathlib import Path


P2P_TASK_NAME = "rotunbot_target_repro"
CHECKPOINT_RELATIVE_PATH = (
    "Aug16_02-57-06_uniform_t1_long500_from3809/model_4150.pt"
)
OBSERVATION_DIM = 19
FRAME_STACK = 20
ACTION_DIM = 2
CONTROL_TYPE = "DIRECT_VP_TORQUE"
VELOCITY_GAIN = 100.0
POSITION_GAIN = 600.0
SUCCESS_DISTANCE_M = 0.20
SUCCESS_SPEED_MPS = 0.10


def require_checkpoint(path):
    """Return an absolute checkpoint path or fail before policy construction."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            "Frozen uniform-4150 checkpoint does not exist: "
            f"{resolved}"
        )
    return resolved
