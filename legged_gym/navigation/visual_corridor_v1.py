"""Geometry helpers for the first direct-SRU visual corridor stage."""

from legged_gym.navigation.corridor_scenarios import make_straight_scenario
from legged_gym.navigation.v62_corridor_task import make_wall_segments


V1_CORRIDOR_WIDTH_M = 2.0
V1_CORRIDOR_LENGTH_M = 6.0
V1_WALL_THICKNESS_M = 0.10


def build_v1_straight_geometry(
    width_m=V1_CORRIDOR_WIDTH_M,
    length_m=V1_CORRIDOR_LENGTH_M,
):
    """Return local wall segments and fallback-depth AABBs for V1.

    The two representations share the same centerline and are expressed in an
    environment-local frame.  The actor sees only the camera depth; the AABBs
    are used by the deterministic fallback backend and safety telemetry.
    """
    width_m = float(width_m)
    length_m = float(length_m)
    thickness = float(V1_WALL_THICKNESS_M)
    if width_m <= 0.0 or length_m <= 0.0:
        raise ValueError("V1 corridor width and length must be positive")
    scenario = make_straight_scenario(width_m, length_m, seed=0)
    segments = make_wall_segments(scenario.centerline)
    half_width = width_m / 2.0
    half_thickness = thickness / 2.0
    obstacles = (
        ((length_m / 2.0, -half_width), (length_m / 2.0, half_thickness)),
        ((length_m / 2.0, half_width), (length_m / 2.0, half_thickness)),
    )
    return segments, obstacles
