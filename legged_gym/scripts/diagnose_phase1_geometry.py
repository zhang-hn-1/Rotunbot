"""Phase1-1: pure Robot-frame <-> World-frame geometry closure test."""

import math


CASES = (
    ("Forward", (1.0, 0.0)),
    ("Left", (0.0, 1.0)),
    ("Right", (0.0, -1.0)),
    ("Back", (-1.0, 0.0)),
)
YAWS_DEG = (0.0, 90.0, 180.0, -90.0)


def local_to_world(local, yaw):
    x, y = local
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * x - s * y, s * x + c * y)


def world_to_local(world, yaw):
    x, y = world
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * x + s * y, -s * x + c * y)


def main():
    max_error = 0.0
    for yaw_deg in YAWS_DEG:
        yaw = math.radians(yaw_deg)
        for name, local in CASES:
            world = local_to_world(local, yaw)
            recovered = world_to_local(world, yaw)
            error = math.hypot(recovered[0] - local[0], recovered[1] - local[1])
            max_error = max(max_error, error)
            print(
                f"yaw={yaw_deg:>4.0f} local={name:<7} "
                f"q_local=({local[0]: .6f},{local[1]: .6f}) "
                f"world=({world[0]: .6f},{world[1]: .6f}) "
                f"recovered=({recovered[0]: .6f},{recovered[1]: .6f}) "
                f"error={error:.3e}"
            )
    print(f"GEOMETRY_MAX_ERROR={max_error:.9e}")
    if max_error >= 1.0e-5:
        raise SystemExit("geometry closure failed")
    print("GEOMETRY PASS")


if __name__ == "__main__":
    main()
