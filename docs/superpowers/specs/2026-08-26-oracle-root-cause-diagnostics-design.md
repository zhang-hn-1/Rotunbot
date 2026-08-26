# Oracle Root-Cause Diagnostics Design

## Goal

Locate where Frozen uniform-4150 execution collides during Oracle Raw and Reachability evaluation, without changing the frozen policy, planner, waypoint thresholds, reachability envelope, turn-aware threshold, maze, or episode manifest.

## Scope and invariants

- The diagnostic Raw100 run uses the accepted Raw v1 manifest and checkpoint.
- Existing Oracle control behavior is observationally instrumented only.
- `POST_SWITCH_COLLISION` uses the fixed primary window `steps_since_goal_switch <= 10`; raw step counts remain in every collision record.
- Collision classification has one deterministic primary class plus overlapping boolean labels.
- Wall clearance is measured from the robot exterior to wall surfaces; center-to-center distances are not used as clearance.
- No Planner, P2P, Reachability, Depth/CNN/SRU/PPO, or Teacher Dataset changes are in scope.

## Diagnostic geometry

At every step, derive `current_cell`, the active `waypoint_cell`, and the next BFS cell from the current actual pose and active path. Compute local-goal distance and bearing from the local goal vector, the robot yaw from the actual root orientation, and cross-track error as the point-to-segment distance from robot XY to the current BFS segment between the current and next BFS cell centers.

For each wall rectangle, compute the XY distance from robot center to the rectangle, then subtract `robot_collision_radius`. Log both `nearest_wall_surface_distance` (center-to-surface distance) and `robot_clearance` (exterior clearance). A non-positive clearance is geometrically consistent with the maze collision condition.

## Collision labels

The primary class is selected deterministically in this order:

1. `FINAL_APPROACH_COLLISION`: the active Oracle phase is `FINAL_APPROACH`.
2. `POST_SWITCH_COLLISION`: `steps_since_goal_switch <= 10`.
3. `CORNER_CUT_COLLISION`: the active BFS segment has a turn of at least 45 degrees toward its next segment and the current waypoint has not been reached.
4. `STRAIGHT_CORRIDOR_COLLISION`: the active BFS segment has no qualifying turn and the robot is inside the current cell corridor geometry.
5. `APPROACH_COLLISION`: active NAVIGATE phase with an active local goal, outside the post-switch window.
6. `OTHER`: fallback.

Every collision also stores `is_final_approach`, `is_post_switch`, `is_corner`, `is_straight_corridor`, and `is_approach`. The summary includes primary counts/rates and post-switch sensitivity counts for windows `<=5`, `<=10`, and `<=20` steps.

## Control Gates

- C1 runs a fixed straight corridor case and reports success, collision, cross-track error, minimum robot clearance, and trajectory.
- C2 runs a fixed single 90-degree corner with initial forward speeds 0, 0.2, 0.4, and 0.6 m/s, preserving Frozen 4150 control after initialization.
- C3 runs a fixed wall-detour waypoint sequence and reports the same trajectory and collision diagnostics.

All three gates are evaluation-only and do not alter the accepted Raw100 protocol.
