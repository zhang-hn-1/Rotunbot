# T-junction MVP Design

## Goal

Extend the frozen V1 visual SRU velocity-policy chain with a minimal fixed-width
T-junction whose left and right tasks differ only by `goal_xy_robot`. The MVP
must prove that the student selects the goal-conditioned branch while retaining
the already-passed Straight and L behavior.

## Design

The new geometry is a symmetric 3.0 m wide T: the robot approaches along the
positive-x stem, the front wall terminates at the junction, and equal-length
left/right branches extend along y. `T_LEFT` and `T_RIGHT` reuse the same wall
segments, spawn pose, yaw, episode horizon, depth configuration, and seed; only
the global exit goal and episode-local waypoint sequence change. The existing
`V1WaypointManager` remains responsible only for local waypoint progression.
Terminal success continues to use the global goal, while the active local
waypoint is installed through `set_observation_goal_world`.

The teacher emits the existing normalized high-level action, which is passed
through the existing velocity projection and Frozen V62 controller. The
collector asserts `depth_backend_actual == "isaacgym"`, stores the existing V1
dataset schema with scenario/goal metadata, and writes an audit JSON. A pure
metrics module computes branch choice, turn completion, exit reach, paired
counterfactual consistency, and normal/zero/swapped-goal ablation results so
these checks remain unit-testable without IsaacGym.

Mixed imitation reuses the existing Straight, L, and T datasets with integer
resampling weights 1:3:5, preserving T=16 and the recurrent ABI. The T student
evaluator uses real depth, recurrent done resets, local waypoint goals, and the
existing V62 command path. It writes per-episode traces, failure histograms,
counterfactual pairs, ablations, and one machine-readable gate.

## Scope exclusions

No S corridor, multiple junctions, cross junction, planner, encoder/SRU/V62
changes, RL/PPO fine-tuning, ROS, or sim-to-real work is included.

## Acceptance

Teacher: 20 episodes per side, each side at least 95% success, zero collision
and wrong turn. Student: 20 episodes per side, each side at least 95% success,
zero collision, at most 5% timeout, at least 95% turn completion, and at least
95% paired goal consistency. Straight/L regressions must remain at least 95%
success with zero collision. Any non-ROS pytest failure blocks the final
`T_JUNCTION_MVP_PASS` verdict.
