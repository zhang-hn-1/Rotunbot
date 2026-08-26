# Robot-frame Local P2P Design

## Goal

Create a new low-level controller task whose policy receives the current local goal in the robot frame, rather than requiring the policy to infer a relative goal from world-frame coordinates, world position, and yaw.

The existing `rotunbot_target_repro` and `rotunbot_local_p2p` tasks remain unchanged as baselines.

## Architecture

The new task stores a fixed episode target in world XY only as simulator-side state. At every observation update it converts the target displacement into the current robot yaw frame and exposes only the normalized local displacement to the actor. The policy is single-frame and proprioceptive: local goal, body-frame linear/angular velocity, projected gravity, joint state, and previous action.

The first task has no maze, no planner, no goal switching, and no stop-speed requirement. It uses the existing Rotunbot action interface: action 0 drives body-X motion and action 1 drives body-Y motion. The task is registered separately as `rotunbot_local_goal` and uses the standard PPO runner with a compact `17 -> 256 -> 128 -> 64 -> 2` actor.

## Observation contract

The observation has exactly 17 values in this order:

```text
[local_goal_x / 3.0,
 local_goal_y / 3.0,
 body_lin_vel_x, body_lin_vel_y, body_lin_vel_z,
 body_ang_vel_x, body_ang_vel_y, body_ang_vel_z,
 projected_gravity_x, projected_gravity_y, projected_gravity_z,
 dof_pos_0, dof_pos_1,
 dof_vel_0, dof_vel_1,
 previous_action_0, previous_action_1]
```

The local goal is recomputed from the latched world target and current robot yaw. It is not stored in an observation history. The first release uses one frame only.

## Training contract

Episodes sample one target at reset and terminate on local distance `< 0.35 m`, timeout, instability, or out-of-bounds. Timeout is 6 s. The reward scales are progress `1.0`, reach `5.0`, time `-0.01`, and action smoothness `-0.001`, with `only_positive_rewards=False`.

Curriculum stages are explicit and selectable:

| Stage | Distance | Bearing |
|---|---:|---:|
| A | 0.5–2.0 m | -45°–45° |
| B | 0.5–2.5 m | -90°–90° |
| C | 0.5–3.0 m | -180°–180° |

Training randomizes world XY and yaw while keeping the local target distribution independent of world pose. Goal switching is excluded from this first controller release.

## Evaluation contract

The evaluator uses a fixed 500-episode matrix over distances `[0.5, 1.0, 1.5, 2.0, 3.0]`, bearings `[0, 45, 90, -45, -90, 180]`, multiple world positions, and multiple initial yaws. It reports reach, timeout, divergence, near miss, minimum distance, final distance, completion time, action clip ratio, and world-yaw group differences.

The formal Gate is reach `>=95%`, divergence `<=2%`, and maximum success-rate difference between world-yaw groups `<=5 percentage points`.

## Non-goals

This change does not modify the old P2P checkpoint interface, maze termination, Oracle BFS, depth observations, CNN/SRU, or high-level PPO.
