# Hierarchical Navigation Baseline

This branch implements non-learning hierarchical navigation infrastructure
around one immutable low-level skill.

## Frozen low-level controller

- Branch base: `0f6bde79e7f57e458d26907016368e6e4b7ee6b`
- Policy task: `rotunbot_target_repro`
- Checkpoint: `logs/rotunbot_target_repro/Aug16_02-57-06_uniform_t1_long500_from3809/model_4150.pt`
- Observation: 19 values per frame, 20-frame actor history
- Action: 2 values (velocity target and position target)
- Executor: `DIRECT_VP_TORQUE`
- Accepted gains: velocity `100`, position `600`
- Formal success: distance `<= 0.20 m` and linear speed `<= 0.10 m/s`
- Local waypoint reach: distance `<= 0.35 m`; speed is not required.

The checkpoint must be provided explicitly to runtime evaluators. A missing
checkpoint is an error; evaluators never create or evaluate a random policy.

## Hierarchical contract

```text
Global goal
  -> ground-truth Oracle BFS
  -> robot-frame local waypoint
  -> measured reachability filter
  -> world-frame temporary goal
  -> frozen uniform 4150
  -> Rotunbot
```

The robot-frame waypoint is never passed directly to the actor. The adapter
converts it to the absolute world-frame target channels expected by the
original P2P observation. A waypoint switch changes only `env.commands[:, :2]`
and rewrites the target channels of the existing history frames in place; it
does not append an extra observation frame.
The wrapper suppresses the original success-reset side effect only for an
intermediate waypoint; it does not change the P2P success calculation, reward,
controller, network, history dimensions, or action interface.

The Oracle scheduler has two explicit phases. In `NAVIGATE`, it executes the
next BFS cell and replans only after the local waypoint is reached. Once the
measured robot cell equals the global goal cell, it enters `FINAL_APPROACH`,
sets the temporary goal to the exact global goal, and stops local waypoint
replanning/counting. A dynamic exit is recorded as `final_approach_escape`
rather than silently restarting ordinary navigation.

Turn-aware switching is an independent opt-in evaluator policy. Turns below
45 degrees retain the distance-only 0.35 m switch; turns at or above 45
degrees additionally require speed <= 0.30 m/s. The Frozen 4150 controller is
unchanged.

## Gates

1. `evaluate_single_local_goal.py`: single-goal executor check using the
   distance-only `0.35 m` local-waypoint criterion.
2. `evaluate_single_local_goal_coverage.py`: the formal 4x8 distance/bearing
   coverage matrix, with three repetitions per case.
3. `evaluate_goal_switch.py`: continuous straight, L, S, rectangle, and sharp
   direction sequences without episode reset. It records the action pair at
   each switch boundary and `average_waypoint_completion_time_s`.
4. `evaluate_oracle_maze.py --smoke`: ten raw Oracle episodes with no
   reachability filter. This gate checks planner, coordinate, state-history,
   and checkpoint/control integrity separately from physical failures.
5. `evaluate_oracle_maze.py`: fixed-manifest Raw 100 episodes. Its summary
   reports Global SR, collision/timeout/waypoint-failure rates, local-waypoint
   reach, actual/BFS path lengths, Maze SPL, completion time, waypoint count,
   and the required failure-reason histogram. The Maze protocol budget is
   `120 s`; the original uniform-4150 P2P protocol remains `60 s`.
6. `measure_reachability.py`: fixed-action motion measurements and a
   data-derived radial envelope. This is deferred until Raw 100 is complete.
7. `collect_oracle_depth_dataset.py`: provider-neutral depth collection only
   after Oracle closed-loop validation; no depth model is trained here.
8. `evaluate_oracle_diagnostics.py`: Raw or Reachability diagnostic replay.
   It logs the active/current/next BFS cells, robot-frame goal geometry,
   switch age, turn/reachability state, raw and filtered goals, clip ratio,
   exterior wall clearance, and cross-track error. Collision rows use the
   reset-before-reset terminal pose cached by the maze environment.
9. `evaluate_control_diagnostics.py`: non-training C1 straight corridor, C2
   single 90-degree corner with initial speeds 0/0.2/0.4/0.6 m/s, and C3
   fixed wall detour. These gates use the Frozen 4150 policy and do not alter
   the Oracle planner.

## Collision diagnostic definitions

`nearest_wall_surface_distance` is the distance from robot center to the
nearest axis-aligned wall surface. `robot_clearance` subtracts the configured
robot collision radius, so non-positive clearance corresponds to geometric
wall overlap. `reachability_clip_ratio` is the fractional reduction in local
goal norm after deterministic filtering.

Each collision has one primary class in this fixed order:
`FINAL_APPROACH_COLLISION`, `POST_SWITCH_COLLISION` (switch age <= 10
simulation steps), `CORNER_CUT_COLLISION`, `STRAIGHT_CORRIDOR_COLLISION`,
`APPROACH_COLLISION`, `OTHER`. The same record also stores overlapping boolean
labels for final approach, post-switch, corner, straight corridor, and
approach. Summary windows <=5, <=10, and <=20 preserve sensitivity to the
post-switch threshold without rerunning simulation.

Pure geometry and serialization tests do not require Isaac Gym. GPU commands
must use `/home/jason/legged_gym/.venv/bin/python` on the target machine.
