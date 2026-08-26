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

Pure geometry and serialization tests do not require Isaac Gym. GPU commands
must use `/home/jason/legged_gym/.venv/bin/python` on the target machine.
