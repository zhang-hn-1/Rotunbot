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

1. `evaluate_single_local_goal.py`: conservative 0.5/1.0/1.5 m goals at
   0/+-30/+-45 degrees.
2. `evaluate_goal_switch.py`: continuous straight, L, S, rectangle, and sharp
   direction sequences without episode reset.
3. `evaluate_oracle_maze.py`: seed-0 ground-truth maze, reachable goals, and
   closed-loop replanning from the actual robot pose.
4. `measure_reachability.py`: fixed-action motion measurements and a
   data-derived radial envelope.
5. `collect_oracle_depth_dataset.py`: provider-neutral depth collection only
   after Oracle closed-loop validation; no depth model is trained here.

Pure geometry and serialization tests do not require Isaac Gym. GPU commands
must use `/home/jason/legged_gym/.venv/bin/python` on the target machine.
