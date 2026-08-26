# Rotunbot procedural maze

The legacy `rotunbot_maze` folder contains Python implementations rather than
an OBJ/STL map.  The code is integrated here as a separate task so it does not
overwrite the existing target, obstacle, ball-display, or checkpoint-evaluation
implementations.

## Project layout

- `legged_gym/maps/rotunbot_maze.py`: Isaac-Gym-independent map generation,
  reachability, and grid-to-world conversion.
- `legged_gym/envs/rotunbot/maze/rotunbot_maze.py`: Isaac Gym wall actors,
  reachable target sampling, and the Rotunbot task.
- `legged_gym/envs/rotunbot/maze/rotunbot_maze_config.py`: map, robot, and PPO
  settings for the registered `rotunbot_maze` task.
- `legged_gym/scripts/smoke_test_maze.py`: one-ball physics smoke test that does
  not require a trained checkpoint.
- `legged_gym/planners/oracle_local_subgoal.py`: map-only shortest-path Oracle
  that produces latched local subgoals for Phase 1.
- `legged_gym/scripts/smoke_test_oracle_planner.py`: hierarchical planner smoke
  test at 2 Hz; its analytic adapter is replaceable by a frozen blind P2P policy.
- `legged_gym/scripts/teleop_maze.py`: interactive WASD control using the same
  normalized action interface that a learned policy uses.
- `legged_gym/tests/test_rotunbot_maze_map.py`: deterministic map unit tests.

## Validation commands

Run the map-only tests on any Python environment with NumPy:

```bash
PYTHONPATH=. python legged_gym/tests/test_rotunbot_maze_map.py
```

Run one ball in the real Isaac Gym simulator:

```bash
python -m legged_gym.scripts.smoke_test_maze --headless
```

Drive the ball interactively in the viewer:

```bash
python -m legged_gym.scripts.teleop_maze
```

Keyboard controls are W/S for forward/reverse, A/D for left/right, Space to
stop, R to reset, and Esc to quit.  Commands remain active after a key press;
press W or S again to straighten the steering.

Remove `--headless` to inspect the maze in the viewer.  Override the default
300 simulation steps with the `ROTUNBOT_MAZE_SMOKE_STEPS` environment variable.

Run the Phase-1 hierarchical smoke test with one environment:

```bash
python -m legged_gym.scripts.smoke_test_oracle_planner --headless --sim_device cuda:0 --rl_device cuda:0
```

The test keeps the final maze goal separate from the latched local controller
goal, so reaching a waypoint does not reset the episode.

Train the dedicated task without changing the existing obstacle experiment:

```bash
python -m legged_gym.scripts.train --task rotunbot_maze --headless
```

## Obstacle-avoidance training status

The task already supplies reachable goal sampling and a vectorized collision
penalty covering every maze wall.  Set `maze.terminate_on_collision = True`
when collision should end an episode.

For general obstacle avoidance rather than memorizing this seeded map, add a
robot-relative perception vector (for example 16-32 normalized lidar rays), a
fixed-size randomized wall/map pool, and randomized connected start/goal pairs
before a long training run.
