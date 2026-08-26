# Rotunbot Hierarchical Visual Navigation Design

## Goal

Build every non-learning component required for hierarchical navigation while
keeping the accepted uniform-4150 point-to-point policy completely frozen.

## Baseline and branch policy

- Development branch: `hierarchical-visual-nav`.
- Baseline commit: `0f6bde79e7f57e458d26907016368e6e4b7ee6b`.
- Frozen low-level controller: `uniform 4150`, loaded from
  `Aug16_02-57-06_uniform_t1_long500_from3809/model_4150.pt` when available.
- The branch must not import or depend on the `depth_camera` branch, the SRU
  policy, a new CNN, PPO updates, or a new local-P2P policy.

## Frozen policy contract

The adapter and planner must produce the same world-frame absolute target that
the accepted policy originally observes. The policy contract is therefore:

- 19 values per observation frame.
- 20-frame actor history.
- Two actions: velocity target and position target.
- Existing `DIRECT_VP_TORQUE` executor and its accepted gains.
- Existing formal success rule: distance at most 0.20 m and linear speed at
  most 0.10 m/s within the episode limit.

No hierarchical module may pass a robot-frame local goal directly to the
policy actor.

## Components

### Local goal geometry

Create a dependency-light adapter with explicit NumPy interfaces:

```python
local_to_world(robot_xy, robot_yaw, local_goal_xy) -> world_goal_xy
world_to_local(robot_xy, robot_yaw, world_goal_xy) -> local_goal_xy
```

The forward transform is `robot_xy + R(robot_yaw) @ local_goal_xy`. The inverse
uses the transpose of the same rotation matrix. Inputs are finite two-element
vectors and the functions return `float64` two-element vectors.

### Oracle BFS planner

Create a pure-Python/NumPy planner that consumes the existing occupancy-grid
representation and returns a four-neighbor cell path. It must reject invalid,
wall, or unreachable start/goal cells. A waypoint selector returns the next
cell center, without lookahead in the first version. Existing deterministic
maze generation and cell-center conversion remain the source of truth.

### Reachability envelope

Create serializable reachability samples for fixed action pairs. The envelope
stores action bounds and measured body-frame displacement/velocity metadata,
but does not alter controller gains. The first deterministic filter clips a
local goal to the configured safe radial boundary while preserving its bearing.

### Evaluation gates

Implement independent scripts for:

1. Single local-goal execution using the frozen policy and no environment reset
   until the goal, timeout, or instability condition.
2. Continuous waypoint switching without clearing robot state, observation
   history, previous action, or the episode.
3. Oracle-maze execution from a fixed-seed maze with random reachable goals.
4. Physical reachability sweeps over the requested action pairs.

Every script writes machine-readable JSON/CSV summaries and trajectory records.
The Oracle evaluator separates local-policy, goal-switch, collision, timeout,
waypoint, bounds, and instability failures.

### Dataset interface

Add a provider-neutral closed-loop collector. It receives a depth frame from a
future sensor provider and stores the frame together with pose, global goal,
robot-frame goal, oracle local waypoint, temporary world goal, previous
waypoint, collision state, timestamp, and episode identifier. It records only
actual closed-loop rollouts; it does not train or modify any model.

## Data flow

```text
Global goal
  -> Oracle BFS
  -> local waypoint in robot frame
  -> reachability filter
  -> Local Goal Adapter
  -> temporary world-frame goal
  -> frozen uniform-4150 observation/action loop
  -> robot state update
  -> replan or switch waypoint without reset
```

The global goal is fixed for one episode. A local waypoint update never marks
global success and never resets the environment. Only the final global goal
terminates the episode.

## Test and validation strategy

- Unit-test all requested yaw angles and forward/backward/left/right/diagonal
  local goals.
- Verify local-to-world-to-local round-trip error below `1e-10` in the pure
  geometry tests.
- Test BFS paths on the deterministic maze, including invalid and unreachable
  requests.
- Test reachability clipping at, inside, and outside the configured envelope.
- Run Python compilation and the existing map tests.
- On the provided machine, run GPU smoke/evaluation commands with
  `/home/jason/legged_gym/.venv/bin/python` when Isaac Gym is importable.
- Do not claim physical or GPU policy performance when the checkpoint or Isaac
  Gym runtime is unavailable.
