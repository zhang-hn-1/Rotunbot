# Depth-Aware Local Executor Design

## 1. Goal

Refactor the current Rotunbot navigation experiments into a separate depth-aware
local-navigation task. The new task combines the existing maze/BFS geometry,
depth-camera sensing, robot-frame local-goal representation, and the existing
two-dimensional Rotunbot control interface without changing the historical
tasks or their checkpoints.

The first policy version (V0) is newly initialized and intentionally simple:
Depth CNN + state/local-goal fusion + MLP actor. SRU, attention, recurrent
memory, residual modulation, and old checkpoint weights are deferred until the
V0 behavior is understood.

## 2. System Boundary

The system is divided into an external navigation layer and a local execution
layer.

```text
maze / ground-truth map / BFS
            ↓
feasible active local waypoint
            ↓
world_goal_to_robot_xy()
            ↓
depth image + local goal + proprioception + previous action
            ↓
Depth-Aware Local Executor
            ↓
existing 2-D Rotunbot action
            ↓
existing Rotunbot planar velocity/position controller
```

The actor must not receive absolute world XY, absolute yaw, or ground-truth
map occupancy. During the first full-maze phase, the map and BFS planner may
generate local goals outside the actor. This preserves the separation between
planning and local visual execution.

The current action semantics are explicitly planar, not differential-drive
steering: action 0 commands body-X motion and action 1 commands body-Y motion.
The measured yaw response is negligible for the current controller, so the
design does not assume that action 1 turns the robot.

The existing `rotunbot_target_depth`, `rotunbot_maze`, `rotunbot_local_p2p`,
and `rotunbot_local_goal` tasks remain available as baselines. The refactor
adds a new task rather than changing their observation contracts or loading
their checkpoints.

## 3. New Task

The new registered task is:

```text
rotunbot_maze_local_depth
```

It reuses the current Rotunbot physics, maze actors, collision geometry, BFS
utilities, diagnostics, and evaluation conventions. The environment maintains
three distinct goal states:

```text
global_goal_xy_world       final episode destination
active_local_goal_xy_world current BFS waypoint
active_local_goal_xy_robot robot-frame target exposed to the actor
```

The actor sees only `active_local_goal_xy_robot`. BFS may update the active
waypoint without changing the final global goal.

### V0 observation contract

Each observation contains one frame with 272 values:

```text
0:3     body-frame projected gravity
3:6     body-frame linear velocity
6:9     body-frame angular velocity
9       actuated joint position
10:12   two joint velocities
12:14   robot-frame local goal XY
14:16   previous action
16:272  normalized depth image, 8×32
```

The depth convention is fixed:

```text
0.0 = near plane
1.0 = far plane / open space
```

The observation must be finite, and the depth slice must remain in `[0, 1]`.
No frame-stack compatibility with the old depth task is required for V0.

## 4. Depth Sensor Layer

Depth sensing is extracted from the current monolithic target-depth task into
a reusable `DepthCameraMixin`.

The mixin owns:

- camera creation and attachment to `base_link`;
- Isaac Gym GPU depth tensor acquisition;
- camera render/update access;
- depth normalization and resizing;
- configurable Gaussian noise, dropout, and quantization;
- deterministic ray/AABB fallback for headless smoke tests and fast debugging.

The formal visual-navigation evaluation must include the real Isaac Gym
`IMAGE_DEPTH` path. The fallback is not treated as evidence that the physical
camera path works; it is a controlled diagnostic source.

The default V0 camera shape is width 32, height 8, horizontal FOV 105 degrees,
near plane 0.05 m, far plane 8.0 m, and local pose `(0.42, 0.0, 0.0)` with an
identity quaternion.

## 5. Local Goal Interface

The local goal conversion is a pure, independently testable operation:

```python
def world_goal_to_robot_xy(robot_xy, robot_yaw, goal_xy_world):
    delta = goal_xy_world - robot_xy
    local_x = cos(yaw) * delta_x + sin(yaw) * delta_y
    local_y = -sin(yaw) * delta_x + cos(yaw) * delta_y
    return local_x, local_y
```

This interface is the only goal representation exposed to the V0 actor. Goal
sampling and BFS waypoint selection remain environment/planner responsibilities.
The active waypoint is selected as a feasible local waypoint: its distance,
lateral displacement, and bearing must be compatible with the current
executor's measured planar motion and the forward camera coverage. The first
stage uses a conservative reachable range (`g_x` in 0.4–1.5 m and
`|g_y|` ≤ 0.6 m) before widening the distribution.

## 6. V0 Policy

The new `ActorCriticDepthLocal` follows the existing PPO actor-critic API so
it can be used by the repository's training and inference tooling:

- depth branch: three stride-2 convolution layers with ELU activations;
- state branch: a small MLP for the 16 non-image values;
- fusion: concatenate 64-dimensional image and state features;
- actor: `128 → 256 → 128 → 2` MLP;
- critic: compatible value head using the configured critic observation;
- action distribution: existing diagonal Normal PPO interface.

V0 explicitly excludes SRU, LSTM, GRU, self-attention, cross-attention,
frame-stack recurrence, residual paths, and initialization from any historical
checkpoint such as `model_4150.pt`.

The existing action semantics and torque/control pipeline are unchanged. The
projected gravity vector is used instead of roll/pitch/yaw, so absolute yaw is
not leaked through the orientation fields.

## 7. Reward and Termination

The new task uses local-navigation rewards:

```text
local progress
+ local-goal reach
+ near-wall penalty
- collision
- action rate / action smoothness
```

Local progress is the decrease in active local-goal distance between steps.
When the active waypoint changes, the previous-distance baseline is reset and
that transition contributes zero progress; cross-waypoint distance is never
treated as a real motion signal. Local waypoint reach uses a radius of 0.35 m.

The clearance term is a near-wall penalty only:

```text
-max(0, safety_distance - nearest_wall_distance)
```

It becomes zero beyond the safety distance, remains a training reward only,
and cannot compete indefinitely with progress by rewarding stationary behavior.
The initial reward scales are local progress 3.0, local reach 20.0, wall
penalty 0.5, collision -20.0, and action-rate penalty -0.01. The action-rate
penalty remains deliberately small so the weak body-Y response is not
suppressed.

For Stage 0–3, reaching the active local goal is episode success and causes a
reset. For Stage 4, the environment distinguishes:

```text
waypoint_reached   advance the active waypoint and continue the episode
global_goal_reached terminal episode success
```

Timeout, instability, out-of-bounds, collision, waypoint reach, and global
success are separately reported. All reward terms must be finite and use the
same collision definition as the maze environment.

## 8. Implementation Boundaries

Expected new or modified files:

```text
legged_gym/envs/rotunbot/maze/rotunbot_maze_camera.py
legged_gym/envs/rotunbot/maze/rotunbot_maze_local_depth.py
legged_gym/envs/rotunbot/maze/rotunbot_maze_local_depth_config.py
legged_gym/dwl/actor_critic_depth_local.py
legged_gym/navigation/local_goal.py
legged_gym/scripts/evaluate_depth_local.py
legged_gym/scripts/evaluate_depth_local_maze.py
legged_gym/tests/test_local_goal.py
legged_gym/tests/test_depth_camera_math.py
legged_gym/tests/test_depth_local_observation.py
legged_gym/tests/test_actor_critic_depth_local.py
legged_gym/tests/test_depth_local_rewards.py
legged_gym/tests/test_waypoint_switching.py
legged_gym/envs/__init__.py
```

The current target-depth camera code is the implementation source for the
sensor behavior, but its maze/goal/task-specific logic must not be copied into
the reusable mixin. Existing baseline files are modified only when required
for registration or a shared, behavior-preserving interface.

## 9. Verification Strategy

The first implementation batch must pass before longer training:

1. pure robot-frame goal transform tests;
2. fallback depth math, shape, range, and finite-value tests;
3. exact 272-value observation contract tests;
4. actor inference shape `(N, 2)` and critic shape `(N, 1)` tests;
5. reward sign, local-progress, reach, collision, and termination tests;
6. waypoint-switching consistency: no cross-waypoint progress spike and no
   Stage-4 episode reset at an intermediate waypoint;
7. one-iteration headless PPO smoke with a newly initialized policy;
8. a check that no absolute world XY or map occupancy enters the actor input;
9. a check that no historical model checkpoint is loaded by the new task;
10. evaluation metadata explicitly records `depth_backend` as `fallback` or
    `isaacgym`, and formal visual evaluation rejects `fallback`.

No later curriculum stage starts before the previous gate passes.

## 10. Training Stages and Gates

The staged validation sequence is:

| Stage | Scenario | Gate |
|---|---|---|
| 0 | open space, reachable local targets | local success rate ≥ 90% |
| 1 | bounded straight corridor | 4/4 collision-free, cross-track target ≤ 0.40 m |
| 2 | 90-degree corner with feasible waypoints | success ≥ 80%, collision ≤ 20% |
| 3 | planner-provided blocked-path detours | local success ≥ 80%, collision ≤ 10% |
| 4 | full maze with BFS local goals | global success ≥ 80%, collision ≤ 10% |

Stage 0 begins with `g_x` in 0.4–1.5 m and `|g_y|` ≤ 0.6 m, then expands
distance and lateral range only after its gate passes. Stage 2 and Stage 3 do
not require a single-frame V0 policy to solve an ambiguous multi-way detour:
the planner supplies a feasible intermediate waypoint, while the executor
uses depth to reach it safely. Real-camera validation repeats representative
corridor, corner, and maze cases and compares real-camera results against the
fallback path. SRU is only considered as V1 after V0 metrics and failure modes
are documented.

## 11. Error Handling and Non-Goals

If the real camera is requested but its tensors cannot be created or rendered,
the formal visual evaluation must fail clearly rather than silently using the
fallback. Fast headless smoke tests may explicitly select the fallback.
`--headless` and `graphics_device_id = -1` are not treated as synonyms: a
headless run with graphics enabled can still use `IMAGE_DEPTH`, while a
headless run without a graphics device must use the fallback. Every evaluation
record includes the selected backend.

Waypoint switching is an explicit state transition. On a switch, the new
active goal is latched, the previous-distance baseline is initialized to the
new goal distance, and the intermediate waypoint does not terminate a full
maze episode.

This refactor does not delete old environments, convert the repository to a
new simulator, upload model files, or introduce a new high-level learned
planner. The first deliverable is a testable V0 local executor and its
evaluation path.
