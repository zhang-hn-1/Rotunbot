# Depth-Aware Local Executor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a newly initialized depth-aware local-navigation task for Rotunbot that executes feasible robot-frame waypoints using depth, proprioception, and the existing planar 2-D action interface.

**Architecture:** Preserve the existing target-depth, maze, and local-goal tasks as baselines. Extract depth sensing into a reusable `DepthCameraMixin`, add a pure robot-frame goal transform, and create `rotunbot_maze_local_depth` with a V0 CNN+MLP actor. Ground-truth maze/BFS code stays outside the actor and supplies an active feasible waypoint; Stage 4 advances waypoints without terminating until the global goal is reached.

**Tech Stack:** Isaac Gym Preview 4, PyTorch, `legged_gym`, existing DWL/RSL-RL PPO components, CUDA, Python unittest.

**Spec:** `docs/superpowers/specs/2026-08-27-depth-aware-local-executor-design.md`

## Global Constraints

- Keep `rotunbot_target_depth`, `rotunbot_maze`, `rotunbot_local_p2p`, and `rotunbot_local_goal` behavior and checkpoint interfaces unchanged.
- Keep action 0 as body-X motion and action 1 as body-Y motion; do not model action 1 as steering.
- Expose only projected gravity, body-frame velocities, joint state, previous action, robot-frame local goal, and normalized depth to the V0 actor.
- Do not expose absolute world XY, absolute yaw, or ground-truth map occupancy to the V0 actor.
- Maintain separate `global_goal_xy_world`, `active_local_goal_xy_world`, and `active_local_goal_xy_robot` state.
- Set `only_positive_rewards = False`; negative progress, wall, collision, and action-rate rewards must reach the PPO total reward.
- Use an 8×32 depth image normalized to `[0, 1]`, with `0.0` near and `1.0` far/open space.
- V0 uses a newly initialized CNN+MLP policy with no SRU, LSTM, GRU, attention, residual modulation, or historical checkpoint weights.
- A real-camera evaluation must use Isaac Gym `IMAGE_DEPTH` and reject a fallback backend; fallback is allowed only when explicitly selected for smoke/debug runs.
- When a waypoint changes, reset the previous-distance baseline and emit zero cross-waypoint progress for that transition.
- No Stage N+1 training starts until the Stage N gate passes.

---

## File Map

### New files

- `legged_gym/envs/rotunbot/maze/rotunbot_maze_camera.py`: reusable camera creation, tensor capture, normalization, noise, and fallback sensing.
- `legged_gym/envs/rotunbot/maze/rotunbot_maze_local_depth.py`: V0 environment, local-goal state, observation, reward, termination, and waypoint advancement.
- `legged_gym/envs/rotunbot/maze/rotunbot_maze_local_depth_config.py`: V0 environment/camera/reward/stage and PPO configuration.
- `legged_gym/dwl/actor_critic_depth_local.py`: 272-input CNN+MLP actor-critic implementing the existing PPO API.
- `legged_gym/navigation/__init__.py`: navigation helper exports.
- `legged_gym/navigation/local_goal.py`: pure world-goal to robot-frame transformation.
- `legged_gym/scripts/evaluate_depth_local.py`: fixed open-space/corridor/corner evaluation.
- `legged_gym/scripts/evaluate_depth_local_maze.py`: full-maze BFS/local-waypoint evaluation.
- `legged_gym/tests/test_depth_camera_math.py`: pure depth normalization/fallback tests.
- `legged_gym/tests/test_depth_observability.py`: 105-degree camera side-obstacle observability diagnostic (Gate 1.5).
- `legged_gym/tests/test_local_goal.py`: robot-frame transform tests.
- `legged_gym/tests/test_depth_local_observation.py`: exact 272-value layout and leakage tests.
- `legged_gym/tests/test_actor_critic_depth_local.py`: actor/critic shape and finite-output tests.
- `legged_gym/tests/test_depth_local_rewards.py`: reward sign and local-progress tests.
- `legged_gym/tests/test_waypoint_switching.py`: waypoint transition and Stage-4 continuation tests.
- `legged_gym/tests/test_depth_local_evaluation.py`: evaluator aggregation and backend-policy tests.
- `legged_gym/scripts/smoke_depth_local.py`: two-environment fallback/camera and one-step smoke.
- `legged_gym/tests/test_depth_local_smoke.py`: smoke contract tests that do not require a long training run.

### Modified files

- `legged_gym/envs/__init__.py`: register `rotunbot_maze_local_depth`.
- `legged_gym/dwl/on_policy_runner_dwl.py`: import the new actor class so config-based policy lookup works.

Historical baseline files are not modified unless registration or a behavior-preserving shared hook requires it.

---

### Task 1: Extract and test the depth sensor layer

**Files:**
- Create: `legged_gym/envs/rotunbot/maze/rotunbot_maze_camera.py`
- Test: `legged_gym/tests/test_depth_camera_math.py`

**Interfaces:**
- `DepthCameraMixin._create_camera_sensors()` creates attached sensors when graphics are available.
- `DepthCameraMixin._init_camera_tensors()` binds Isaac Gym GPU image tensors.
- `DepthCameraMixin._get_depth_fallback_aabbs() -> tuple[torch.Tensor, torch.Tensor]` is supplied by the environment and returns obstacle centers and XY half-extents; the mixin never owns maze geometry.
- `DepthCameraMixin.capture_depth() -> torch.Tensor` returns `[N, 8, 32]` and records `depth_backend_requested` and `depth_backend_actual`.
- `DepthCameraMixin._apply_depth_noise(depth: torch.Tensor) -> torch.Tensor` preserves finite values and `[0, 1]` bounds.

- [ ] **Step 1: Write failing pure tests**

Test that normalized values clamp to `[0, 1]`, invalid/NaN camera values become far-plane values, and a synthetic symmetric corridor produces symmetric left/right columns. Test the geometry-provider interface and the separate requested/actual backend labels.

- [ ] **Step 2: Run the focused test before implementation**

Run:

```bash
/home/jason/legged_gym/.venv/bin/python -m unittest legged_gym.tests.test_depth_camera_math -v
```

Expected result: import failure because `DepthCameraMixin` does not yet exist.

- [ ] **Step 3: Extract the sensor-only code**

Move the camera lifecycle and fallback math currently embedded in `rotunbot_target_depth.py` into the mixin. Keep task-specific maze geometry out of the mixin. Add explicit modes:

```python
depth_backend_requested = "isaacgym"  # formal IMAGE_DEPTH path
depth_backend_requested = "fallback"  # explicit headless ray/AABB path
depth_backend_actual = "isaacgym" or "fallback"

def _get_depth_fallback_aabbs(self):
    """Return environment-owned obstacle centers and XY half-extents."""
    raise NotImplementedError
```

Do not silently switch from a requested real camera to fallback during formal evaluation; if `depth_backend_requested == "isaacgym"` and the actual path is unavailable, raise a clear `RuntimeError`.

- [ ] **Step 4: Run the focused tests and compile the mixin**

```bash
/home/jason/legged_gym/.venv/bin/python -m unittest legged_gym.tests.test_depth_camera_math -v
/home/jason/legged_gym/.venv/bin/python -m py_compile legged_gym/envs/rotunbot/maze/rotunbot_maze_camera.py
```

- [ ] **Step 5: Commit the sensor layer**

```bash
git add legged_gym/envs/rotunbot/maze/rotunbot_maze_camera.py legged_gym/tests/test_depth_camera_math.py
git commit -m "feat: extract reusable depth camera sensing"
```

### Task 2: Add the pure robot-frame local-goal interface

**Files:**
- Create: `legged_gym/navigation/__init__.py`
- Create: `legged_gym/navigation/local_goal.py`
- Test: `legged_gym/tests/test_local_goal.py`

**Interfaces:**
- `world_goal_to_robot_xy(robot_xy: torch.Tensor, robot_yaw: torch.Tensor, goal_xy_world: torch.Tensor) -> torch.Tensor` returns `[N, 2]`.

- [ ] **Step 1: Write failing transform tests**

```python
def test_forward_at_zero_yaw():
    out = world_goal_to_robot_xy(
        torch.tensor([[0.0, 0.0]]), torch.tensor([0.0]), torch.tensor([[2.0, 0.0]])
    )
    assert torch.allclose(out, torch.tensor([[2.0, 0.0]]), atol=1e-6)

def test_world_y_goal_is_robot_forward_at_ninety_degrees():
    out = world_goal_to_robot_xy(
        torch.tensor([[0.0, 0.0]]), torch.tensor([torch.pi / 2]), torch.tensor([[0.0, 2.0]])
    )
    assert torch.allclose(out, torch.tensor([[2.0, 0.0]]), atol=1e-5)
```

Add inverse checks for `0`, `90`, `180`, and `-90` degrees and reject mismatched shapes.

- [ ] **Step 2: Run the tests before implementation**

```bash
/home/jason/legged_gym/.venv/bin/python -m unittest legged_gym.tests.test_local_goal -v
```

- [ ] **Step 3: Implement the vectorized inverse planar rotation**

```python
delta = goal_xy_world - robot_xy
c, s = torch.cos(robot_yaw), torch.sin(robot_yaw)
return torch.stack((c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1]), dim=-1)
```

- [ ] **Step 4: Run tests and commit**

```bash
/home/jason/legged_gym/.venv/bin/python -m unittest legged_gym.tests.test_local_goal -v
git add legged_gym/navigation legged_gym/tests/test_local_goal.py
git commit -m "feat: add robot-frame local goal transform"
```

### Task 3: Implement the V0 local depth actor

**Files:**
- Create: `legged_gym/dwl/actor_critic_depth_local.py`
- Modify: `legged_gym/dwl/on_policy_runner_dwl.py`
- Test: `legged_gym/tests/test_actor_critic_depth_local.py`

**Interfaces:**
- Constructor accepts the existing runner arguments `num_short_obs`, `num_single_obs`, `num_critic_obs`, `num_actions` plus `depth_height=8`, `depth_width=32`, and PPO policy settings.
- `act`, `act_inference`, `evaluate`, `get_actions_log_prob`, `action_mean`, `action_std`, `entropy`, and `reset` match the existing actor-critic API.
- `act_inference(torch.zeros(4, 272))` returns `[4, 2]`; `evaluate(torch.zeros(4, 18))` returns `[4, 1]`.

- [ ] **Step 1: Write failing network contract tests**

Assert the observation split is state `[:16]` and depth `[16:]`, the output shapes are correct, outputs are finite, and construction rejects an observation size other than 272.

- [ ] **Step 2: Run the focused tests before implementation**

```bash
/home/jason/legged_gym/.venv/bin/python -m unittest legged_gym.tests.test_actor_critic_depth_local -v
```

- [ ] **Step 3: Implement V0**

Use three stride-2 convolutions `1→16→32→64`, an image projection to 64 dimensions, a state MLP `16→64→64`, fusion to 128 dimensions, and actor layers `128→256→128→2`. Keep the critic as a separate MLP over an 18-dimensional privileged state: the actor's clean 16-value state plus normalized nearest-wall clearance and collision flag. Use a newly initialized diagonal Normal action distribution and do not import or load any checkpoint.

- [ ] **Step 4: Run tests and compile**

```bash
/home/jason/legged_gym/.venv/bin/python -m unittest legged_gym.tests.test_actor_critic_depth_local -v
/home/jason/legged_gym/.venv/bin/python -m py_compile legged_gym/dwl/actor_critic_depth_local.py
```

- [ ] **Step 5: Commit the actor**

```bash
git add legged_gym/dwl/actor_critic_depth_local.py legged_gym/dwl/on_policy_runner_dwl.py legged_gym/tests/test_actor_critic_depth_local.py
git commit -m "feat: add V0 depth-aware local actor"
```

### Task 4: Create and register the new local-depth environment

**Files:**
- Create: `legged_gym/envs/rotunbot/maze/rotunbot_maze_local_depth.py`
- Create: `legged_gym/envs/rotunbot/maze/rotunbot_maze_local_depth_config.py`
- Modify: `legged_gym/envs/__init__.py`
- Test: `legged_gym/tests/test_depth_local_observation.py`

**Interfaces:**
- Task name: `rotunbot_maze_local_depth`.
- Environment state: `global_goal_xy_world`, `active_local_goal_xy_world`, `active_local_goal_xy_robot`, `waypoint_reached`, `global_goal_reached`, `waypoint_changed`, `prev_local_goal_dist`, `depth_backend_requested`, and `depth_backend_actual`.
- `get_observations()` returns `[N, 272]`; privileged observations return `[N, 18]`.

- [ ] **Step 1: Add failing registration and observation tests**

Test that the registry contains the new task, configures `num_observations == 272`, and defines the exact layout:

```text
0:3     projected gravity
3:6     body linear velocity
6:9     body angular velocity
9       actuated joint position
10:12   two joint velocities
12:14   robot-frame active local goal
14:16   previous action
16:272  depth image
```

Add a test that changing world position and absolute yaw while preserving the robot-frame goal does not place world XY or yaw in the actor observation.

- [ ] **Step 2: Run tests before implementation**

```bash
/home/jason/legged_gym/.venv/bin/python -m unittest legged_gym.tests.test_depth_local_observation -v
```

- [ ] **Step 3: Implement the environment and config**

Use `class RotunbotMazeLocalDepth(DepthCameraMixin, RotunbotMaze)`. Reuse the existing maze actors, collision geometry, physics, and planar controller. Maintain global and active local goals separately; for Stage 0–3 the active local goal is also the episode target, while Stage 4 uses the BFS waypoint state. Set `num_single_obs=272`, `num_short_obs=272`, `frame_stack=1`, and `num_observations=272` explicitly so the current runner's positional constructor contract is unambiguous. Keep the privileged observation at 18 values: the clean 16-value state plus normalized nearest-wall clearance and collision flag.

Configure camera width 32, height 8, FOV 105 degrees, near/far planes 0.05/8.0 m, explicit `depth_backend`, and `enable_camera_sensors_in_headless` independently from `--headless`. Use `projected_gravity` in the observation, never Euler yaw.

- [ ] **Step 4: Run pure tests, registry import, and compile**

```bash
python -m pytest -q legged_gym/tests/test_depth_local_observation.py
/home/jason/legged_gym/.venv/bin/python -m py_compile legged_gym/envs/rotunbot/maze/rotunbot_maze_local_depth.py legged_gym/envs/rotunbot/maze/rotunbot_maze_local_depth_config.py
```

- [ ] **Step 5: Commit the environment**

```bash
git add legged_gym/envs/rotunbot/maze/rotunbot_maze_local_depth.py legged_gym/envs/rotunbot/maze/rotunbot_maze_local_depth_config.py legged_gym/envs/__init__.py legged_gym/tests/test_depth_local_observation.py
git commit -m "feat: register depth-aware local navigation task"
```

### Task 5: Add local rewards, feasible waypoints, and switching semantics

**Files:**
- Modify: `legged_gym/envs/rotunbot/maze/rotunbot_maze_local_depth.py`
- Modify: `legged_gym/envs/rotunbot/maze/rotunbot_maze_local_depth_config.py`
- Test: `legged_gym/tests/test_depth_local_rewards.py`
- Test: `legged_gym/tests/test_waypoint_switching.py`

**Interfaces:**
- `_reward_local_progress() -> torch.Tensor` returns zero on a waypoint-change transition.
- `_reward_wall_penalty() -> torch.Tensor` returns `-relu(safety_distance - nearest_wall_distance)`.
- `_advance_active_waypoint()` updates the active world waypoint, recomputes the robot-frame goal, and resets the progress baseline.
- `check_termination()` separates local waypoint reach from terminal global success in Stage 4.

- [ ] **Step 1: Write failing reward and switching tests**

Cover: positive progress when the active goal gets closer; negative progress when it gets farther; zero progress immediately after switching A→B; finite near-wall penalty that becomes zero beyond the safety distance; and Stage-4 waypoint reach that advances without setting episode reset.

- [ ] **Step 2: Run focused tests before implementation**

```bash
   /home/jason/legged_gym/.venv/bin/python -m unittest legged_gym.tests.test_depth_local_rewards legged_gym.tests.test_waypoint_switching -v
```

- [ ] **Step 3: Implement the state machine and rewards**

Use local progress scale 3.0, local reach scale 20.0, wall penalty scale 0.5, collision scale -20.0, and action-rate scale -0.01. Set `only_positive_rewards=False` explicitly. For Stage 0–3, active-goal reach sets episode success. For Stage 4, compose timeout, instability, out-of-bounds, collision, and global-goal termination locally rather than calling the parent full-target success path; active-goal reach calls `_advance_active_waypoint()` and only `global_goal_reached` terminates successfully. Ensure the first transition after a switch has `waypoint_changed=True` and contributes zero progress.

Select active waypoints from BFS candidates only when their explicit feasibility fields (`distance_limit`, `lateral_limit`, `minimum_forward_component`, and configurable `bearing_limit`) are compatible with the measured planar action response and camera coverage. Leave `bearing_limit` configurable until Gate 1.5. Stage 0 starts with `g_x` 0.4–1.5 m and `|g_y|` ≤ 0.6 m.

- [ ] **Step 4: Run reward/switching tests and existing map tests**

```bash
   /home/jason/legged_gym/.venv/bin/python -m unittest legged_gym.tests.test_depth_local_rewards legged_gym.tests.test_waypoint_switching legged_gym.tests.test_oracle_local_subgoal legged_gym.tests.test_rotunbot_maze_map -v
```

- [ ] **Step 5: Commit reward and switching logic**

```bash
git add legged_gym/envs/rotunbot/maze/rotunbot_maze_local_depth.py legged_gym/envs/rotunbot/maze/rotunbot_maze_local_depth_config.py legged_gym/tests/test_depth_local_rewards.py legged_gym/tests/test_waypoint_switching.py
git commit -m "feat: add local navigation rewards and waypoint state"
```

### Task 6: Add training/evaluation tools and backend auditing

**Files:**
- Create: `legged_gym/scripts/evaluate_depth_local.py`
- Create: `legged_gym/scripts/evaluate_depth_local_maze.py`
- Test: `legged_gym/tests/test_depth_local_evaluation.py`

**Interfaces:**
- Both evaluators record `depth_backend_requested`, `depth_backend_actual`, local success, global success, waypoint reach count, collision, timeout, final distance, path length, and completion time.
- `evaluate_depth_local_maze.py` uses BFS outside the actor and passes only the current feasible waypoint to the environment.
- Formal real-camera evaluation exits nonzero unless `depth_backend == "isaacgym"`.

- [ ] **Step 1: Write evaluator aggregation tests**

Build a deterministic synthetic record set and assert correct local/global success rates, collision rate, waypoint count, and backend rejection.

- [ ] **Step 2: Run tests before implementation**

```bash
   /home/jason/legged_gym/.venv/bin/python -m unittest legged_gym.tests.test_depth_local_evaluation -v
```

- [ ] **Step 3: Implement fixed evaluation grids**

Use the Stage 0 open-space grid with random initial yaw and reachable local targets, then the corridor/corner cases, and finally a fixed maze manifest. Include both backend labels in every JSON report and reject fallback when formal real-camera evaluation is requested.

- [ ] **Step 4: Run evaluator unit tests and compile checks**

```bash
   /home/jason/legged_gym/.venv/bin/python -m unittest legged_gym.tests.test_depth_local_evaluation -v
   /home/jason/legged_gym/.venv/bin/python -m py_compile legged_gym/scripts/evaluate_depth_local.py legged_gym/scripts/evaluate_depth_local_maze.py
```

- [ ] **Step 5: Commit the evaluation tools**

```bash
git add legged_gym/scripts/evaluate_depth_local.py legged_gym/scripts/evaluate_depth_local_maze.py legged_gym/tests/test_depth_local_evaluation.py
git commit -m "feat: add depth local evaluation and backend audit"
```

### Task 7: Run V0 smoke and staged validation

**Files:**
- Create: `legged_gym/scripts/smoke_depth_local.py`
- Test: `legged_gym/tests/test_depth_local_smoke.py`
- Modify: none unless a test identifies a root-cause defect.

- [ ] **Step 1: Add a one-iteration smoke contract**

The fallback smoke must instantiate two environments, reset and step them, verify observations `[2, 272]`, finite rewards, depth bounds, action output `[2, 2]`, no historical checkpoint path, and `depth_backend_requested=fallback` with `depth_backend_actual=fallback`. A separate real-camera smoke must request `isaacgym`, require `depth_backend_actual=isaacgym`, and run with graphics enabled; `--headless` alone must not be treated as `graphics_device_id=-1`.

- [ ] **Step 2: Run the smoke before long training**

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /home/jason/legged_gym/.venv/bin/python legged_gym/scripts/smoke_depth_local.py --headless --sim_device=cuda:0 --rl_device=cuda:0 --depth-backend fallback
```

- [ ] **Step 3: Run one PPO update with a new policy**

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /home/jason/legged_gym/.venv/bin/python legged_gym/scripts/train.py --task rotunbot_maze_local_depth --headless --num_envs 2 --max_iterations 1
```

Require a finite rollout, finite PPO loss, and a newly created checkpoint. The new task must not load any existing model.

- [ ] **Step 4: Run staged evaluation**

Run Stage 0 first. Only after its local success gate of 90% passes, run Gate 1.5's side-obstacle observability diagnostic for the 105-degree camera. Then evaluate the bounded corridor, feasible-waypoint corner, planner-provided detour, and full maze. Validate representative cases with real Isaac Gym `IMAGE_DEPTH` after fallback smoke succeeds; formal camera evaluation rejects fallback.

- [ ] **Step 5: Commit smoke tooling and final verified state**

```bash
git add legged_gym/scripts/smoke_depth_local.py legged_gym/tests/test_depth_local_smoke.py
git commit -m "test: add depth-aware local executor smoke"
```

## Verification Checklist

- [ ] Existing baseline tasks import without changes to their observation contracts.
- [ ] Pure local-goal and depth math tests pass.
- [ ] The new environment returns exactly 272 actor values and 18 privileged critic values.
- [ ] Actor input contains projected gravity, not yaw.
- [ ] No absolute world XY or map occupancy reaches the actor.
- [ ] Waypoint switching produces no cross-waypoint progress spike.
- [ ] Intermediate Stage-4 waypoint reach does not reset the episode.
- [ ] Fallback and real-camera backends are explicitly recorded.
- [ ] Formal camera evaluation rejects fallback.
- [ ] Gate 1.5 records side-obstacle observability before Stage 2.
- [ ] One PPO update succeeds with a newly initialized V0 policy.
- [ ] No model or historical checkpoint is loaded by the new task.
