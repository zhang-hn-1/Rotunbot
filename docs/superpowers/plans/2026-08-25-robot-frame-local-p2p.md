# Robot-frame Local P2P Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with review checkpoints.

**Goal:** Add and validate a new single-frame Robot-frame Local P2P task without changing the existing P2P or maze tasks.

**Architecture:** Add a separate `rotunbot_local_goal` environment derived from the compatible Rotunbot target task. Keep the episode target in world coordinates internally, convert its planar displacement to the current yaw frame for a 17-D observation, and train with a standard compact PPO actor. Add unit tests for the pure coordinate/observation contract and GPU smoke/evaluation scripts for the simulator boundary.

**Tech Stack:** Python 3.8, PyTorch 1.10.0+cu113, Isaac Gym Preview 4, legged_gym task registry, standard rsl_rl PPO, unittest.

**Spec:** `docs/superpowers/specs/2026-08-25-robot-frame-local-p2p-design.md`

## Global Constraints

- Preserve the existing `rotunbot_target_repro`, `rotunbot_local_p2p`, maze code, and all existing checkpoints.
- Use explicit Robot-frame local goal `[dx, dy]`; do not expose world target, world XY, or absolute yaw to the new actor.
- Use exactly one current observation frame with 17 values; do not put the goal in history.
- Keep action semantics and control gains unchanged.
- Use `only_positive_rewards=False`, local arrival radius `0.35 m`, and no stop-speed requirement.
- Do not add maze, planner, depth, CNN, SRU, goal-switch, collision, heading, yaw, or global-goal reward terms.
- Run tests and smoke with `/home/jason/legged_gym/.venv` and GPU where Isaac Gym is required.

---

### Task 1: Pure local-goal transformation and observation contract

**Files:**
- Create: `legged_gym/envs/rotunbot/local_goal_p2p/local_goal_utils.py`
- Create: `legged_gym/tests/test_local_goal_p2p.py`

**Interfaces:**
- `world_to_robot_xy(world_delta: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor` accepts `[..., 2]` displacement and `[...]` yaw and returns `[..., 2]` local displacement.
- `build_local_observation(local_goal, base_lin_vel, base_ang_vel, projected_gravity, dof_pos, dof_vel, previous_actions, max_goal_distance) -> torch.Tensor` returns shape `[N, 17]` in the spec order.

- [ ] **Step 1: Write the failing tests**

  Add tests for yaw `0, 90, 180, -90` and forward/left/right/back vectors, asserting the inverse rotation results. Add a test that constructs a two-row observation and asserts shape 17, exact field ordering, normalized goal, and previous action in the final two positions.

- [ ] **Step 2: Run the tests and verify the expected failure**

  Run:

  ```bash
  PATH=/home/jason/legged_gym/.venv/bin:$PATH \
  /home/jason/legged_gym/.venv/bin/python -m unittest \
  legged_gym.tests.test_local_goal_p2p -v
  ```

  Expected: import or attribute failure because the new utility module does not yet exist.

- [ ] **Step 3: Implement the minimal pure Torch utilities**

  Implement inverse planar rotation and strict concatenation in the declared order. Validate matching batch sizes and positive `max_goal_distance`; raise `ValueError` for invalid shapes or scale.

- [ ] **Step 4: Run the tests and verify they pass**

  Re-run the exact unittest command and require all tests to pass with exit code 0.

- [ ] **Step 5: Run the existing independent tests**

  Run:

  ```bash
  PATH=/home/jason/legged_gym/.venv/bin:$PATH \
  /home/jason/legged_gym/.venv/bin/python -m unittest \
  legged_gym.tests.test_oracle_local_subgoal \
  legged_gym.tests.test_rotunbot_maze_map -v
  ```

  Require no regressions.

### Task 2: New environment configuration and registration

**Files:**
- Create: `legged_gym/envs/rotunbot/local_goal_p2p/__init__.py`
- Create: `legged_gym/envs/rotunbot/local_goal_p2p/rotunbot_local_goal_config.py`
- Create: `legged_gym/envs/rotunbot/local_goal_p2p/rotunbot_local_goal.py`
- Modify: `legged_gym/envs/__init__.py`
- Modify: `legged_gym/envs/rotunbot/__init__.py` if required by the existing import layout
- Test: `legged_gym/tests/test_local_goal_p2p.py`

**Interfaces:**
- Register task name `rotunbot_local_goal` with class `RotunbotLocalGoal`, `RotunbotLocalGoalCfg`, and `RotunbotLocalGoalCfgPPO`.
- Environment exposes `local_goal`, `world_goal`, `success_buf`, `time_out_buf`, `terminal_goal_dist`, `clip_count`, and `action_count`.

- [ ] **Step 1: Add registration/import tests before implementation**

  Add a registry test that imports `legged_gym.envs`, asserts `rotunbot_local_goal` is present, and asserts the configured observation size is 17.

- [ ] **Step 2: Run the registration test to verify it fails**

  Run the focused unittest and confirm it fails because the task is not registered.

- [ ] **Step 3: Implement config and environment**

  Subclass the compatible Rotunbot reproduction task only for asset/control/reset infrastructure. Override observation construction, target sampling, callback, termination, and the four local rewards. Set `frame_stack=1`, `num_single_obs=17`, `num_observations=17`, `num_privileged_obs=17`, episode length 6 s, and standard PPO actor `[256,128,64]`. Use explicit stage ranges A/B/C and no target curriculum or command resampling. Compute current local goal from the latched world target and current yaw each observation step. Feed `self.last_actions`, not the current action, as the final observation fields. Track raw-action clip ratio before the base action clipping is applied.

- [ ] **Step 4: Run registration and pure tests**

  Re-run the focused test file and the existing independent tests. Require exit code 0.

### Task 3: GPU environment smoke and data-flow assertions

**Files:**
- Create: `legged_gym/scripts/smoke_local_goal_p2p.py`
- Modify: `legged_gym/tests/test_local_goal_p2p.py`

**Interfaces:**
- Smoke creates `rotunbot_local_goal` with two GPU environments and checks reset/step output, observation shape, latched target, local-goal invariance under equal robot-frame goals, success predicate, and finite reward.

- [ ] **Step 1: Add assertions and run the smoke command before implementation**

  Add the expected assertions to a callable smoke function, run it before the new environment exists, and confirm the failure is caused by missing task registration rather than an unrelated import error.

- [ ] **Step 2: Implement the smoke script**

  Use `task_registry.make_env` with `--headless --sim_device=cuda:0 --rl_device=cuda:0`, disable noise/domain randomization, set two environments to equal local goals at different world positions/yaws, and print observation/reward/done/clip metrics.

- [ ] **Step 3: Run the GPU smoke**

  Run:

  ```bash
  PATH=/home/jason/legged_gym/.venv/bin:$PATH CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  /home/jason/legged_gym/.venv/bin/python legged_gym/scripts/smoke_local_goal_p2p.py \
  --headless --sim_device=cuda:0 --rl_device=cuda:0
  ```

  Require exit code 0, `obs_shape=(2,17)`, finite values, and no unexpected reset before timeout/reach.

### Task 4: Training and evaluation tooling

**Files:**
- Create: `legged_gym/scripts/train_local_goal_p2p.py`
- Create: `legged_gym/scripts/evaluate_local_goal_p2p.py`
- Modify: `legged_gym/envs/rotunbot/local_goal_p2p/rotunbot_local_goal_config.py` only if CLI stage overrides require explicit config fields

**Interfaces:**
- Training accepts `LOCAL_GOAL_STAGE=A|B|C`, `LOCAL_GOAL_NUM_ENVS`, `LOCAL_GOAL_MAX_ITERATIONS`, and `LOCAL_GOAL_CHECKPOINT` for continuation.
- Evaluation accepts a checkpoint path, uses 500 fixed episodes by default, and writes JSON containing aggregate metrics, per distance/bearing/yaw group metrics, clip ratio, and raw `d_min`/final distance summaries.

- [ ] **Step 1: Add evaluator contract tests**

  Test JSON aggregation on a small deterministic synthetic record set, including success rate, divergence rate, near-miss bins, and maximum yaw-group gap.

- [ ] **Step 2: Run the evaluator contract tests and verify they fail**

  Confirm the aggregation module/functions are absent or incomplete.

- [ ] **Step 3: Implement training and evaluation**

  Training uses standard PPO and saves periodic checkpoints without loading the incompatible old DWL checkpoint. Evaluation fixes random seeds and enumerates distance/bearing/world-pose cases while preserving one target per episode. It must fail with a nonzero exit code when the formal Gate is not met.

- [ ] **Step 4: Run tooling tests and compile checks**

  Run the focused unit tests and:

  ```bash
  PATH=/home/jason/legged_gym/.venv/bin:$PATH \
  /home/jason/legged_gym/.venv/bin/python -m py_compile \
  legged_gym/envs/rotunbot/local_goal_p2p/*.py \
  legged_gym/scripts/train_local_goal_p2p.py \
  legged_gym/scripts/evaluate_local_goal_p2p.py
  ```

### Task 5: Stage A GPU training and formal baseline evaluation

**Files:**
- Create: `logs/rotunbot_local_goal_p2p/<run>/` generated checkpoints and metrics
- Modify: none unless a verified failure identifies a root-cause bug

**Interfaces:**
- Stage A trains distance 0.5–2.0 m and bearing -45°–45°.
- Stage B and C are not started until Stage A smoke/evaluation artifacts are valid.

- [ ] **Step 1: Run a short Stage A training smoke**

  Start with a small GPU run and verify checkpoints are written, raw/clip metrics are finite, and the environment does not silently resample goals.

- [ ] **Step 2: Run fixed Stage A evaluation**

  Use the new evaluator and record per-yaw groups. Do not claim a Gate pass unless all thresholds are directly reported.

- [ ] **Step 3: Decide progression from evidence**

  If Stage A is below target, inspect observation/action/termination metrics before changing reward or network. If it passes, continue to Stage B with the checkpoint and then Stage C using the same fixed evaluation protocol.

- [ ] **Step 4: Stop before Goal Switch/Maze**

  Do not add continuous goal switching, walls, Oracle BFS, depth, CNN, SRU, or high-level PPO in this implementation cycle.

## Verification checklist

- [ ] Pure geometry and observation contract tests pass.
- [ ] Existing Oracle/map tests pass.
- [ ] GPU smoke reports `[N,17]` observations and finite rewards.
- [ ] New actor is standard compact PPO and does not load the old DWL checkpoint.
- [ ] Evaluation reports success, divergence, near miss, yaw-group gap, and action clipping.
- [ ] No old P2P or maze file has been modified except task registration imports.
