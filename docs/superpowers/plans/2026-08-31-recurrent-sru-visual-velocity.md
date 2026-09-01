# Recurrent SRU Visual Velocity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair and verify the real Isaac Gym depth channel, then introduce a true high-level-time recurrent SRU policy that learns to emit executable `(v_cmd, w_cmd)` through the frozen V62 stack, without starting V2 or maze curricula.

**Architecture:** Keep the existing Depth + robot-frame goal + proprioception + previous command observation contract and frozen V62 command chain. Replace only the stateless length-one SRU call with an explicit actor hidden state updated once per 5 Hz macro decision; carry per-environment hidden states through rollout collection, reset done environments, and train on ordered sequences with masks. Build and validate a ground-truth V1 velocity teacher before imitation or PPO.

**Tech Stack:** Python 3.8, PyTorch 1.10/CUDA 11.3, Isaac Gym Preview 4, custom DWL PPO runner/storage, unittest, existing CSV/JSON evaluation artifacts.

**Spec:** `/home/jason/.codex/attachments/d2600584-4afe-46f6-900a-deb10814e74b/pasted-text.txt`

## Global Constraints

- Keep `SRU -> (v_cmd,w_cmd) -> Frozen V62 -> actuator`; do not add a waypoint policy or direct actuator output.
- Keep the V62 tracker, Transition Manager, Governor, Feasible Projection, and actuator mapping frozen unless a reproducible bottom-level defect is found.
- Do not begin L-shaped, double-turn, S-shaped, complex-maze, V2, or later curricula in this plan.
- Do not begin recurrent formal training until real IMAGE_DEPTH passes the three physical sanity tests.
- Preserve `repeat=10` and the approximately 5 Hz high-level decision rate.
- Treat the existing stateless V1 checkpoint and its 6m Gate FAIL as historical baseline evidence, not a recurrent checkpoint.
- Do not convert `-Inf` to a finite value and call the camera repaired without raw-pipeline evidence.
- Keep training and evaluation Isaac Gym simulations in separate processes.

---

### Task 1: Capture the real IMAGE_DEPTH failure with a minimal diagnostic

**Files:**
- Create: `legged_gym/scripts/audit_v1_depth_physical_sanity.py`
- Modify: `legged_gym/envs/rotunbot/maze/rotunbot_maze_camera.py`
- Test: `legged_gym/tests/test_v1_depth_pipeline.py`

**Interfaces:**
- Consumes: V1 camera configuration, `DepthCameraMixin`, Isaac Gym `IMAGE_DEPTH` tensor APIs.
- Produces: JSON/CSV records for raw and normalized depth, finite ratios, camera/sim configuration, and the exact render/access call sequence.

- [x] Write failing tests for raw-value classification and a three-distance sanity-result validator: valid converted wall distance must satisfy `d_0.5 < d_2.0 < d_5.0`, while all encoder tensors must be finite.
- [x] Run the focused depth tests and confirm the new validator is absent/fails.
- [x] Implement the diagnostic with a one-wall scene or the smallest existing V1 scene, explicitly executing `fetch_results`, `step_graphics`, `render_all_camera_sensors`, `start_access_image_tensors`, tensor read, and `end_access_image_tensors` with access scoped by `try/finally`.
- [x] Record both GPU tensor and CPU `get_camera_image` observations, camera handles, `graphics_device_id`, `enable_tensors`, near/far plane, pose, and headless setting; do not silently fall back.
- [x] Run the diagnostic at 0.5m, 2.0m, and 5.0m. The reproducible cause was an incorrect diagnostic Y-axis quaternion; production V1 identity pose sees the wall and preserves the central depth ordering.
- [x] Commit the diagnostic and tests only after the raw evidence is reproducible.

### Task 2: Repair the camera lifecycle or prove a simulator limitation

**Files:**
- Modify: `legged_gym/envs/rotunbot/maze/rotunbot_maze_camera.py`
- Modify: `legged_gym/envs/rotunbot/direct_velocity/rotunbot_direct_velocity.py`
- Modify: `legged_gym/envs/rotunbot/visual_corridor_v1/rotunbot_visual_corridor_v1_config.py`
- Test: `legged_gym/tests/test_v1_depth_pipeline.py`

**Interfaces:**
- Consumes: Task 1 raw evidence.
- Produces: A real IMAGE_DEPTH capture path whose normalized observation has finite ratio 100%, or an explicit blocked report with no false PASS.

- [x] Add failing lifecycle tests using a fake Gym object that assert graphics/render/start/end ordering and exactly one `end_access_image_tensors` for every successful or failed access.
- [x] Run those tests to establish the pre-fix behavior.
- [x] Implement the evidence-supported render synchronization and explicit signed-depth conversion, retaining raw audit fields.
- [x] Run the fake lifecycle tests and fallback normalization tests.
- [x] Run the real one-wall 0.5/2/5m audit in Isaac Gym: production identity pose gives monotonic converted center depth and 100% finite encoder input.
- [ ] If real raw output remains all `-Inf`, record the exact blocking cause and stop before recurrent training; do not mask it in policy input.

### Task 3: Define the recurrent actor ABI and per-environment hidden state

**Files:**
- Modify: `legged_gym/dwl/actor_critic_depth.py`
- Modify: `legged_gym/dwl/actor_critic_direct_velocity.py`
- Test: `legged_gym/tests/test_v1_sru_state_audit.py`

**Interfaces:**
- Consumes: finite real-depth observation and current direct-velocity policy dimensions.
- Produces: `act(obs, hidden_states, masks) -> action`, `evaluate(..., hidden_states, masks)`, `get_hidden_states()`, and explicit hidden shape `[num_layers, num_envs, hidden_dim]` or the documented equivalent.

- [x] Write failing tests proving hidden carry, batch isolation, and done-mask reset behavior.
- [x] Run the tests and confirm the current `is_recurrent=False` implementation fails them.
- [x] Extend `SpatialRecurrentUnit.forward(sequence, hidden=None, masks=None)` to carry a supplied hidden state and honor per-step masks.
- [x] Make `ActorCriticDirectVelocity.is_recurrent=True`; update actor inference/training calls to consume and return actor hidden state while retaining the frozen velocity head and V62 action interface.
- [x] Add previous actual velocity through an explicit V1 ABI change: actor 273 -> 275 and critic 19 -> 21, sampled at the 5 Hz command boundary, with migration tests for the old 272/18 and 273/19 layouts.
- [x] Run state tests and a deterministic two-step recurrence test.

### Task 4: Make DWL rollout storage and PPO sequence-aware

**Files:**
- Modify: `legged_gym/dwl/ppo_dwl.py`
- Modify: `legged_gym/dwl/rollout_storage_dwl.py`
- Modify: `legged_gym/dwl/on_policy_runner_dwl.py`
- Test: `legged_gym/tests/test_v1_recurrent_rollout.py`

**Interfaces:**
- Consumes: recurrent actor ABI from Task 3 and 5 Hz macro transitions from the current runner.
- Produces: ordered sequence minibatches carrying initial hidden state, per-step done masks, sequence length, batch size, and hidden shape metadata.

- [x] Write tests with two environments and a short rollout proving sequence order, episode-boundary masks, and initial hidden-state storage.
- [x] Confirmed the flat generator cannot satisfy the recurrent sequence contract.
- [x] Store hidden states at macro-transition boundaries and preserve time before sequence construction.
- [x] Implement an ordered `[T, B, obs]` generator with initial `[B, hidden]` state and `[T, B]` masks.
- [x] Pass ordered sequences/masks through PPO and log recurrent sequence metadata.
- [x] Run recurrent rollout tests and a one-update CPU synthetic gradient test.

### Task 5: Build and audit the V1 velocity teacher

**Files:**
- Create: `legged_gym/navigation/v1_velocity_teacher.py`
- Create: `legged_gym/scripts/eval_v1_velocity_teacher.py`
- Modify: `legged_gym/navigation/direct_velocity.py` only if a shared projection helper is required and tests preserve behavior.
- Test: `legged_gym/tests/test_v1_velocity_teacher.py`

**Interfaces:**
- Consumes: robot-frame `goal_xy_robot`, current actual velocity, obstacle/clearance distance, and measured reliable V62 command bounds.
- Produces: `teacher_velocity_command(goal_xy_robot, actual_velocity, obstacle_distance, config) -> [v_teacher, w_teacher]` plus raw/requested/applied projection diagnostics.

- [x] Write tests for forward/side goal behavior, speed reduction at large bearing/near obstacle, finite outputs, and command bounds.
- [x] Set teacher limits from the current reliable executed V62 command domain (`0.25 m/s`, `0.10 rad/s`, `R=2.0 m`, envelope 1.0).
- [x] Implement a bounded proportional-heading teacher with explicit minimum-turn-radius-compatible speed reduction and no reverse output.
- [x] Test raw teacher commands through the unchanged Feasible Projection and record bounded projection correction diagnostics.
- [x] Run the final teacher-only 1.0m×100, 1.5m×100, 2.0m×100, and 2.5m×100 evaluations with fixed goals and episode-level CSV/JSON artifacts. Final seed-2026 results are 100/100, 100/100, 100/100, and 100/100 with zero collisions/timeouts; the first 2.5m attempt (93/100) is retained as a diagnostic failure artifact.
- [x] Do not start imitation if any short-distance teacher set fails its declared success/safety threshold; the teacher was corrected and re-gated before dataset work.

### Task 6: Add teacher-label collection and recurrent imitation warm start

**Files:**
- Create: `legged_gym/navigation/v1_teacher_dataset.py`
- Create: `legged_gym/scripts/train_sru_visual_corridor_v1_imitation.py`
- Modify: `legged_gym/envs/rotunbot/visual_corridor_v1/rotunbot_visual_corridor_v1.py` for label/debug fields only.
- Test: `legged_gym/tests/test_v1_teacher_dataset.py`

**Interfaces:**
- Consumes: finite real-depth sequences, teacher commands, recurrent hidden initialization, and fixed 5 Hz macro timing.
- Produces: deterministic sequence dataset and recurrent checkpoint with teacher-command loss and command/projection audit metrics.

- [x] Write failing tests for dataset schema, chronological sequence grouping, done-boundary splitting, and reproducible seed ordering.
- [x] Implement collection with preprocessed real IMAGE_DEPTH, goal, proprioception, previous command, previous actual velocity, teacher command, done, and metadata fields.
- [ ] Implement recurrent imitation over ordered sequences with masked MSE/Huber velocity loss and explicit command-domain checks.
- [ ] Run dataset/schema tests and a synthetic imitation convergence test.
- [ ] Run a short real-depth imitation smoke only after Task 2 PASS; record sequence metadata and teacher-vs-student command correction.

### Task 7: Closed-loop recurrent validation, then PPO fine-tuning and V1 Gate

**Files:**
- Modify: `legged_gym/scripts/eval_sru_visual_corridor_v1.py`
- Modify: `legged_gym/scripts/run_sru_visual_corridor_v1_curriculum.py`
- Modify: `legged_gym/scripts/train_sru_visual_corridor_v1.py`
- Modify: `docs/sru_visual_corridor_curriculum_report.md`
- Test: `legged_gym/tests/test_v1_recurrent_evaluation.py`

**Interfaces:**
- Consumes: recurrent imitation checkpoint and isolated-process training/evaluation orchestration.
- Produces: short-distance closed-loop evidence, then PPO fine-tuned checkpoint, repeated independent 30+30 evaluations, and fixed 6m×100 formal Gate artifacts.

- [ ] Write failing evaluator tests proving hidden state is reset per episode, carried across 5 Hz decisions, and not mixed across parallel environments.
- [ ] Implement recurrent evaluator calls with explicit hidden-state initialization/reset and hidden diagnostics in episode CSV.
- [ ] Run short closed-loop validation at 1.0/1.5/2.0/2.5m and reject checkpoints with hidden leakage, nonfinite input, or command-domain violations.
- [ ] Only after closed-loop validation, launch isolated-process PPO fine-tuning with recurrent sequence logs and model-only checkpoint provenance.
- [ ] Re-run the performance-gated current/next 30+30 evaluator and promote only on the existing measured thresholds.
- [ ] Run fixed 6m×100 with exact initial distance 6.0m, record SPL using the existing definition, and update V1 status from evidence.
- [ ] Keep V2/L/double-turn/S/maze blocked unless V1 formally passes.

## Verification Checklist

- [ ] Real IMAGE_DEPTH raw and converted one-wall measurements at 0.5/2/5m are monotonic and finite after conversion.
- [ ] Final encoder input is finite ratio 100%, with raw invalid/no-return counts separately reported.
- [ ] Recurrent hidden state has explicit environment dimension; done masks reset only done environments.
- [ ] PPO minibatches preserve time order and report sequence metadata.
- [ ] Teacher passes all short-distance standalone evaluations before imitation.
- [ ] Student closed-loop evaluation uses the frozen V62 chain and no actuator-output shortcut.
- [ ] V1 30+30 and fixed 6m×100 results are independent, episode-level, reproducible artifacts.
- [ ] No V2 or maze curriculum starts in this plan.
