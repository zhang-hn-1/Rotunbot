# Corridor Curriculum Navigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify the serial navigation stack from frozen V62 spatial velocity control through Velocity Local Goal, Oracle corridor, depth, SRU, and Maze.

**Architecture:** Keep the existing V62 command path immutable after Phase A. Add a shared scenario/evaluation artifact layer, then train a high-level policy whose only action is normalized `(v_cmd, w_cmd)`; execute that command through a frozen V62 actor in the same simulator. Oracle and visual planners output only feasible local waypoints, and each later learner inherits only the prior stage's approved checkpoint.

**Tech Stack:** Python 3.8, PyTorch 1.10.0+cu113, Isaac Gym Preview 4, existing Legged Gym PPO/LH/DWL runners, NumPy, CSV/JSON, Matplotlib, unittest.

**Spec:** `docs/superpowers/specs/2026-08-29-corridor-curriculum-navigation-design.md` and the user source `/home/jason/.codex/attachments/d4f71254-2b34-4958-a9f9-c0f20812f163/pasted-text.txt`

## Global Constraints

- Work only in `/home/jason/.codex/worktrees/codex-corridor-curriculum-navigation` on `codex/corridor-curriculum-navigation`.
- Base commit is `6aa9f531f991ae101053b1ebe3973cb340daa0d1`.
- Use external frozen checkpoint `/home/jason/Rotunbot_SRU50_V62_SafeYaw_Final_Verified_20260829/model/model_150.pt`; require SHA256 `d7173fbbb113ab790d25b0587e82a73abd7ffad9ab2ed148387ba04084944f1b`.
- Use resolved V62 limits from `RotunbotVelSRU50SafeYawResidualV62Cfg`; do not create a second speed envelope.
- Runtime is physics `0.005 s`, decimation `4`, low-level `50 Hz`, upper command `5 Hz`, ten low-level steps per upper command.
- Never modify V62 control code or checkpoint to improve a later navigation Gate.
- Never load the historical actuator-action P2P checkpoint into the new Velocity Local Goal task.
- Every stage stops on Current Gate or Regression Gate failure.
- Run tests with `PATH=/home/jason/legged_gym/.venv/bin:$PATH /home/jason/legged_gym/.venv/bin/python`; the venv has no pytest, so existing `unittest` modules are the baseline runner unless a test dependency is explicitly installed and recorded.

---

### Task 0: Freeze Phase 0 baseline evidence

**Files:**
- Create: `docs/corridor_curriculum_navigation_baseline_audit.md`
- Create: `logs/corridor_curriculum_navigation_baseline.json`
- Read: `legged_gym/envs/rotunbot/vel_tracking/rotunbot_vel.py`
- Read: `legged_gym/envs/rotunbot/vel_tracking/feasible_transition_manager.py`
- Read: `docs/v62_feasible_transition_manager_report.md`

**Interfaces:**
- Consumes: Git commit, external checkpoint, resolved V62 config, existing test modules.
- Produces: immutable baseline record with HEAD, branch, working-tree state, checkpoint SHA, runtime frequencies, resolved domain, test command and test count.

- [ ] **Step 1: Record Git and checkpoint identity**

Run:

```bash
git status --short
git branch --show-current
git rev-parse HEAD
sha256sum /home/jason/Rotunbot_SRU50_V62_SafeYaw_Final_Verified_20260829/model/model_150.pt
```

Require branch `codex/corridor-curriculum-navigation`, HEAD `6aa9f53...`, and the required SHA.

- [ ] **Step 2: Resolve runtime configuration from Python**

Print `cfg.sim.dt`, `cfg.control.decimation`, `cfg.commands.upper_level_command_frequency_hz`, `maximum_linear_acceleration`, `maximum_yaw_acceleration`, `max_forward_speed`, `max_yaw_rate`, `minimum_turn_radius`, `feasible_envelope_fraction`, and `command_update_interval_steps(cfg.sim.dt * cfg.control.decimation, 5.0)`.

- [ ] **Step 3: Run the existing 94-test regression**

Run:

```bash
PATH=/home/jason/legged_gym/.venv/bin:$PATH \
/home/jason/legged_gym/.venv/bin/python -m unittest -v \
legged_gym.tests.test_rotunbot_velocity_tracking \
legged_gym.tests.test_feasible_transition_manager \
legged_gym.tests.test_vel_sru50_structured_random
```

Require `Ran 94 tests` and `OK`. If the test command fails before tests execute, record the dependency/import failure and repair only the test environment before continuing.

- [ ] **Step 4: Write the audit artifact**

Store the exact command outputs and resolved values in `logs/corridor_curriculum_navigation_baseline.json`; write the call chain and the 94/94 result in the Markdown report.

- [ ] **Step 5: Commit the baseline audit**

```bash
git add docs/corridor_curriculum_navigation_baseline_audit.md logs/corridor_curriculum_navigation_baseline.json
git commit -m "docs: record corridor navigation baseline audit"
```

Decision: Phase 0 PASS only if identity, SHA and 94-test regression are all verified.

### Task 1: Add shared corridor scenarios and artifact interfaces

**Files:**
- Create: `legged_gym/navigation/corridor_scenarios.py`
- Create: `legged_gym/navigation/corridor_artifacts.py`
- Create: `legged_gym/navigation/corridor_plotting.py`
- Create: `legged_gym/tests/test_corridor_scenarios.py`
- Create: `legged_gym/tests/test_corridor_artifacts.py`
- Modify: `legged_gym/navigation/__init__.py`

**Interfaces:**
- `CorridorScenario(family: str, width_m: float, segments: tuple, seed: int)` stores deterministic geometry.
- `make_straight_scenario(width_m, length_m, seed) -> CorridorScenario`.
- `make_l_scenario(width_m, straight_m, turn_radius_m, seed) -> CorridorScenario`.
- `make_double_turn_scenario(width_m, turn_radius_m, handedness: str, seed) -> CorridorScenario`.
- `GateResult.evaluate(summary: dict, current_rules: dict, regression_rules: dict) -> dict` returns explicit Current/Regression PASS booleans and failure reasons.
- `CheckpointMetadata.from_path(path, parent, stage, seed, iterations) -> dict` records SHA256 and immutable parent identity.
- `EpisodeLogger.write_episode(record: dict)`, `EpisodeLogger.write_summary(summary: dict)` and `EpisodeLogger.write_trajectory(rows)` produce `episodes.csv`, `summary.json` and per-episode trajectory CSV.
- `plot_corridor_artifacts(trajectory_csv, output_dir) -> tuple[str, ...]` writes XY, command-vs-actual and goal-distance PNGs.
- `replay_episode(artifact_dir, episode_id) -> ScenarioReplay` loads one deterministic episode record and its seed without changing the original artifact.

- [ ] **Step 1: Write failing geometry tests**

Assert deterministic seed replay, centerline continuity, expected segment lengths, left/right handedness, positive corridor width, and rejection of non-positive width/radius.

- [ ] **Step 2: Run the scenario tests and observe the expected missing-module failure**

```bash
.../python -m unittest -v legged_gym.tests.test_corridor_scenarios legged_gym.tests.test_corridor_artifacts
```

- [ ] **Step 3: Implement deterministic scenario dataclasses, GateResult, CheckpointMetadata and artifact schemas**

Use only NumPy/Python standard library. Required per-step fields are `episode_id`, `seed`, `scenario_family`, pose, raw/applied commands, actual velocities, goal distance, collision, rate/domain/hidden violations, transition activation, success, timeout, duration and path length.

- [ ] **Step 4: Implement the three required plots and deterministic failed-episode replay loader**

Use Matplotlib's non-interactive backend and write only `xy_trajectory.png`, `velocity_tracking.png` and `goal_distance.png`; replay must preserve `episode_id`, seed and scenario parameters.

- [ ] **Step 5: Run the focused tests and then the 94-test regression**

Require focused tests and all pre-existing tests to pass.

- [ ] **Step 6: Commit the shared infrastructure**

```bash
git add legged_gym/navigation legged_gym/tests/test_corridor_*.py
git commit -m "feat: add corridor scenarios and evaluation artifacts"
```

### Task 2: Add V62 corridor world and pose-based scripted command adapter

**Files:**
- Create: `legged_gym/navigation/v62_corridor_controller.py`
- Create: `legged_gym/scripts/evaluate_v62_corridor.py`
- Create: `legged_gym/scripts/smoke_test_v62_corridor.py`
- Create: `legged_gym/tests/test_v62_corridor_controller.py`

**Interfaces:**
- `PoseBasedCorridorController.update(pose_xy, yaw, scenario) -> np.ndarray([v_cmd, w_cmd])` uses straight/deceleration/turn/acceleration/straight states and pose/distance-to-corner transitions.
- `V62CorridorEvaluator.run(scenario, episodes, seed) -> dict` loads the frozen V62 task and records all artifact fields.
- `add_corridor_walls(env, scenario) -> list` creates static Isaac Gym wall actors; wall geometry is evaluation-only and does not modify V62.

- [ ] **Step 1: Write controller tests for state transitions**

Cover straight command, deceleration before a corner, positive and negative turns, turn exit acceleration, left→right and right→left reversal, command projection, and no fixed-time-only turn transition.

- [ ] **Step 2: Run the focused test and observe failure**

```bash
.../python -m unittest -v legged_gym.tests.test_v62_corridor_controller
```

- [ ] **Step 3: Implement the pose-based state machine**

Use `set_command_targets()` on upper ticks only. Use the resolved V62 command ranges and `project_velocity_commands`; never write `env.commands` as a bypass. Hold each target for `command_update_interval_steps(...)` low-level steps.

- [ ] **Step 4: Implement wall construction, collision accounting and trajectory logging**

Create plane plus static walls for each corridor scenario. Mark collision from contact forces and geometry bounds; record all command and actual velocity fields at 50 Hz.

- [ ] **Step 5: Add a one-episode GPU smoke test**

Run `smoke_test_v62_corridor.py` for a straight and an L scenario; require finite state, nonzero simulation progress, valid command shape and at least one logged transition for the L case.

- [ ] **Step 6: Run focused tests, smoke and 94-test regression**

Do not run formal A0 until all three checks pass.

- [ ] **Step 7: Commit the evaluator**

```bash
git add legged_gym/navigation/v62_corridor_controller.py legged_gym/scripts/evaluate_v62_corridor.py legged_gym/scripts/smoke_test_v62_corridor.py legged_gym/tests/test_v62_corridor_controller.py
git commit -m "feat: add V62 corridor evaluator and scripted controller"
```

### Task 3: Run Stage A0 straight corridor

**Files:**
- Modify: `legged_gym/scripts/evaluate_v62_corridor.py`
- Create: `logs/stage_a0/summary.json`, `logs/stage_a0/episodes.csv`, `logs/stage_a0/plots/`
- Modify: `docs/corridor_curriculum_navigation_report.md`

**Interfaces:**
- Uses `make_straight_scenario(width_m=2.0, length_m=5.0, seed=...)` and `V62CorridorEvaluator`.
- Produces fixed 20-episode A0 artifacts and PASS/FAIL decision.

- [ ] **Step 1: Add a deterministic A0 evaluator test**

Assert scenario width/length, command target shape, 20-episode seed list, and gate calculation `SR==1`, collision `0`, rate/domain/hidden violations `0`, lateral error `<0.20 m`, yaw error `<5°`.

- [ ] **Step 2: Run the test and observe failure before formal execution**

```bash
.../python -m unittest -v legged_gym.tests.test_v62_corridor_controller
```

- [ ] **Step 3: Run A0 on GPU 0**

Use the external checkpoint and task `rotunbot_vel_sru50_v62_feasible_transition_manager`; command approximately `v_cmd=0.10`, `w_cmd=0`; set the environment horizon from path length and speed rather than the default short episode.

- [ ] **Step 4: Verify A0 artifacts and gate**

Require `20/20` success, collision `0`, rate/domain/hidden violations `0`, final lateral error `<0.20 m`, final yaw error `<5°`, valid plots, and exact checkpoint metadata.

- [ ] **Step 5: Commit A0 evidence**

```bash
git add docs/corridor_curriculum_navigation_report.md
git commit -m "eval: record Stage A0 straight corridor gate"
```

Decision: A0 PASS is required before A1.

### Task 4: Run Stage A1 L corridor

**Files:**
- Modify: `legged_gym/navigation/v62_corridor_controller.py`
- Modify: `legged_gym/scripts/evaluate_v62_corridor.py`
- Create: `logs/stage_a1/summary.json`, `logs/stage_a1/episodes.csv`, `logs/stage_a1/plots/`

**Interfaces:**
- Uses `make_l_scenario(width_m=2.0, straight_m=3.0, turn_radius_m=2.0, seed=...)`.
- Controller exposes transition counters and turn entry/exit metrics.

- [ ] **Step 1: Add failing A1 pose-transition tests**

Assert the controller reduces forward command near the corner, commands a feasible nonzero yaw, switches turn direction from pose/yaw error, resumes forward motion after the turn, and increments transition activation.

- [ ] **Step 2: Implement the minimal A1 state transitions**

Use corridor-relative position, distance-to-corner and yaw error; never switch solely after a fixed number of steps.

- [ ] **Step 3: Run focused tests, smoke and 94-test regression**

- [ ] **Step 4: Run 20 fixed A1 episodes**

Require SR `>=95%`, collision `0`, rate/domain/hidden violations `0`, Transition activation `>0`, turn-exit yaw error `<7°`, and maximum centerline deviation `<0.30 m`.

- [ ] **Step 5: Commit A1 evidence and stop on FAIL**

```bash
git add docs/corridor_curriculum_navigation_report.md
git commit -m "eval: record Stage A1 L corridor gate"
```

### Task 5: Run Stage A2 double-turn corridor

**Files:**
- Modify: `legged_gym/scripts/evaluate_v62_corridor.py`
- Create: `logs/stage_a2/summary.json`, `logs/stage_a2/episodes.csv`, `logs/stage_a2/plots/`

**Interfaces:**
- Runs left→right and right→left scenarios with the same width `2.0 m`.
- Reports first/second turn yaw error, residual yaw before second turn, reversal completion time and all safety counters.

- [ ] **Step 1: Add failing A2 coverage tests**

Assert both handedness sequences are present and the second command changes yaw sign only through `set_command_targets()` and the Transition Manager.

- [ ] **Step 2: Run the focused test and implement missing scenario coverage**

- [ ] **Step 3: Run 30 fixed A2 episodes**

Require SR `>=90%`, collision `<=1`, rate/domain/hidden violations `0`, and valid reversal metrics.

- [ ] **Step 4: Run complete 94-test regression and archive A2 artifacts**

- [ ] **Step 5: Commit Freeze Point 1**

```bash
git add docs/corridor_curriculum_navigation_report.md
git commit -m "eval: freeze V62 spatial control after Stage A"
```

Proceed to Phase B only if A0, A1 and A2 are all PASS.

### Task 6: Build the Velocity Local Goal training environment

**Files:**
- Create: `legged_gym/envs/rotunbot/velocity_local_goal/velocity_local_goal.py`
- Create: `legged_gym/envs/rotunbot/velocity_local_goal/velocity_local_goal_config.py`
- Create: `legged_gym/navigation/frozen_v62_executor.py`
- Create: `legged_gym/scripts/train_velocity_local_goal.py`
- Create: `legged_gym/tests/test_velocity_local_goal.py`
- Modify: `legged_gym/envs/__init__.py`

**Interfaces:**
- `FrozenV62Executor(checkpoint_path, device, env)`: loads the V62 actor weights, exposes `step_held_command(target_commands, low_level_steps=10)`, and never exposes trainable parameters to the outer optimizer.
- `VelocityLocalGoalEnv.step(action) -> (obs, reward, done, info)`: maps normalized action `[a_v,a_w]` through the resolved V62 command envelope, calls `set_command_targets()`, runs ten frozen V62 steps, and returns the outer 5 Hz observation.
- `build_local_goal_observation(goal_x_body, goal_y_body, current_v, current_w, previous_v_cmd, previous_w_cmd) -> tensor[6]`.

- [ ] **Step 1: Write failing contract tests**

Cover six-value body-frame observation, no global coordinates, normalized two-action mapping, use of resolved command limits, ten-step hold, frozen executor parameters, reset of transition state, reward sign, success radius `0.15 m`, timeout and divergence termination.

- [ ] **Step 2: Run focused tests and observe missing implementation failure**

```bash
.../python -m unittest -v legged_gym.tests.test_velocity_local_goal
```

- [ ] **Step 3: Implement the frozen executor and task registration**

Instantiate the existing V62 actor architecture from its task config, load only `model_state_dict`, set eval mode, and feed its inference action into the existing V62 plant. Do not modify `rotunbot_vel.py`.

- [ ] **Step 4: Implement observation, action mapping and rewards**

Use progress, one-time success bonus, timeout/divergence penalties, small command-change penalties, collision interface, and `only_positive_rewards=False`. Log every reward term.

- [ ] **Step 5: Add GPU smoke test**

Run two environments for 20 outer steps; assert finite observations/rewards, action shape `(2,2)`, command hold of ten low-level steps, and zero bottom-level violations for a feasible command.

- [ ] **Step 6: Run focused tests, smoke and full regression**

- [ ] **Step 7: Commit the training environment**

```bash
git add legged_gym/envs/rotunbot/velocity_local_goal legged_gym/navigation/frozen_v62_executor.py legged_gym/scripts/train_velocity_local_goal.py legged_gym/tests/test_velocity_local_goal.py legged_gym/envs/__init__.py
git commit -m "feat: add velocity local goal task over frozen V62"
```

### Task 7: Train and gate B1, B2 and B3

**Files:**
- Modify: `legged_gym/scripts/train_velocity_local_goal.py`
- Create: `legged_gym/scripts/evaluate_velocity_local_goal.py`
- Create: `logs/phase_b/`
- Modify: `docs/corridor_curriculum_navigation_report.md`

**Interfaces:**
- `evaluate_velocity_local_goal(checkpoint, stage, seed_list) -> summary` uses fixed seeds `0,1,2` when no repository list exists and emits `summary.json`, `episodes.csv` and plots.
- Training stages use B1 from scratch, B2 from `B1_best.pt`, B3 from `B2_best.pt`.

- [ ] **Step 1: Add failing stage-distribution and gate tests**

Assert B1 `[0.5,1.0] m, ±10°`, B2 `[0.5,1.5] m, ±30°`, B3 `[0.5,2.0] m, ±45°`; assert 70/30, 70/20/10 replay ratios and exact gate thresholds.

- [ ] **Step 2: Run tests and implement stage configuration**

Use max iterations B1 `1000`, B2 `600`, B3 `800`; keep existing PPO algorithm/network defaults except observation/action/reward/task fields.

- [ ] **Step 3: Train B1 on GPU 0 and formally evaluate 100 episodes**

Require SR `>=95%`, divergence `<=2%`, timeout `<=5%`, and all bottom-level violations `0`; save checkpoint SHA and parent metadata.

- [ ] **Step 4: If B1 PASS, train B2 from B1_best and evaluate Current plus B1 Regression**

Require B2 SR `>=90%`, B1 SR `>=93%`, all safety violations `0`; otherwise stop.

- [ ] **Step 5: If B2 PASS, train B3 from B2_best and evaluate Current plus B1/B2 Regression**

Require B3 SR `>=90%`, B2 SR `>=90%`, B1 SR `>=93%`, all safety violations `0`; save `Local Goal Controller V1`.

- [ ] **Step 6: Run full regression, archive artifacts and commit Freeze Point 2 candidate**

Proceed to Phase C only if B3 and all regression gates PASS.

### Task 8: Add Oracle centerline waypoint corridor curriculum

**Files:**
- Create: `legged_gym/navigation/corridor_waypoint_oracle.py`
- Create: `legged_gym/tests/test_corridor_waypoint_oracle.py`
- Modify: `legged_gym/planners/oracle_local_subgoal.py`
- Create: `legged_gym/scripts/smoke_test_oracle_velocity_stack.py`

**Interfaces:**
- `CorridorWaypointOracle(scenario, local_distance_limit, bearing_limit_deg, lookahead_m=0.6).next_waypoint(pose) -> waypoint_world_xy`.
- Waypoints are continuous centerline points, shortened near turns, and clamped to B3 capability; the Oracle never supplies a direct actuator command.

- [ ] **Step 1: Write failing waypoint continuity and feasibility tests**

Assert centerline continuity, lookahead cap, turn-near shortening, distance limit, bearing limit, and no 90° discontinuous jump.

- [ ] **Step 2: Implement Oracle and run focused tests**

- [ ] **Step 3: Run the hierarchical GPU smoke test with the frozen B3/V62 stack**

Require planner tick at `2 Hz` in the smoke adapter, finite states, valid local waypoint, no direct actuator output, and one completed episode.

- [ ] **Step 4: Run full regression and commit Oracle curriculum infrastructure**

### Task 9: Train and gate C1 through C5 width curriculum

**Files:**
- Create: `legged_gym/scripts/train_oracle_corridor.py`
- Create: `legged_gym/scripts/evaluate_oracle_corridor.py`
- Create: `logs/phase_c/`
- Modify: `docs/corridor_curriculum_navigation_report.md`

**Interfaces:**
- C1/C2/C3/C4 load the previous best checkpoint and sample the exact mixtures from the spec.
- C5.1/C5.2/C5.3 are engineering gates at widths `1.8/1.6/1.4 m`; C5.4 at `1.2 m` is a capability-boundary experiment and reports, but does not require, 90% SR.

- [ ] **Step 1: Add failing curriculum and regression tests**

Assert C1 straight, C2 L, C3 double-turn, C4 S, and C5 widths; assert parent checkpoint identity and replay percentages.

- [ ] **Step 2: Implement C1-C4 training/evaluation and run tests**

Budgets are C1 `400`, C2 `600`, C3 `800`, C4 `1000`; each evaluation includes its required corridor gate plus previous-task regressions.

- [ ] **Step 3: Run C1 and stop unless Current/Regression gates PASS**

C1 requires corridor SR `>=95%`, collision `<=2%`, B3 `>=88%`, B2 `>=88%`, B1 `>=93%`, safety violations `0`.

- [ ] **Step 4: Run C2, C3 and C4 serially with gate checks**

C2 requires L SR `>=90%`, straight `>=95%`; C3 requires double-turn `>=90%`, L `>=90%`, straight `>=95%`; C4 requires S `>=90%`, double-turn `>=88%`, L `>=90%`, straight `>=95%`.

- [ ] **Step 5: Run C5.1, C5.2, C5.3 and C5.4 serially**

Require engineering thresholds `95/90/85%` for C5.1/C5.2/C5.3; record full boundary metrics for C5.4 including clearance and centerline error.

- [ ] **Step 6: Freeze the best Local Goal checkpoint and commit**

Create `Frozen Velocity Local Goal Policy` metadata with parent chain, SHA256 and Freeze Point 2 decision. Do not start D until all required C gates PASS.

### Task 10: Phase D Oracle random corridor validation

**Files:**
- Modify: `legged_gym/scripts/evaluate_oracle_corridor.py`
- Create: `logs/phase_d/`
- Modify: `docs/corridor_curriculum_navigation_report.md`

**Interfaces:**
- `sample_supported_random_corridor(seed) -> CorridorScenario` samples only length/width/turn/radius/start/goal ranges covered by C.
- Formal evaluator runs at least 200 episodes with the frozen Local Goal + V62 stack.

- [ ] **Step 1: Add failing supported-range and reproducibility tests**

- [ ] **Step 2: Implement random scenario generation and per-family summaries**

- [ ] **Step 3: Run 200 fixed-seed episodes and produce plots/CSV/JSON**

Require overall SR `>=90%`, collision `<=5%`, rate/domain/hidden violations `0`, and family-level results.

- [ ] **Step 4: Run all prior regression suites and commit Phase D PASS/FAIL**

Only D PASS permits visual planner work.

### Task 11: Phase E single-frame Depth Planner

**Files:**
- Create/modify: `legged_gym/envs/rotunbot/target_point/velocity_depth_planner_config.py`
- Create: `legged_gym/planners/depth_waypoint_student.py`
- Create: `legged_gym/scripts/train_depth_waypoint_student.py`
- Create: `legged_gym/scripts/evaluate_depth_waypoint_student.py`
- Create: `legged_gym/tests/test_depth_waypoint_student.py`
- Create: `logs/phase_e/`

**Interfaces:**
- `DepthWaypointStudent(depth_frame, proprioception, previous_waypoint, global_goal_direction) -> waypoint_body_xy`.
- `OracleWaypointTeacher(full_geometry, pose, goal) -> waypoint_body_xy`.
- Output clamp enforces Local Goal capability distance/bearing; no actuator output.

- [ ] **Step 1: Write failing observation/output/clamp tests**

Assert single-frame input, no full-map leakage, two-coordinate waypoint output, capability clamp and deterministic normalization.

- [ ] **Step 2: Implement teacher labels and Huber feasibility loss**

- [ ] **Step 3: Implement E1-E4 curriculum and E5 supported random corridor dataset**

Use E1 straight, E2 L, E3 double-turn, E4 S, E5 random corridor with prior replay; initialize only from the approved visual encoder path.

- [ ] **Step 4: Add DAgger collection at Student states**

At every collection rollout, relabel the visited state with Oracle and append examples; record teacher/student waypoint errors.

- [ ] **Step 5: Run E1-E5 serially with gates**

Require E1 `>=95%`, E2 L `>=90%` and straight `>=95%`, E3 double-turn `>=90%`, E4 S `>=85%`, E5 `>=85%`; always report Oracle-vs-Student gap and all low-level safety metrics.

- [ ] **Step 6: Run full regression, archive Single-Frame Depth Baseline and commit**

### Task 12: Phase F SRU Planner

**Files:**
- Create: `legged_gym/planners/sru_waypoint_planner.py`
- Create: `legged_gym/scripts/train_sru_waypoint.py`
- Create: `legged_gym/scripts/evaluate_sru_waypoint.py`
- Create: `legged_gym/tests/test_sru_waypoint_planner.py`
- Create: `logs/phase_f/`

**Interfaces:**
- `SRUWaypointPlanner.forward(depth_sequence, proprioception_sequence, previous_waypoint) -> waypoint_body_xy`.
- `reset(hidden_state, done_mask)` must clear hidden state per completed environment.

- [ ] **Step 1: Write failing sequence/hidden-reset tests**

Assert fixed sequence shape, hidden-state reset, batch independence, output clamp, and no actuator action.

- [ ] **Step 2: Implement Depth Encoder reuse and SRU waypoint head**

Initialize visual weights from the approved E5 checkpoint; train sequence imitation before closed-loop RL.

- [ ] **Step 3: Run F1 straight and F2 turn curricula with gates**

F1 must not fall below the Single-Frame baseline; F2 reports turn anticipation, waypoint jitter and heading-command jitter.

- [ ] **Step 4: Run F3 on exactly the E5 fixed evaluation set**

Compare Oracle, Single-Frame and SRU on SR, collision, timeout, SPL/path efficiency, centerline deviation, waypoint error, waypoint jitter and hidden reset correctness.

- [ ] **Step 5: Run full regression and commit Phase F decision**

### Task 13: Phase G Maze integration and final verification

**Files:**
- Modify: `legged_gym/envs/rotunbot/maze/rotunbot_maze_local_depth.py`
- Modify: `legged_gym/scripts/evaluate_maze_checkpoint.py`
- Create: `legged_gym/scripts/evaluate_maze_velocity_stack.py`
- Create: `legged_gym/tests/test_maze_velocity_stack_contract.py`
- Modify: `docs/corridor_curriculum_navigation_report.md`
- Create: `logs/phase_g/`

**Interfaces:**
- Maze path is `Oracle global path -> SRU/local waypoint -> Frozen Velocity Local Goal -> Frozen V62`.
- Maze migration must not reintroduce depth-to-actuator or end-to-end actuator action.

- [ ] **Step 1: Write failing Maze interface and freeze-identity tests**

Assert planner output is local waypoint, Local Goal output is `(v,w)`, V62 receives it through `set_command_targets()`, and all frozen SHA values match metadata.

- [ ] **Step 2: Implement Oracle-backed Maze integration**

Use Oracle global path first, then the frozen SRU/local stack; preserve maze collision geometry and existing map code.

- [ ] **Step 3: Run Maze smoke and bounded evaluation**

Require finite state, valid resets, collision accounting, local waypoint feasibility and no direct actuator planner output.

- [ ] **Step 4: Run formal Maze evaluation only after all prior Gates PASS**

Report Current/Regression results and failure family breakdown.

- [ ] **Step 5: Run the complete regression and final verification**

Run unit tests, all stage smoke tests, `git diff --check`, checkpoint SHA checks and artifact schema checks. Do not claim final completion unless every required Gate and artifact is present.

- [ ] **Step 6: Commit final report and archive metadata**

```bash
git add docs/corridor_curriculum_navigation_report.md docs/superpowers/specs docs/superpowers/plans
git commit -m "docs: finalize corridor curriculum navigation verification"
git push origin codex/corridor-curriculum-navigation
```
