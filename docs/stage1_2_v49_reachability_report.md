# Stage1.2 — V49 State/Reset Validation + Dynamic Reachability

Date: 2026-08-29
Branch: `codex/stage1-2-v49-reachability-audit`
Checkpoint: `model_300.pt` (frozen V49 release, SHA256 `5cd24ff2...1769a`)
Task: `rotunbot_vel_sru50_v49_integration`
Asset: `Rotunbot_test2.urdf`

## 1. Executive conclusion

- **Reset/state contamination: YES, an episode-reset omission existed; FIXED.** The missing runtime resets were current `actions`, command-profile phase/amplitude/reference flags, and related profile state. After the fix, the explicit reset audit is 39 PASS / 0 FAIL / 9 NOT_AVAILABLE. The 9 unavailable states are recurrent/history/latency buffers not exposed by this task.
- **Residual A/B numerical equivalence: INCONCLUSIVE at strict control-level equivalence.** The common-prefix comparison is not numerically equivalent: the first strict difference is a `3.34e-6` derivative-history difference at policy step 0; under the control tolerance (`1e-4` absolute, `1e-3` relative), the first reported difference is action at step 4 (`3.84e-4`) and it later grows. Fresh processes are grossly reproducible and do not swap between forward and reverse behavior, so the remaining difference is consistent with simulator/GPU numerical or contact-state residuals rather than run-order contamination.
- **V49 has a state-dependent 0.2 s dynamic reachable set: YES.** At requested initial `v0=.06/.08`, terminal yaw magnitude stayed below about `0.0015 rad/s`; at `v0=.10/.12`, the observed terminal yaw range reached about `0.0219 rad/s` in the coarse sweep. Initial yaw state also matters because nonzero initial yaw persists into the terminal measurement.
- **`project_velocity_commands()` alone is sufficient: NO.** It is a static command projection. It does not predict the observed 200 ms response, terminal tracking error, or dependence on current velocity and command transition.
- **Dynamic Reference Governor: INCONCLUSIVE / recommended as the next stage, not implemented here.** The evidence supports state-aware command selection, but Stage1.2 does not choose a governor structure or claim a success-rate improvement.
- **V49 retraining: INCONCLUSIVE.** The reset defect is fixed, but clean runs still show a repeatable small forward transient/reversal and poor yaw response in parts of the coarse set. A retraining decision requires a controlled release-criteria experiment after the dynamic envelope is established.

The formal coarse sweep used 4 initial forward-speed targets (`.06/.08/.10/.12`), 3 initial yaw-rate targets (`-.02/0/.02`), 4 target forward speeds, 5 target yaw rates, and 5 repeats: **1200/1200 transitions completed**. Every transition held the target for exactly 10 policy steps at 50 Hz (= 0.2 s); all 12,000 transition trace rows are at policy level.

## 2. Reset and determinism findings

### A/B common-prefix test

Trajectory A `(1,0) → (2,0) → (3,0)` and trajectory B `(1,0) → (2,0.25) → (3,0)` were prepared with the same Stage1 pose generator, episode id 4, checkpoint, seed, and first waypoint `(1,0)`. The evaluator compared command, observation, available history/action state, joint state, and measured body velocity before the first waypoint switch.

Results:

- same initial pose: `true`
- strict comparison: `FAIL`; first difference: policy step 0, `tracking_error_derivative`, absolute `3.34e-6`
- control-tolerance comparison: `FAIL`; first difference: policy step 4, `actions`, absolute `3.84e-4`, relative `1.74e-2`
- later maximum differences included observation `1.54e-2`, root state `3.77e-3`, and measured yaw velocity `3.84e-3`

This is not reported as a PASS. The diagnostic localizes the first control-relevant divergence to the action path after a small derivative-state difference. The remaining in-process divergence is not reproduced as gross behavior changes in fresh processes.

### Fresh-process reproducibility and run order

`A4`, `B5`, `B6`, and `B7` were each run 10 times in independent Python processes. Each case had identical aggregate results across all 10 repeats:

| Case | Direction agreement in v | Negative measured-v fraction | v MAE (m/s) | Minimum measured-v (m/s) |
|---|---:|---:|---:|---:|
| A4 | 0.93 | 0.07 | 0.04159 | -0.00375 |
| B5 | 0.86 | 0.14 | 0.04492 | -0.00221 |
| B6 | 0.91 | 0.09 | 0.04357 | -0.00191 |
| B7 | 0.85 | 0.15 | 0.04740 | -0.01161 |

The required sequences `A4→B4`, `B4→A4`, `A1→A4`, `A4→A4`, and `A4→A4→A4` were also run. Repeated A4 remained at approximately 0.93 direction agreement and 0.07 negative-v fraction; no successful/failed predecessor caused a forward/reverse mode flip. Differences in the repeated A4 terminal aggregates were at the order of `1e-6` in MAE and `1e-7` in minimum velocity.

### Explicit reset audit and permitted fix

Before the fix, the audit found failures in current `actions`, command-profile phase/speed/amplitude/offset/yaw runtime values, and smooth/reference flags. The fix is limited to `RotunbotVel.reset_idx()` and clears those episode runtime states; it does not alter the checkpoint, network, gains, reward, projection envelope, URDF, or control equations. The evaluator also clears the same runtime histories after the existing `BaseTask.reset()` compatibility step and before installing the first explicit evaluation command.

After the fix:

```text
available states: 39 PASS, 0 FAIL
not exposed by current task: 9 NOT_AVAILABLE
```

Unavailable variables are `obs_history`, `observation_history`, `critic_history`, `history`, `hidden_state`, `recurrent_hidden_state`, `action_delay_buffer`, `torque_delay_buffer`, and `latency_buffer`. They are reported as unavailable, not fabricated or silently treated as zero.

### Forward reversal conclusion

The clean A4 result still contains a small early negative measured-v sample, but it is not the Stage1.1-style persistent reverse mode: the minimum is only `-0.00375 m/s`, the negative fraction is 7%, and every fresh repeat is identical. Therefore the answer is **no gross stochastic forward/reverse reversal after reset**, while a small repeatable transient remains and is relevant to low-speed response characterization.

## 3. Dynamic reachability findings

### Protocol

Each transition was:

```text
reset → V49 tracking of (v0,w0) → 10-step stability window
      → switch to raw target → static project_velocity_commands()
      → hold projected target for 10 V49 policy steps
```

No formal transition directly injected a root velocity. All 1200 requested initial states stabilized within the configured 180-step establishment budget. The analysis tolerances are `0.12 m/s` and `0.025 rad/s`, matching `linear_tracking_sigma` and `angular_tracking_sigma` in the existing V49 configuration; these are analysis thresholds, not modified release gates.

### 20/100/200 ms aggregate tracking error

| Horizon | Mean absolute v error (m/s) | Mean absolute w error (rad/s) |
|---|---:|---:|
| 20 ms | 0.02403 | 0.00889 |
| 100 ms | 0.02540 | 0.00947 |
| 200 ms | 0.02832 | 0.00846 |

Across the complete coarse set, the combined 200 ms `tracking_reachable` rate was 95.7%. This high number is partly because the existing angular tolerance is wider than the low-speed yaw response itself; it must not be interpreted as proof that the desired yaw command was dynamically achieved. The stricter direction-plus-20%-response definition gave a combined `response_reachable` rate of 16.7%.

### Static projection versus actual response

For target forward speeds `.06` and `.08`, the static projection produced only zero yaw for the relevant parts of the raw yaw sweep. At target forward speeds `.10` and `.12`, the projected yaw values covered approximately `[-.02,-.01,0,.01,.02]` in this coarse set. Thus raw target availability and projected target availability must be separated in any envelope.

Even when projected yaw was nonzero, a zero-initial-yaw transition commonly ended near zero yaw rate. For initial `w0=0`, examples at 200 ms were:

| Initial v | Projected target w | Mean actual w at 200 ms |
|---:|---:|---:|
| .06 | -.02 | -.00120 |
| .06 | +.02 | -.00030 |
| .08 | -.02 | -.00118 |
| .08 | +.02 | -.00033 |
| .10 | -.02 | -.00111 |
| .10 | +.02 | +.00010 |
| .12 | -.02 | -.00079 |
| .12 | +.02 | +.00043 |

This demonstrates that the static projected target can be admissible while the 200 ms yaw response is weak or wrong-sign relative to the requested yaw change.

### Low-speed yaw collapse

| Requested initial v | Mean established v | 200 ms tracking-reachable | Combined response-reachable | Yaw sign-failure rate | Maximum `|actual w|` at 200 ms |
|---:|---:|---:|---:|---:|---:|
| .06 | .0708 | 1.000 | .150 | .800 | .00133 |
| .08 | .0920 | 1.000 | .303 | .800 | .00145 |
| .10 | .0996 | .907 | .120 | .507 | .02136 |
| .12 | .1168 | .920 | .093 | .407 | .02194 |

The strongest discontinuity is between the `<.10` and `≥.10` operating regions: low-speed actual yaw remains around `1e-3 rad/s`, while the higher-speed cases reach approximately `2e-2 rad/s` in the coarse observations. The 0.08 boundary is also meaningful: both `.06` and `.08` groups have an 80% yaw sign-failure rate, while `.10/.12` are lower but still poor. Therefore Stage1.1’s `v<.10` and `v<.08` observations are supported as useful diagnostic boundaries, not as a claim that either is a universal hard limit.

The maximum-|w| numbers include nonzero initial-yaw states and should be read as an observed state-dependent envelope, not as a pure response to a zero-initial-yaw command. The raw 50Hz trace preserves the distinction.

### Command transition dependence

Using raw trace rows paired by transition, the mean absolute terminal yaw error rose with the magnitude of `Δw`:

| `Δw` (rad/s) | Mean absolute terminal yaw error (rad/s) |
|---:|---:|
| .01 | .0083–.0089 |
| .02 | .0135–.0147 |
| .03 | .0238 |
| .04 | .0340 |

This supports H3 as a contributor, but not as a standalone explanation: the same transition can be easy or hard depending on current `v` and `w`, and low-speed static projection removes some raw yaw changes before V49 sees them.

## 4. Reachability envelope

The measured coarse envelope is state-dependent rather than a single rectangle in `(v,w)`:

- For `v0≈.06–.08`, observed 200 ms yaw response from near-zero yaw is approximately `[-.0015,.0015] rad/s`, despite some raw targets of `±.02 rad/s`. The static projection also removes some of those raw targets at low target speed.
- For `v0≈.10–.12`, observed terminal yaw can reach about `±.02 rad/s` when initial yaw momentum and target direction are favorable, but zero-initial-yaw transitions still frequently remain near zero. The actual dynamic set is therefore narrower than the static projected set and depends on the current state.
- Forward speed was generally within the loose analysis tolerance, but the 200 ms endpoint worsened as command/state combinations moved across the coarse grid; aggregate v error was 0.0283 m/s at 200 ms.

The precise per-transition envelope is in `reachability_grid.csv`; it retains initial v/w bin, raw target, projected target, repeat statistics, response rate, tracking rate, and sign-failure rates. No extrapolation beyond the measured coarse grid is made here.

## 5. H1–H8 revised ranking

| Hypothesis | Rating | Evidence |
|---|---|---|
| H1 Smooth-reference gating | NOT SUPPORTED | Stage1.1 smooth-reference comparison did not improve the failure pattern; this sweep used `smooth_reference_flag=False` consistently. |
| H2 Low-speed yaw reachability collapse | STRONGLY SUPPORTED | `<.10` groups had near-zero yaw response and 80% yaw sign failure; `≥.10` groups reached materially larger yaw magnitudes. |
| H3 Command-transition OOD | WEAKLY SUPPORTED | Terminal yaw error increased with `|Δw|`, but state speed and projection interact with it. |
| H4 Static feasible set insufficient | STRONGLY SUPPORTED | Static projection admits nonzero targets that are not reached in 200 ms; current v/w state changes the result. |
| H5 High-level stopping mismatch | NOT SUPPORTED as primary cause | Stage1.1 terminal-stopping evidence did not explain the early low-level response failure. |
| H6 Intrinsic V49 braking tail | WEAKLY SUPPORTED | Small repeatable negative-v transient remains after reset repair; this stage did not perform a dedicated braking-only sweep. |
| H7 URDF transfer mismatch | NOT SUPPORTED | Stage1.1 comparison of `Rotunbot_test2.urdf` and `Rotunbot.urdf` did not show a decisive transfer explanation. |
| H8 Reset/internal-state inconsistency | SUPPORTED, resolved for exposed state | Explicit audit found omitted runtime resets and the fix produced 39/39 available-state PASS; remaining A/B divergence is small-to-control-level and fresh-process gross behavior is reproducible. |

## 6. Implication for high-level controller

### A. Required information

The high-level controller cannot safely rely only on `v_min/v_max`, `w_min/w_max`, and minimum turn radius. It needs at least a **state-dependent dynamic reachable region** indexed by current forward speed, current yaw rate, and command transition. The static projection remains useful as a first safety filter, but it is not a 200 ms feasibility model.

### B. Online feasibility indicator

`e_v = v_cmd-v_actual` and `e_w = w_cmd-w_actual` are useful online indicators, but not sufficient alone. The same terminal error can arise from different initial speeds and command transitions. The minimum practical indicator set supported by this experiment is:

```text
desired/projected v,w
actual v,w
previous projected command
Δv, Δw
short tracking-error history
```

Joint state is valuable for diagnosis and possibly for a future learned model, but this Stage1.2 sweep does not establish that it is required by a first governor implementation. Recurrent/history inputs are unavailable in the current task and must not be assumed.

### C. Minimum future Governor inputs

Use `desired/projected command`, `actual velocity`, `previous command`, `Δv/Δw`, and short tracking-error history. Add an explicit state-dependent envelope indexed at least by current `|v|` and yaw-rate state. The data does not justify using only target command limits.

### D. Recommended future structure

The evidence supports evaluating the following architecture in a later stage:

```text
static projection
  + command transition/rate limiting
  + state-dependent reachable envelope
  + tracking-error feedback
```

This report does **not** implement a Dynamic Reference Governor, change V49, or claim that this structure improves navigation success. Stage1.2 stops at measurement and characterization.

## 7. Artifacts

Runtime artifacts are local and intentionally not committed:

- `logs/stage1_2_state_audit/A_B_prefix_equivalence.csv`
- `logs/stage1_2_state_audit/A_B_prefix_equivalence_summary.json`
- `logs/stage1_2_state_audit/reset_state_audit.csv`
- `logs/stage1_2_state_audit/reset_state_audit.md`
- `logs/stage1_2_state_audit/fresh_process_reproducibility.csv`
- `logs/stage1_2_state_audit/fresh_process_reproducibility.json`
- `logs/stage1_2_reachability/raw_50hz_trace.csv`
- `logs/stage1_2_reachability/transition_summary.csv`
- `logs/stage1_2_reachability/reachability_grid.csv`
- `logs/stage1_2_reachability/reachability_summary.json`
- `logs/stage1_2_reachability/static_vs_dynamic_envelope.png`
- `logs/stage1_2_reachability/target_w_vs_actual_w_by_initial_v.png`
- `logs/stage1_2_reachability/initial_v_vs_max_dynamic_abs_w.png`
- `logs/stage1_2_reachability/delta_w_vs_terminal_w_error.png`
- `logs/stage1_2_reachability/velocity_space_trajectories.png`

The committed code provides reproducible audit/sweep/plot scripts and pure unit-tested metric helpers; runtime CSV/plots remain local as required.
