# SDD ledger — plan: docs/superpowers/plans/2026-08-29-corridor-curriculum-navigation.md

## Preflight review

Binding spec: `docs/superpowers/specs/2026-08-29-corridor-curriculum-navigation-design.md` plus `/home/jason/.codex/attachments/d4f71254-2b34-4958-a9f9-c0f20812f163/pasted-text.txt`.

| Task(s) | Producer → consumer / self-consistency check | Finding |
|---|---|---|
| 0 | Baseline identity and 94-test evidence | Consistent; completed in `d878808` and later audit/report commits. |
| 1 | Scenario/artifact interfaces used by Tasks 2–10 | Consistent; implemented in `13d43c8`. |
| 2 | V62 corridor evaluator consumed by Tasks 3–5 | Consistent; implemented in `49ee964` plus causal fixes. |
| 3 | A0 fixed straight gate | Consistent; formal S0A evidence is 20/20 with zero safety violations. |
| 4 | A1 L gate consumes Task 2 controller | Consistent; formal S0B evidence is 20/20 with zero safety violations. |
| 5 | A2 double-turn gate freezes V62 spatial control | Consistent; formal S0C evidence is 30/30 with zero safety violations. |
| 6 | Velocity Local Goal environment consumes frozen V62 | Consistent; implemented by direct SRU velocity stack in `b37bf60` and follow-up fixes without changing V62. |
| 7 | B1 → B2 → B3 checkpoint chain | Partially complete: B1/S1 PASS, B2/S2 PASS, B3/S2B FAIL. The latest reverse-recovery experiments conflict with the binding forward-only `v_cmd >= 0` requirement. |
| 8 | Oracle waypoints consume B3 capability and feed Task 9 | Plan-consistent; must not start GPU smoke until B3 PASS. |
| 9 | C1 → C2 → C3 → C4 → C5 serial checkpoint chain | Plan-consistent; blocked by B3 Gate, not authorized to start yet. |
| 10 | Random corridor consumes ranges proven by Task 9 | Plan-consistent; out of current requested scope and blocked by Task 9. |
| 11 | Depth planner consumes frozen Task 9 stack | Plan-consistent; out of current requested scope. |
| 12 | SRU planner consumes E5 encoder | Plan-consistent; out of current requested scope. |
| 13 | Maze consumes all prior frozen checkpoints | Plan-consistent; out of current requested scope. |
| 1,2 | `CorridorScenario`/artifacts → V62 evaluator | Interface exists and is exercised by formal S0 artifacts. |
| 2,3–5 | `set_command_targets()` and safety counters → A gates | Interface is frozen after A2 PASS. |
| 6,7 | Direct velocity task/evaluator → B-stage training and gates | Current evaluator lacks some required safety/artifact fields; must be completed before B3 PASS. |
| 7,8 | B3 capability → Oracle clamp/lookahead | B3 must enforce forward-only action mapping before capability is frozen. |
| 8,9 | Oracle centerline waypoint → corridor curriculum | No conflict; waypoint remains a local target and never an actuator action. |
| 9,10 | Proven C ranges → supported random corridor | No conflict; Task 10 must not sample beyond C coverage. |
| 10,11 | Oracle teacher → depth student | No conflict; outside current execution batch. |
| 11,12 | E5 encoder → SRU initialization | No conflict; outside current execution batch. |
| 12,13 | Frozen SRU/local stack → Maze | No conflict; outside current execution batch. |

Ruling: Treat S1/S2/S2B as the implemented B1/B2/B3 sequence — the distributions, checkpoint inheritance and Gate thresholds match the binding spec — cost if wrong: stage names and artifact paths need migration, but the trained policy contract is unchanged.

Superseded ruling: Reverse-recovery commits were initially treated as diagnostic only because commit `68b8379` added a forward-only note.

Ruling: B3 may use bounded reverse recovery through the frozen V62 Transition Manager. The original pasted specification contains no forward-only requirement, while a fixed-seed 1000-goal Dubins audit found 130/1000 exact-mixture S2B targets need more than the horizon's absolute maximum 7.5 m of forward-only travel to enter the success disk (estimated ceiling 87%). Cost if wrong: later stages will inherit a reverse-capable local controller, but raw/applied commands and transition activations remain audited and V62 safety is unchanged.

Task 0: complete (commits `6aa9f53..d878808`, evidence present)
Task 1: complete (commit `13d43c8`)
Task 2: complete (commits `49ee964..93fa2a0`)
Task 3: complete (formal S0A 20/20, zero safety violations)
Task 4: complete (formal S0B 20/20, zero safety violations)
Task 5: complete (formal S0C 30/30, zero safety violations)
Task 6: complete (commits `b37bf60..17de2bb` and focused tests)
Task 7: in progress — B1/S1 PASS, B2/S2 PASS, B3/S2B FAIL at 79/100 with 21 timeouts and 0 collisions.

Latest continuation on 2026-08-31:

- The recovery-aware observation migration is implemented and independently
  reviewed: actor observation 272 -> 273 and critic observation 18 -> 19,
  with model-only warm starts and a fresh optimizer.
- A 300-iteration recovery-aware S2B run from S2 reached 66/100 on the fixed
  100-episode gate, with zero collision, divergence, rate, domain, and hidden
  projection violations; S1/S2 regression passed.
- A bounded recovery action-prior experiment was rejected by evidence: it
  reached only 50/100 and introduced 778 feasible-domain violations. The
  experiment was fully reverted from the code path; its artifacts remain under
  `logs/phase_b/S2B_recoveryprior_model300`.
- A second 300-iteration run from the former best S2B `model_900.pt`, using an
  experimental reverse-lateral reward sign, reached 68/100 with zero safety
  violations. It also failed the B3 gate and the experimental sign change was
  reverted. The prior best formal result remains 80/100 at
  `logs/rotunbot_sru_direct_velocity_s2b/Aug30_23-05-08_/model_900.pt`.
- Task 8 Oracle centerline infrastructure and fail-closed smoke checks passed
  independent review. The GPU smoke is intentionally not run with a synthetic
  gate artifact; it awaits an approved B3 checkpoint.

Root cause finding: failed S2B episodes already yaw toward the goal, but keep advancing after the goal enters the rear half-plane and form an orbit. The existing infeasible-turn detector omits the factor of two in `R = d/(2 sin(|b|))`, so it under-detects targets inside the direct-turn circle. A stateless countersteer probe also failed, confirming that B3 needs a learned/stateful multi-phase maneuver rather than another same-side yaw bonus.

Review of `b33eca5`: needs fixes. The evaluator aliases hidden projection jumps to rate violations, terminal episodes can omit the triggering post-step telemetry, and tests do not cover negative commands through the real V62 projection/transition boundary. Fix round 1 is assigned to the resumed implementer before any B3 gate claim.

Training finding: model `model_900.pt` reached 80/100, while `model_1100.pt` fell to 65/100. Successful/failing traces show raw reverse commands and, later, applied reverse commands, but failures drive away. The initial hypothesis was that reverse drive needed the opposite yaw sign; the signed-unicycle relation `beta_dot = -w - (v/r)sin(beta)` disproved that hypothesis. The bounded TDD fix keeps the same goal-side yaw sign for forward and reverse commands, and was accepted by independent review.
