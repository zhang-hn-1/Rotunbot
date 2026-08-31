# High-Level Direct-Velocity Action Timing Design

## Scope

This design applies only to on-policy training for direct-velocity tasks. It
does not change the frozen V62 velocity tracker, transition manager, governor,
feasible projection, actuator mapping, or the primitive Isaac Gym environment
step.

## Current and target clocks

The current repository configuration advances `sim.dt=0.005 s` with control
decimation `4`, so `env.dt=0.02 s` and the runner observes 50 Hz primitive
environment steps. The direct-velocity policy command frequency is configured
as 5 Hz. The macro duration is therefore 0.20 s and the repeat factor is
derived as:

```text
repeat = round((1 / upper_level_command_frequency_hz) / env.dt)
```

The implementation rejects a ratio whose distance from the nearest integer is
greater than `1e-6`; fractional scheduling is deliberately not implemented.

## Ownership and transition shape

The runner samples one stochastic policy action at `state_t`, then calls the
existing primitive `env.step()` `repeat` times with the same action. The direct
velocity environment already accepts the action only at its configured
high-level tick and keeps V62 execution at its existing lower-level rate. The
runner therefore owns PPO action holding and storage cadence; the environment
continues to own command projection, transition safety, physics, reset, and
termination.

One PPO storage row is exactly:

```text
(obs_t, critic_obs_t, action_t, log_prob_t, value_t,
 macro_reward_t, done_t, next_obs_t)
```

No action sampled during the ten current primitive substeps is stored or
resampled. The repeat remains dynamic rather than hard-coded.

## Reward aggregation

Let `r_j` be the reward returned by primitive substep `j`, with `j=0..k-1`,
and let `gamma_p` be the configured primitive discount. The stored reward is:

```text
R_t = sum(j=0..k-1, gamma_p**j * r_j)
```

Rewards after the first done for an environment are excluded. A done at
substep 2, 3, or 4 therefore contributes only rewards through that terminal
substep and never mixes the automatically reset next episode into the old
macro transition.

## Discount and GAE semantics

The PPO optimizer sees macro transitions, so the macro discount is:

```text
gamma_macro = gamma_p ** k
lambda_macro = lambda_p ** k
```

This preserves both the physical discount horizon and the GAE recursion
factor because:

```text
gamma_macro * lambda_macro = (gamma_p * lambda_p) ** k
```

The configured learning rate, entropy coefficient, clip, network, and reward
weights are unchanged.

## Done, timeout, and reset semantics

Done is tracked per environment across primitive substeps. Once an env emits
success, collision, timeout, or another terminal flag, it is removed from the
active mask for the rest of the macro. Isaac Gym may automatically reset it
inside the primitive `env.step()`; any observations/rewards from that reset
episode are not included in `R_t`.

Timeout bootstrap is carried as a dedicated per-env reward correction at the
primitive timeout discount (`gamma_p**(j+1) * V_t`), preserving the existing
runner convention without pretending that the reset observation is the old
episode's successor. Success and collision do not bootstrap. Parallel envs
continue their own active substeps independently.

## Compatibility

The current direct-velocity configuration therefore derives `repeat=10`.
Tasks opt in through `cfg.env.high_level_action_timing_enabled`. Tasks
without the flag retain the historical one-call/one-storage-row behavior.
Direct velocity S2 adaptation and V1 training enable the flag; formal
evaluators stay explicit about their own primitive stepping and action holding.
