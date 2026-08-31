"""Timing and per-environment accounting for high-level PPO transitions."""

from collections import namedtuple

import torch


MacroStepResult = namedtuple(
    "MacroStepResult", ("rewards", "dones", "timeout_bootstrap")
)


def timing_row(policy_sample_id, primitive_step, raw_action, requested_command, applied_command):
    """Build one stable row for the deterministic command-hold artifact."""
    return {
        "policy_sample_id": int(policy_sample_id),
        "primitive_step": int(primitive_step),
        "raw_action_v": float(raw_action[0]),
        "raw_action_w": float(raw_action[1]),
        "requested_v_cmd": float(requested_command[0]),
        "requested_w_cmd": float(requested_command[1]),
        "applied_v_cmd": float(applied_command[0]),
        "applied_w_cmd": float(applied_command[1]),
    }


def derive_action_repeat(env_dt, high_level_frequency_hz, tolerance=1.0e-6):
    """Derive an integer primitive-step repeat for one high-level action."""
    env_dt = float(env_dt)
    frequency = float(high_level_frequency_hz)
    if env_dt <= 0.0 or frequency <= 0.0:
        raise ValueError("env_dt and high_level_frequency_hz must be positive")
    ratio = (1.0 / frequency) / env_dt
    repeat = int(round(ratio))
    if repeat < 1 or abs(ratio - repeat) > float(tolerance):
        raise ValueError(
            "high-level period must be an integer number of primitive steps: "
            "ratio=%.9f" % ratio
        )
    return repeat


class MacroStepAccumulator:
    """Accumulate one held-action transition without crossing episode bounds."""

    def __init__(self, num_envs, repeat, primitive_gamma, device=None):
        if int(num_envs) < 1 or int(repeat) < 1:
            raise ValueError("num_envs and repeat must be positive")
        if not 0.0 < float(primitive_gamma) <= 1.0:
            raise ValueError("primitive_gamma must be in (0, 1]")
        self.num_envs = int(num_envs)
        self.repeat = int(repeat)
        self.primitive_gamma = float(primitive_gamma)
        self.device = device
        self.rewards = None
        self.dones = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.timeout_bootstrap = torch.zeros(self.num_envs, device=device)

    @staticmethod
    def _vector(value, device, dtype=None):
        tensor = torch.as_tensor(value, device=device)
        if tensor.ndim > 1:
            tensor = tensor.reshape(tensor.shape[0], -1)[:, 0]
        if tensor.ndim != 1:
            raise ValueError("macro-step values must have shape [num_envs]")
        return tensor.to(dtype=dtype) if dtype is not None else tensor

    def add(self, rewards, dones, timeouts, values, primitive_index):
        """Add one primitive result and return the pre-step active mask."""
        index = int(primitive_index)
        if index < 0 or index >= self.repeat:
            raise ValueError("primitive_index is outside the configured repeat")
        rewards = self._vector(rewards, self.dones.device, torch.float32)
        dones = self._vector(dones, self.dones.device, torch.bool)
        timeouts = self._vector(timeouts, self.dones.device, torch.bool)
        values = self._vector(values, self.dones.device, torch.float32)
        if rewards.shape[0] != self.num_envs:
            raise ValueError("macro-step batch size mismatch")
        active = ~self.dones
        if self.rewards is None:
            self.rewards = torch.zeros_like(rewards)
        self.rewards.add_(active.float() * (self.primitive_gamma ** index) * rewards)
        newly_done = active & dones
        self.timeout_bootstrap[newly_done] = (
            self.primitive_gamma ** (index + 1) * values[newly_done]
        ) * timeouts[newly_done].float()
        self.dones |= newly_done
        return active

    def result(self):
        if self.rewards is None:
            self.rewards = torch.zeros(self.num_envs, device=self.dones.device)
        return MacroStepResult(
            self.rewards.unsqueeze(1),
            self.dones.unsqueeze(1),
            self.timeout_bootstrap,
        )
