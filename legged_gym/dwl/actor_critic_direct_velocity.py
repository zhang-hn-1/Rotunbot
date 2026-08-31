"""Direct SRU navigation policy whose action is the desired ``(v, w)``."""

from pathlib import Path
from collections.abc import Mapping

import torch
import torch.nn as nn
from torch.distributions import Normal

from .actor_critic_depth import DepthAttentionEncoder, SpatialRecurrentUnit


def _activation(name):
    return {"relu": nn.ReLU, "tanh": nn.Tanh, "selu": nn.SELU}.get(name, nn.ELU)()


class ActorCriticDirectVelocity(nn.Module):
    """Depth/goal/proprioception SRU with a two-channel velocity head."""

    is_recurrent = True

    def __init__(
        self,
        num_short_obs,
        num_proprio_obs,
        num_critic_obs,
        num_actions,
        depth_height=8,
        depth_width=32,
        proprio_dim=12,
        goal_dim=2,
        previous_command_dim=2,
        previous_actual_velocity_dim=0,
        encoder_dim=64,
        attention_heads=4,
        hidden_dim=128,
        actor_hidden_dims=(256, 128),
        critic_hidden_dims=(256, 128),
        activation="elu",
        init_noise_std=0.2,
        min_noise_std=0.05,
        max_noise_std=0.8,
        **kwargs,
    ):
        super().__init__()
        self.num_short_obs = int(num_short_obs)
        self.num_single_obs = int(num_proprio_obs)
        self.num_critic_obs = int(num_critic_obs)
        self.num_actions = int(num_actions)
        self.depth_height = int(depth_height)
        self.depth_width = int(depth_width)
        self.proprio_dim = int(proprio_dim)
        self.goal_dim = int(goal_dim)
        self.previous_command_dim = int(previous_command_dim)
        self.previous_actual_velocity_dim = int(previous_actual_velocity_dim)
        self.depth_dim = self.depth_height * self.depth_width
        self.context_dim = (
            self.proprio_dim
            + self.goal_dim
            + self.previous_command_dim
            + self.previous_actual_velocity_dim
        )
        self.legacy_observation_dim = self.context_dim + self.depth_dim
        self.has_recovery_observation = self.num_single_obs == self.legacy_observation_dim + 1
        if self.num_single_obs not in (
            self.legacy_observation_dim,
            self.legacy_observation_dim + 1,
        ):
            raise ValueError("direct velocity observation size mismatch")
        if self.num_short_obs != self.num_single_obs:
            raise ValueError("direct velocity policy requires one observation frame")
        if self.num_actions != 2:
            raise ValueError("direct velocity policy requires exactly two actions")
        self.policy_context_dim = self.context_dim + int(self.has_recovery_observation)

        self.depth_encoder = DepthAttentionEncoder(
            self.depth_height,
            self.depth_width,
            self.policy_context_dim,
            feature_dim=int(encoder_dim),
            heads=int(attention_heads),
        )
        self.memory = SpatialRecurrentUnit(
            int(encoder_dim) + self.policy_context_dim,
            int(hidden_dim),
        )
        actor_layers = []
        last_dim = int(hidden_dim) + self.policy_context_dim
        for layer_dim in actor_hidden_dims:
            actor_layers.extend((nn.Linear(last_dim, int(layer_dim)), _activation(activation)))
            last_dim = int(layer_dim)
        actor_layers.append(nn.Linear(last_dim, 2))
        self.velocity_head = nn.Sequential(*actor_layers[:-1])
        self.velocity_output = actor_layers[-1]

        critic_layers = []
        last_dim = self.num_critic_obs
        for layer_dim in critic_hidden_dims:
            critic_layers.extend((nn.Linear(last_dim, int(layer_dim)), _activation(activation)))
            last_dim = int(layer_dim)
        critic_layers.append(nn.Linear(last_dim, 1))
        self.critic = nn.Sequential(*critic_layers)
        self.std = nn.Parameter(init_noise_std * torch.ones(2))
        self.min_noise_std = float(min_noise_std)
        self.max_noise_std = float(max_noise_std)
        self.distribution = None
        # The runner owns one actor instance for the vectorized environment.
        # This state is only used during rollout/inference; PPO supplies an
        # explicit initial state and masks while replaying sequences.
        self._hidden_state = None
        Normal.set_default_validate_args = False

    def split_observation(self, observations):
        if observations.ndim != 2 or observations.shape[-1] != self.num_single_obs:
            raise ValueError("expected [N, %d] observations" % self.num_single_obs)
        legacy_observation = observations[:, : self.legacy_observation_dim]
        context = legacy_observation[:, : self.context_dim]
        depth = legacy_observation[:, self.context_dim:].reshape(
            -1, 1, self.depth_height, self.depth_width
        )
        if self.has_recovery_observation:
            context = torch.cat((context, observations[:, -1:]), dim=1)
        return context, depth

    def _actor_features(
        self, observations, hidden_states=None, masks=None, update_state=False
    ):
        if observations.ndim == 2:
            context, depth = self.split_observation(observations)
            visual = self.depth_encoder(depth, context)
            sequence = torch.cat((visual, context), dim=-1).unsqueeze(1)
            if hidden_states is None:
                hidden_states = self._hidden_state
            if masks is not None:
                if masks.ndim == 1:
                    masks = masks.unsqueeze(1)
                elif masks.ndim == 2 and masks.shape[-1] != 1:
                    raise ValueError("single-step recurrent masks must have shape [N, 1]")
            recurrent, hidden = self.memory(
                sequence,
                hidden=hidden_states,
                masks=masks,
                return_sequence=True,
            )
            features = torch.cat((recurrent[:, -1], context), dim=-1)
        elif observations.ndim == 3:
            # RolloutStorage presents [time, batch, observation].  Encode all
            # frames together, then run the SRU in chronological order.
            time_steps, batch_size, _ = observations.shape
            flat_context, flat_depth = self.split_observation(
                observations.reshape(time_steps * batch_size, -1)
            )
            flat_visual = self.depth_encoder(flat_depth, flat_context)
            context = flat_context.reshape(time_steps, batch_size, -1)
            sequence = torch.cat(
                (
                    flat_visual.reshape(time_steps, batch_size, -1),
                    context,
                ),
                dim=-1,
            ).transpose(0, 1)
            if hidden_states is None:
                hidden_states = self._hidden_state
            sequence_masks = None
            if masks is not None:
                if masks.ndim != 2 or tuple(masks.shape) != (time_steps, batch_size):
                    raise ValueError("sequence masks must have shape [time, batch]")
                sequence_masks = masks.transpose(0, 1)
            recurrent, hidden = self.memory(
                sequence,
                hidden=hidden_states,
                masks=sequence_masks,
                return_sequence=True,
            )
            features = torch.cat((recurrent.transpose(0, 1), context), dim=-1)
        else:
            raise ValueError("direct velocity actor expects [N, D] or [T, N, D]")
        if update_state:
            self._hidden_state = hidden.detach()
        return features

    def _mean(self, observations, hidden_states=None, masks=None, update_state=True):
        features = self._actor_features(
            observations,
            hidden_states=hidden_states,
            masks=masks,
            update_state=update_state,
        )
        return torch.tanh(self.velocity_output(self.velocity_head(features)))

    def _update_distribution(
        self, observations, masks=None, hidden_states=None, update_state=False
    ):
        mean = self._mean(
            observations,
            hidden_states=hidden_states,
            masks=masks,
            update_state=update_state,
        )
        with torch.no_grad():
            self.std.clamp_(self.min_noise_std, self.max_noise_std)
        self.distribution = Normal(mean, self.std.expand_as(mean))

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def act(self, observations, masks=None, hidden_states=None, **kwargs):
        self._update_distribution(
            observations,
            masks=masks,
            hidden_states=hidden_states,
            update_state=hidden_states is None,
        )
        return self.distribution.sample()

    def act_inference(self, observations):
        return self._mean(observations, update_state=True)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)

    def get_hidden_states(self):
        if self._hidden_state is None:
            return (None, None)
        return (self._hidden_state.detach(), None)

    def reset(self, dones=None):
        if self._hidden_state is None or dones is None:
            return None
        dones = dones.reshape(-1).to(device=self._hidden_state.device, dtype=torch.bool)
        if dones.shape[0] != self._hidden_state.shape[0]:
            raise ValueError("done mask does not match recurrent hidden batch")
        self._hidden_state = torch.where(
            dones.unsqueeze(1),
            torch.zeros_like(self._hidden_state),
            self._hidden_state,
        )
        return None


_MIGRATABLE_INPUT_WEIGHTS = (
    "depth_encoder.cross_query.weight",
    "memory.input_projection.weight",
    "memory.spatial_projection.weight",
    "memory.gates.weight",
    "velocity_head.0.weight",
    "critic.0.weight",
)


def _migration_error(detail):
    raise RuntimeError("unsupported direct-velocity checkpoint mismatch: " + detail)


def _append_zero_input_column(source, target):
    if source.ndim != 2 or target.ndim != 2:
        _migration_error("expected two-dimensional input weight")
    if source.shape[0] != target.shape[0] or target.shape[1] != source.shape[1] + 1:
        _migration_error("expected one appended input column")
    migrated = torch.zeros_like(target)
    migrated[:, : source.shape[1]] = source
    return migrated


def _insert_zero_gate_input_column(source, target, old_input_dim, hidden_dim):
    if (
        source.ndim != 2
        or target.ndim != 2
        or source.shape != (2 * hidden_dim, old_input_dim + hidden_dim)
        or target.shape != (2 * hidden_dim, old_input_dim + 1 + hidden_dim)
    ):
        _migration_error("unexpected SpatialRecurrentUnit gate shape")
    migrated = torch.zeros_like(target)
    migrated[:, :old_input_dim] = source[:, :old_input_dim]
    migrated[:, old_input_dim + 1:] = source[:, old_input_dim:]
    return migrated


def _migrate_context_suffix(source, target, prefix_dim, base_context_dim=16):
    """Insert new context fields before an optional final recovery bit."""
    source_context_dim = source.shape[1] - prefix_dim
    target_context_dim = target.shape[1] - prefix_dim
    source_has_recovery = source_context_dim > base_context_dim
    target_has_recovery = target_context_dim > base_context_dim
    if source_context_dim - int(source_has_recovery) != base_context_dim:
        _migration_error("unexpected source context shape")
    if target_context_dim - int(target_has_recovery) < base_context_dim:
        _migration_error("unexpected context-suffix input shape")
    migrated = torch.zeros_like(target)
    migrated[:, :prefix_dim] = source[:, :prefix_dim]
    migrated[:, prefix_dim:prefix_dim + base_context_dim] = source[:, prefix_dim:prefix_dim + base_context_dim]
    if target_has_recovery and source_has_recovery:
        migrated[:, -1] = source[:, -1]
    return migrated


def _migrate_gate_input(source, target, prefix_dim, base_context_dim, hidden_dim):
    source_input_dim = source.shape[1] - hidden_dim
    target_input_dim = target.shape[1] - hidden_dim
    source_context_dim = source_input_dim - prefix_dim
    target_context_dim = target_input_dim - prefix_dim
    source_has_recovery = source_context_dim > base_context_dim
    target_has_recovery = target_context_dim > base_context_dim
    if source_context_dim - int(source_has_recovery) != base_context_dim or target_context_dim - int(target_has_recovery) < base_context_dim:
        _migration_error("unexpected recurrent gate context shape")
    migrated = torch.zeros_like(target)
    migrated[:, :prefix_dim] = source[:, :prefix_dim]
    migrated[:, prefix_dim:prefix_dim + base_context_dim] = source[:, prefix_dim:prefix_dim + base_context_dim]
    if source_has_recovery and target_has_recovery:
        migrated[:, prefix_dim + target_context_dim - 1] = source[:, prefix_dim + source_context_dim - 1]
    migrated[:, prefix_dim + target_context_dim:prefix_dim + target_context_dim + hidden_dim] = source[:, prefix_dim + source_context_dim:prefix_dim + source_context_dim + hidden_dim]
    return migrated


def _migrate_critic_input(source, target, base_context_dim=16):
    source_has_recovery = source.shape[1] > 18
    target_has_recovery = target.shape[1] > 18
    source_base = source.shape[1] - int(source_has_recovery)
    target_base = target.shape[1] - int(target_has_recovery)
    if source_base < base_context_dim or target_base < base_context_dim:
        _migration_error("unexpected critic input shape")
    migrated = torch.zeros_like(target)
    migrated[:, :base_context_dim] = source[:, :base_context_dim]
    source_tail = source[:, base_context_dim:source_base]
    target_tail_start = base_context_dim + (target_base - source_base)
    migrated[:, target_tail_start:target_base] = source_tail
    if source_has_recovery and target_has_recovery:
        migrated[:, -1] = source[:, -1]
    return migrated


def migrate_direct_velocity_state_dict(source_state_dict, target_state_dict):
    """Map the legacy 272/18 direct policy into the 273/19 observation ABI.

    Only the six first-layer matrices affected by the appended recovery bit
    may differ.  Their new recovery input columns are initialized to zero so
    an old policy is exactly preserved when the recovery bit is zero.
    """
    if not isinstance(source_state_dict, Mapping) or not isinstance(target_state_dict, Mapping):
        _migration_error("state dictionaries are required")
    source_keys = set(source_state_dict)
    target_keys = set(target_state_dict)
    if source_keys != target_keys:
        _migration_error("state-dict keys differ")

    changed = {
        key
        for key in target_keys
        if tuple(source_state_dict[key].shape) != tuple(target_state_dict[key].shape)
    }
    if not changed:
        return {key: value.detach().clone() for key, value in source_state_dict.items()}
    if changed != set(_MIGRATABLE_INPUT_WEIGHTS):
        _migration_error("only recovery-input matrices may change shape")

    cross_source = source_state_dict["depth_encoder.cross_query.weight"]
    cross_target = target_state_dict["depth_encoder.cross_query.weight"]
    if cross_source.ndim != 2 or cross_target.ndim != 2:
        _migration_error("unexpected depth cross-query shape")
    encoder_dim = cross_source.shape[0]
    if cross_target.shape[0] != encoder_dim:
        _migration_error("depth cross-query feature dimension changed")

    input_source = source_state_dict["memory.input_projection.weight"]
    input_target = target_state_dict["memory.input_projection.weight"]
    spatial_source = source_state_dict["memory.spatial_projection.weight"]
    spatial_target = target_state_dict["memory.spatial_projection.weight"]
    hidden_dim = input_source.shape[0] if input_source.ndim == 2 else -1
    if input_source.ndim != 2 or input_target.ndim != 2 or input_source.shape[0] != input_target.shape[0]:
        _migration_error("unexpected SpatialRecurrentUnit input/projection shape")
    hidden_dim = input_source.shape[0]
    if spatial_source.shape != input_source.shape or spatial_target.shape != input_target.shape:
        _migration_error("unexpected SpatialRecurrentUnit spatial shape")

    actor_source = source_state_dict["velocity_head.0.weight"]
    actor_target = target_state_dict["velocity_head.0.weight"]
    critic_source = source_state_dict["critic.0.weight"]
    critic_target = target_state_dict["critic.0.weight"]
    if actor_source.ndim != 2 or actor_target.ndim != 2 or actor_source.shape[0] != actor_target.shape[0]:
        _migration_error("unexpected actor first-layer shape")
    if critic_source.ndim != 2 or critic_target.ndim != 2 or critic_source.shape[0] != critic_target.shape[0]:
        _migration_error("unexpected critic first-layer shape")

    migrated = {key: value.detach().clone() for key, value in source_state_dict.items()}
    migrated["depth_encoder.cross_query.weight"] = _migrate_context_suffix(
        cross_source, cross_target, 0, 16
    )
    migrated["memory.input_projection.weight"] = _migrate_context_suffix(
        input_source, input_target, encoder_dim, 16
    )
    migrated["memory.spatial_projection.weight"] = _migrate_context_suffix(
        spatial_source, spatial_target, encoder_dim, 16
    )
    migrated["memory.gates.weight"] = _migrate_gate_input(
        source_state_dict["memory.gates.weight"],
        target_state_dict["memory.gates.weight"],
        encoder_dim,
        16,
        hidden_dim,
    )
    migrated["velocity_head.0.weight"] = _migrate_context_suffix(
        actor_source, actor_target, hidden_dim, 16
    )
    migrated["critic.0.weight"] = _migrate_critic_input(
        critic_source, critic_target
    )
    return migrated


def load_direct_velocity_warm_start(actor_critic, checkpoint_path, map_location="cpu"):
    """Load only compatible direct-policy weights; never restore PPO state."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(str(checkpoint_path), map_location=map_location)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model_state_dict"), Mapping):
        _migration_error("checkpoint lacks model_state_dict")
    source = payload["model_state_dict"]
    target = actor_critic.state_dict()
    migrated = any(
        tuple(source[key].shape) != tuple(target[key].shape)
        for key in source.keys() & target.keys()
    )
    actor_critic.load_state_dict(
        migrate_direct_velocity_state_dict(source, target), strict=True
    )
    return {
        "checkpoint": str(checkpoint_path),
        "migrated": migrated,
        "source_iteration": int(payload.get("iter", 0)),
    }
