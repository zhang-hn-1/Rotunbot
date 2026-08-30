"""Direct SRU navigation policy whose action is the desired ``(v, w)``."""

import torch
import torch.nn as nn
from torch.distributions import Normal

from .actor_critic_depth import DepthAttentionEncoder, SpatialRecurrentUnit


def _activation(name):
    return {"relu": nn.ReLU, "tanh": nn.Tanh, "selu": nn.SELU}.get(name, nn.ELU)()


class ActorCriticDirectVelocity(nn.Module):
    """Depth/goal/proprioception SRU with a two-channel velocity head."""

    is_recurrent = False

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
        self.depth_dim = self.depth_height * self.depth_width
        self.context_dim = self.proprio_dim + self.goal_dim + self.previous_command_dim
        if self.num_single_obs != self.context_dim + self.depth_dim:
            raise ValueError("direct velocity observation size mismatch")
        if self.num_short_obs != self.num_single_obs:
            raise ValueError("direct velocity policy requires one observation frame")
        if self.num_actions != 2:
            raise ValueError("direct velocity policy requires exactly two actions")

        self.depth_encoder = DepthAttentionEncoder(
            self.depth_height,
            self.depth_width,
            self.context_dim,
            feature_dim=int(encoder_dim),
            heads=int(attention_heads),
        )
        self.memory = SpatialRecurrentUnit(
            int(encoder_dim) + self.context_dim,
            int(hidden_dim),
        )
        actor_layers = []
        last_dim = int(hidden_dim) + self.context_dim
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
        Normal.set_default_validate_args = False

    def split_observation(self, observations):
        if observations.ndim != 2 or observations.shape[-1] != self.num_single_obs:
            raise ValueError("expected [N, %d] observations" % self.num_single_obs)
        context = observations[:, : self.context_dim]
        depth = observations[:, self.context_dim:].reshape(
            -1, 1, self.depth_height, self.depth_width
        )
        return context, depth

    def _actor_features(self, observations):
        context, depth = self.split_observation(observations)
        visual = self.depth_encoder(depth, context)
        sequence = torch.cat((visual, context), dim=-1).unsqueeze(1)
        hidden = self.memory(sequence)
        return torch.cat((hidden, context), dim=-1)

    def _mean(self, observations):
        features = self._actor_features(observations)
        return torch.tanh(self.velocity_output(self.velocity_head(features)))

    def _update_distribution(self, observations):
        mean = self._mean(observations)
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

    def act(self, observations, **kwargs):
        self._update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations):
        return self._mean(observations)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)

    def reset(self, dones=None):
        return None
