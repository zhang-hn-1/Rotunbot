"""Newly initialized V0 CNN+MLP actor for local depth navigation."""

import torch
import torch.nn as nn
from torch.distributions import Normal


def _activation(name):
    return {"relu": nn.ReLU, "tanh": nn.Tanh, "selu": nn.SELU}.get(name, nn.ELU)()


class ActorCriticDepthLocal(nn.Module):
    """272-value actor input and 18-value privileged critic input.

    The first 16 actor values are clean proprioception/local-goal state and
    the last 256 values are one normalized 8x32 depth image.  There is no
    temporal memory or checkpoint-loading path in this V0 policy.
    """

    is_recurrent = False

    def __init__(
        self,
        num_short_obs,
        num_single_obs,
        num_critic_obs,
        num_actions,
        depth_height=8,
        depth_width=32,
        state_dim=16,
        actor_hidden_dims=(256, 128),
        critic_hidden_dims=(256, 128),
        activation="elu",
        init_noise_std=0.3,
        min_noise_std=0.1,
        max_noise_std=0.8,
        **kwargs,
    ):
        super().__init__()
        self.num_short_obs = int(num_short_obs)
        self.num_single_obs = int(num_single_obs)
        self.num_critic_obs = int(num_critic_obs)
        self.num_actions = int(num_actions)
        self.depth_height = int(depth_height)
        self.depth_width = int(depth_width)
        self.state_dim = int(state_dim)
        self.depth_dim = self.depth_height * self.depth_width
        if self.num_single_obs != self.state_dim + self.depth_dim:
            raise ValueError("ActorCriticDepthLocal requires exactly 272 actor inputs")
        if self.num_short_obs != self.num_single_obs:
            raise ValueError("V0 requires num_short_obs == num_single_obs == 272")
        if self.num_critic_obs != 18:
            raise ValueError("ActorCriticDepthLocal requires an 18-value privileged critic")

        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            _activation(activation),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            _activation(activation),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            _activation(activation),
            nn.Flatten(),
        )
        with torch.no_grad():
            encoded_size = int(self.depth_encoder(torch.zeros(1, 1, self.depth_height, self.depth_width)).shape[-1])
        self.depth_projection = nn.Sequential(nn.Linear(encoded_size, 64), _activation(activation))
        self.state_encoder = nn.Sequential(nn.Linear(self.state_dim, 64), _activation(activation), nn.Linear(64, 64), _activation(activation))
        self.fusion = nn.Sequential(nn.Linear(128, 128), _activation(activation))
        actor_layers = []
        last_dim = 128
        for layer_dim in actor_hidden_dims:
            actor_layers.extend((nn.Linear(last_dim, int(layer_dim)), _activation(activation)))
            last_dim = int(layer_dim)
        actor_layers.append(nn.Linear(last_dim, self.num_actions))
        self.actor = nn.Sequential(*actor_layers)

        critic_layers = []
        last_dim = self.num_critic_obs
        for layer_dim in critic_hidden_dims:
            critic_layers.extend((nn.Linear(last_dim, int(layer_dim)), _activation(activation)))
            last_dim = int(layer_dim)
        critic_layers.append(nn.Linear(last_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        self.std = nn.Parameter(torch.full((self.num_actions,), float(init_noise_std)))
        self.min_noise_std = float(min_noise_std)
        self.max_noise_std = float(max_noise_std)
        self.distribution = None
        Normal.set_default_validate_args = False

    def split_observation(self, observations):
        if observations.ndim != 2 or observations.shape[-1] != self.num_single_obs:
            raise ValueError(f"expected [N, {self.num_single_obs}] observations")
        state = observations[:, : self.state_dim]
        depth = observations[:, self.state_dim :].reshape(-1, 1, self.depth_height, self.depth_width)
        return state, depth

    def _actor_features(self, observations):
        state, depth = self.split_observation(observations)
        visual = self.depth_projection(self.depth_encoder(depth))
        return self.fusion(torch.cat((self.state_encoder(state), visual), dim=-1))

    def _update_distribution(self, observations):
        mean = self.actor(self._actor_features(observations))
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
        return self.actor(self._actor_features(observations))

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations, **kwargs):
        if critic_observations.ndim != 2 or critic_observations.shape[-1] != self.num_critic_obs:
            raise ValueError(f"expected [N, {self.num_critic_obs}] critic observations")
        return self.critic(critic_observations)

    def reset(self, dones=None):
        return None
