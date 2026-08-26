"""Depth-image actor-critic with lightweight spatial recurrent memory.

The network follows the useful part of the SRU navigation design: spatial
features are compressed with self/cross attention and then fused over time by
an element-wise spatial transform inside a recurrent unit.  The environment
still supplies a short history window so this remains compatible with the
existing feed-forward PPO runner.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


class DepthAttentionEncoder(nn.Module):
    """Encode a normalized depth image into a goal-conditioned feature."""

    def __init__(self, height, width, proprio_dim, feature_dim=64, heads=4):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
        )
        with torch.no_grad():
            sample = torch.zeros(1, 1, height, width)
            sample = self.backbone(sample)
        self.feature_height = int(sample.shape[-2])
        self.feature_width = int(sample.shape[-1])
        self.self_attention = nn.MultiheadAttention(
            feature_dim, heads, batch_first=True
        )
        self.cross_query = nn.Linear(proprio_dim, feature_dim)
        self.cross_attention = nn.MultiheadAttention(
            feature_dim, heads, batch_first=True
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, depth, proprio):
        feature_map = self.backbone(depth)
        tokens = feature_map.flatten(2).transpose(1, 2)
        refined, _ = self.self_attention(tokens, tokens, tokens, need_weights=False)
        refined = self.norm(tokens + refined)
        query = self.cross_query(proprio).unsqueeze(1)
        compressed, _ = self.cross_attention(
            query, refined, refined, need_weights=False
        )
        return compressed[:, 0]


class SpatialRecurrentUnit(nn.Module):
    """SRU-GRU style recurrence with an element-wise spatial transform."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.spatial_projection = nn.Linear(input_dim, hidden_dim)
        self.gates = nn.Linear(input_dim + hidden_dim, 2 * hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, sequence):
        batch_size = sequence.shape[0]
        hidden = sequence.new_zeros(batch_size, self.hidden_dim)
        for step in range(sequence.shape[1]):
            current = sequence[:, step]
            update, reset = torch.sigmoid(
                self.gates(torch.cat((current, hidden), dim=-1))
            ).chunk(2, dim=-1)
            spatial = torch.tanh(self.spatial_projection(current))
            candidate_input = self.input_projection(current) + reset * hidden
            candidate = torch.tanh(spatial * candidate_input)
            hidden = (1.0 - update) * hidden + update * candidate
        return hidden


def _activation(name):
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "selu":
        return nn.SELU()
    return nn.ELU()


class ActorCriticDepth(nn.Module):
    """PPO actor-critic for stacked proprioception and depth images."""

    is_recurrent = False

    def __init__(
        self,
        num_short_obs,
        num_proprio_obs,
        num_critic_obs,
        num_actions,
        depth_dim=512,
        proprio_dim=19,
        depth_height=16,
        depth_width=32,
        in_channels=8,
        encoder_dim=64,
        attention_heads=4,
        hidden_dim=128,
        actor_hidden_dims=(256, 128),
        critic_hidden_dims=(256, 128),
        activation="elu",
        init_noise_std=0.5,
        min_noise_std=0.1,
        max_noise_std=1.5,
        **kwargs,
    ):
        super().__init__()
        self.num_short_obs = int(num_short_obs)
        self.num_single_obs = int(num_proprio_obs)
        self.num_critic_obs = int(num_critic_obs)
        self.num_actions = int(num_actions)
        self.depth_dim = int(depth_dim)
        self.proprio_dim = int(proprio_dim)
        self.depth_height = int(depth_height)
        self.depth_width = int(depth_width)
        self.frame_stack = int(in_channels)

        expected_single_obs = self.proprio_dim + self.depth_dim
        if self.num_single_obs != expected_single_obs:
            raise ValueError(
                "ActorCriticDepth received an observation-size mismatch: "
                f"num_single_obs={self.num_single_obs}, "
                f"expected={expected_single_obs}"
            )
        if self.depth_dim != self.depth_height * self.depth_width:
            raise ValueError("depth_dim must equal depth_height * depth_width")

        self.depth_encoder = DepthAttentionEncoder(
            self.depth_height,
            self.depth_width,
            self.proprio_dim,
            feature_dim=int(encoder_dim),
            heads=int(attention_heads),
        )
        recurrent_input_dim = int(encoder_dim) + self.proprio_dim
        self.memory = SpatialRecurrentUnit(recurrent_input_dim, int(hidden_dim))

        actor_layers = []
        actor_input_dim = int(hidden_dim) + self.proprio_dim
        last_dim = actor_input_dim
        actor_hidden_dims = list(actor_hidden_dims)
        for layer_dim in actor_hidden_dims:
            actor_layers.extend((nn.Linear(last_dim, int(layer_dim)), _activation(activation)))
            last_dim = int(layer_dim)
        actor_layers.append(nn.Linear(last_dim, self.num_actions))
        self.actor = nn.Sequential(*actor_layers)

        critic_layers = []
        last_dim = self.num_critic_obs
        critic_hidden_dims = list(critic_hidden_dims)
        for layer_dim in critic_hidden_dims:
            critic_layers.extend((nn.Linear(last_dim, int(layer_dim)), _activation(activation)))
            last_dim = int(layer_dim)
        critic_layers.append(nn.Linear(last_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        self.std = nn.Parameter(init_noise_std * torch.ones(self.num_actions))
        self.min_noise_std = float(min_noise_std)
        self.max_noise_std = float(max_noise_std)
        self.distribution = None
        Normal.set_default_validate_args = False

    def _encode_observation(self, observations):
        if observations.ndim != 2:
            observations = observations.reshape(observations.shape[0], -1)
        frames = observations.reshape(-1, self.frame_stack, self.num_single_obs)
        proprio = frames[..., : self.proprio_dim]
        depth = frames[..., self.proprio_dim :]
        depth = depth.reshape(-1, 1, self.depth_height, self.depth_width)
        proprio_flat = proprio.reshape(-1, self.proprio_dim)
        visual = self.depth_encoder(depth, proprio_flat)
        visual = visual.reshape(-1, self.frame_stack, visual.shape[-1])
        recurrent_input = torch.cat((visual, proprio), dim=-1)
        hidden = self.memory(recurrent_input)
        return torch.cat((hidden, proprio[:, -1]), dim=-1)

    def _update_distribution(self, observations):
        mean = self.actor(observations)
        with torch.no_grad():
            self.std.clamp_(self.min_noise_std, self.max_noise_std)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

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
        self._update_distribution(self._encode_observation(observations))
        return self.distribution.sample()

    def act_inference(self, observations):
        actor_input = self._encode_observation(observations)
        return self.actor(actor_input)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)

    def reset(self, dones=None):
        # The observation history is maintained by the environment, so no
        # persistent hidden tensor is needed by this history-compatible model.
        return None
