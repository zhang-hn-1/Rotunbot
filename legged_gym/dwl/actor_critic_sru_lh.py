"""SRU (Simple Recurrent Unit) actor-critics for the Rotunbot project.

Two integration modes, both compatible with the existing frame-stacked
observation format (20 frames x 19 dims = 380) and the non-recurrent
PPODWL / DWLOnPolicyRunner pipeline:

  * ``ActorCriticSRULH``   -- 方案 A: SRU directly outputs the 2-D action.
        The long-history encoder is an SRU-LSTM memory encoder
        (rotunbot_sru.memory.SRUMemoryEncoder) that scans the frame window
        and produces one memory vector; the actor MLP consumes
        short_history + memory.  This replaces the DWL-CNN encoder.

  * ``ActorCriticSRUModulate`` -- 方案 B: keep the accepted DWL-CNN policy as
        a frozen base (e.g. uniform 4150) and let a small SRU network output
        a residual modulation on the base 2-D action mean.  The robot keeps
        the exact 2-D velocity/position interface; SRU learns to 调制配合.

The SRU-LSTM cell follows the spatially-enhanced formulation of Yang et al.:

    spatial_t = W_s s_t + b_s
    candidate_t = tanh(spatial_t * (W_xg x_t + W_hg h_{t-1} + b_g))

with standard input/forget/output gates.  A separate spatial stream
(goal-relative + body features) multiplicatively modulates memory formation.
"""

import os
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
from torch.distributions import Normal

from rotunbot_sru.memory import SRUMemoryEncoder, build_mlp

# Only the first call has any effect; guard repeated construction.
try:
    Normal.set_default_validate_args(False)
except (TypeError, AttributeError):
    pass


def _clamp_std(std: nn.Parameter, min_std: float, max_std: float) -> None:
    """Clamp the exploration std parameter in-place, like ActorCriticDWL.

    Uses no_grad so the clamp also takes effect when resuming an older
    checkpoint that carries a large std.
    """
    with torch.no_grad():
        lo = float(min_std) if min_std is not None else 1.0e-6
        hi = float(max_std) if max_std is not None else 1.0e6
        std.clamp_(lo, hi)


class ActorCriticSRULH(nn.Module):
    """Frame-stacked SRU policy; SRU memory encoder replaces the DWL-CNN.

    Drop-in for ``ActorCriticDWL``: same positional constructor signature
    (num_short_obs, num_proprio_obs, num_critic_obs, num_actions) and the
    same act / act_inference / evaluate / update_distribution interface used
    by PPODWL.
    """

    is_recurrent = False  # fixed-window mode; no recurrent PPO storage needed

    def __init__(
        self,
        num_short_obs: int,
        num_proprio_obs: int,
        num_critic_obs: int,
        num_actions: int,
        in_channels: int = 20,
        sru_hidden_size: int = 128,
        sru_memory_size: int = 32,
        sru_num_layers: int = 1,
        spatial_feature_mode: str = "rotunbot_18d",
        spatial_indices: Optional[Sequence[int]] = None,
        actor_hidden_dims: Sequence[int] = (512, 256, 128),
        critic_hidden_dims: Sequence[int] = (512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 0.3,
        min_noise_std: Optional[float] = 0.15,
        max_noise_std: Optional[float] = 0.3,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if num_proprio_obs <= 0:
            raise ValueError("num_proprio_obs must be the single-frame observation size")
        if num_short_obs <= 0:
            raise ValueError("num_short_obs must be positive")

        self.num_short_obs = int(num_short_obs)
        self.num_proprio_obs = int(num_proprio_obs)
        self.num_critic_obs = int(num_critic_obs)
        self.num_actions = int(num_actions)
        self.in_channels = int(in_channels)
        self.expected_actor_obs = self.in_channels * self.num_proprio_obs

        self.spatial_feature_mode = str(spatial_feature_mode).lower()
        if self.spatial_feature_mode == "rotunbot_18d":
            if self.num_proprio_obs < 14:
                raise ValueError("rotunbot_18d mode requires at least 14 features per frame")
            indices = []
            spatial_size = 9
        elif self.spatial_feature_mode == "rotunbot_maze_19d":
            if self.num_proprio_obs < 14:
                raise ValueError("rotunbot_maze_19d mode requires at least 14 features per frame")
            indices = []
            spatial_size = 8
        elif self.spatial_feature_mode == "indices":
            indices = list(range(self.num_proprio_obs)) if spatial_indices is None else list(spatial_indices)
            if not indices:
                raise ValueError("spatial_indices cannot be empty")
            if min(indices) < 0 or max(indices) >= self.num_proprio_obs:
                raise ValueError("spatial_indices must index a single observation frame")
            spatial_size = len(indices)
        else:
            raise ValueError("spatial_feature_mode must be rotunbot_18d, rotunbot_maze_19d or indices")
        self.register_buffer(
            "spatial_indices", torch.tensor(indices, dtype=torch.long), persistent=True
        )

        self.memory = SRUMemoryEncoder(
            observation_size=self.num_proprio_obs,
            spatial_size=spatial_size,
            hidden_size=int(sru_hidden_size),
            memory_size=int(sru_memory_size),
            num_layers=int(sru_num_layers),
            activation=activation,
        )
        self.actor = build_mlp(
            self.num_short_obs + int(sru_memory_size),
            list(actor_hidden_dims),
            self.num_actions,
            activation,
        )
        self.critic = build_mlp(
            self.num_critic_obs,
            list(critic_hidden_dims),
            1,
            activation,
        )
        self.std = nn.Parameter(float(init_noise_std) * torch.ones(self.num_actions))
        self.min_noise_std = min_noise_std
        self.max_noise_std = max_noise_std
        self.distribution: Optional[Normal] = None

    # -- observation plumbing ------------------------------------------------

    def _history(self, observations: Tensor) -> Tensor:
        if observations.ndim == 3:
            if observations.shape[1:] != (self.in_channels, self.num_proprio_obs):
                raise ValueError(
                    "3-D observations must have shape "
                    f"[batch, {self.in_channels}, {self.num_proprio_obs}]"
                )
            return observations
        if observations.ndim != 2 or observations.shape[-1] != self.expected_actor_obs:
            raise ValueError(
                f"actor observations must have {self.expected_actor_obs} flattened features; "
                f"got {tuple(observations.shape)}"
            )
        return observations.reshape(-1, self.in_channels, self.num_proprio_obs)

    def _extract_spatial(self, history: Tensor) -> Tensor:
        if self.spatial_feature_mode == "rotunbot_maze_19d":
            # Maze frame (19-D, Euler-based):
            # 0:2 command_xy, 2:5 position_xyz, 5:8 euler_xyz, 8:11 lin_vel,
            # 11:14 ang_vel, 14 dof_pos, 15:17 dof_vel, 17:19 prev actions.
            relative_goal = history[..., 0:2] - history[..., 2:4]
            euler = history[..., 5:8]
            linear_velocity_xy = history[..., 8:10]
            angular_velocity_z = history[..., 13:14]
            return torch.cat(
                (relative_goal, euler, linear_velocity_xy, angular_velocity_z),
                dim=-1,
            )
        if self.spatial_feature_mode == "indices":
            return history.index_select(-1, self.spatial_indices)
        # Single-frame layout (19-D Rotunbot obs):
        # 0:2 command_xy, 2:4 position_xy, 4:8 quaternion, 8:11 linear velocity,
        # 11:14 angular velocity, 14 dof_pos, 15:17 dof_vel, 17:19 prev actions.
        relative_goal = history[..., 0:2] - history[..., 2:4]
        quaternion = history[..., 4:8]
        linear_velocity_xy = history[..., 8:10]
        angular_velocity_z = history[..., 13:14]
        return torch.cat(
            (relative_goal, quaternion, linear_velocity_xy, angular_velocity_z),
            dim=-1,
        )

    def _actor_features(self, observations: Tensor) -> Tensor:
        history = self._history(observations)
        spatial = self._extract_spatial(history)
        memory = self.memory(history, spatial)
        flat = history.reshape(history.shape[0], -1)
        if self.num_short_obs > flat.shape[-1]:
            raise ValueError("num_short_obs exceeds the flattened observation history")
        short_history = flat[:, -self.num_short_obs:]
        return torch.cat((short_history, memory), dim=-1)

    # -- policy interface (PPODWL-compatible) --------------------------------

    def update_distribution(self, observations: Tensor) -> None:
        mean = self.actor(self._actor_features(observations))
        _clamp_std(self.std, self.min_noise_std, self.max_noise_std)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations: Tensor, **kwargs) -> Tensor:
        del kwargs
        self.update_distribution(observations)
        assert self.distribution is not None
        return self.distribution.sample()

    def act_inference(self, observations: Tensor) -> Tensor:
        return self.actor(self._actor_features(observations))

    def evaluate(self, critic_observations: Tensor, **kwargs) -> Tensor:
        del kwargs
        return self.critic(critic_observations)

    def get_actions_log_prob(self, actions: Tensor) -> Tensor:
        if self.distribution is None:
            raise RuntimeError("act() or update_distribution() must be called first")
        return self.distribution.log_prob(actions).sum(dim=-1)

    def reset(self, dones: Optional[Tensor] = None) -> None:
        del dones  # fixed-window mode has no persistent hidden state

    @property
    def action_mean(self) -> Tensor:
        if self.distribution is None:
            raise RuntimeError("distribution has not been initialized")
        return self.distribution.mean

    @property
    def action_std(self) -> Tensor:
        if self.distribution is None:
            raise RuntimeError("distribution has not been initialized")
        return self.distribution.stddev

    @property
    def entropy(self) -> Tensor:
        if self.distribution is None:
            raise RuntimeError("distribution has not been initialized")
        return self.distribution.entropy().sum(dim=-1)


class ActorCriticSRUModulate(nn.Module):
    """方案 B: frozen DWL-CNN base policy + SRU residual modulation.

    The accepted base policy (e.g. uniform 4150) keeps producing the 2-D
    velocity/position targets; an SRU memory encoder reads the same frame
    window and a small MLP outputs a per-env residual ``delta``:

        action_mean = base_action_mean + delta

    Only the SRU part is trained (base frozen), so training starts from the
    accepted checkpoint and the 2-D interface is unchanged.
    """

    is_recurrent = False

    def __init__(
        self,
        num_short_obs: int,
        num_proprio_obs: int,
        num_critic_obs: int,
        num_actions: int,
        base_path: Optional[str] = None,
        base_trainable: bool = False,
        base_proprio_obs: Optional[int] = None,
        mod_gate_distance: Optional[float] = None,
        mod_max_delta: float = 1.0,
        in_channels: int = 20,
        sru_hidden_size: int = 128,
        sru_memory_size: int = 32,
        sru_num_layers: int = 1,
        spatial_feature_mode: str = "rotunbot_18d",
        spatial_indices: Optional[Sequence[int]] = None,
        mod_hidden_dims: Sequence[int] = (256, 128),
        critic_hidden_dims: Sequence[int] = (512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 0.3,
        min_noise_std: Optional[float] = 0.15,
        max_noise_std: Optional[float] = 0.3,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs
        self.num_short_obs = int(num_short_obs)
        self.num_proprio_obs = int(num_proprio_obs)
        self.num_critic_obs = int(num_critic_obs)
        self.num_actions = int(num_actions)
        self.in_channels = int(in_channels)
        self.expected_actor_obs = self.in_channels * self.num_proprio_obs
        self.base_proprio_obs = (
            int(base_proprio_obs) if base_proprio_obs is not None
            else self.num_proprio_obs
        )
        self.mod_gate_distance = (
            float(mod_gate_distance) if mod_gate_distance is not None else None
        )
        self.mod_max_delta = float(mod_max_delta)
        if not (0 < self.base_proprio_obs <= self.num_proprio_obs):
            raise ValueError("base_proprio_obs must be within [1, num_proprio_obs]")

        # ---- frozen base DWL policy (same architecture as uniform 4150) ----
        from legged_gym.dwl.actor_critic_dwl import ActorCriticDWL

        base_policy_cfg = dict(
            in_channels=in_channels,
            kernel_size=[3, 2],
            filter_size=[16, 8],
            stride_size=[1, 1],
            lh_output_dim=16,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            init_noise_std=init_noise_std,
            min_noise_std=min_noise_std,
            max_noise_std=max_noise_std,
        )
        # The base has its own observation layout: when base_proprio_obs is
        # set (e.g. 19-D slice inside a 35-D maze frame) the base short
        # history scales with the student's frames-per-short ratio.
        base_short_obs = (num_short_obs * self.base_proprio_obs) // num_proprio_obs
        self.base = ActorCriticDWL(
            base_short_obs, self.base_proprio_obs, num_critic_obs, num_actions,
            **base_policy_cfg,
        )
        if base_path:
            base_path = str(base_path).replace(
                "{LEGGED_GYM_ROOT_DIR}",
                os.environ.get("LEGGED_GYM_ROOT_DIR", "/home/jason/SphericalRobot_LeggedGym-master-new-map"),
            )
            state = torch.load(base_path, map_location="cpu")
            base_state = state["model_state_dict"]
            filtered = {
                k: v
                for k, v in base_state.items()
                if k in self.base.state_dict()
                and self.base.state_dict()[k].shape == v.shape
            }
            skipped = len(base_state) - len(filtered)
            self.base.load_state_dict(filtered, strict=False)
            if skipped:
                print(
                    f"[SRUModulate] base: {skipped} tensors skipped for shape "
                    f"mismatch (base critic layout differs from checkpoint)",
                    flush=True,
                )
        self.base.eval()
        for p in self.base.parameters():
            p.requires_grad_(bool(base_trainable))

        # ---- SRU spatial/memory encoder (mirrors ActorCriticSRULH) ---------
        self.spatial_feature_mode = str(spatial_feature_mode).lower()
        if self.spatial_feature_mode == "rotunbot_18d":
            indices = []
            spatial_size = 9
        elif self.spatial_feature_mode == "rotunbot_maze_19d":
            indices = []
            spatial_size = 8
        elif self.spatial_feature_mode == "indices":
            indices = list(range(self.num_proprio_obs)) if spatial_indices is None else list(spatial_indices)
            spatial_size = len(indices)
        else:
            raise ValueError("spatial_feature_mode must be rotunbot_18d, rotunbot_maze_19d or indices")
        self.register_buffer(
            "spatial_indices", torch.tensor(indices, dtype=torch.long), persistent=True
        )

        self.memory = SRUMemoryEncoder(
            observation_size=self.num_proprio_obs,
            spatial_size=spatial_size,
            hidden_size=int(sru_hidden_size),
            memory_size=int(sru_memory_size),
            num_layers=int(sru_num_layers),
            activation=activation,
        )
        self.modulator = build_mlp(
            self.num_short_obs + int(sru_memory_size),
            list(mod_hidden_dims),
            self.num_actions,
            activation,
        )
        # Zero-initialize the modulator output so the first rollouts are
        # exactly the frozen base policy (delta = 0); training then grows the
        # residual only where it improves the accepted behavior.
        with torch.no_grad():
            self.modulator[-1].weight.zero_()
            self.modulator[-1].bias.zero_()
        # Independent critic (same privileged obs as the base used).
        self.critic = build_mlp(
            self.num_critic_obs,
            list(critic_hidden_dims),
            1,
            activation,
        )
        self.std = nn.Parameter(float(init_noise_std) * torch.ones(self.num_actions))
        self.min_noise_std = min_noise_std
        self.max_noise_std = max_noise_std
        self.distribution: Optional[Normal] = None

    # -- shared plumbing -----------------------------------------------------

    def _history(self, observations: Tensor) -> Tensor:
        if observations.ndim == 3:
            if observations.shape[1:] != (self.in_channels, self.num_proprio_obs):
                raise ValueError(
                    "3-D observations must have shape "
                    f"[batch, {self.in_channels}, {self.num_proprio_obs}]"
                )
            return observations
        if observations.ndim != 2 or observations.shape[-1] != self.expected_actor_obs:
            raise ValueError(
                f"actor observations must have {self.expected_actor_obs} flattened features; "
                f"got {tuple(observations.shape)}"
            )
        return observations.reshape(-1, self.in_channels, self.num_proprio_obs)

    def _extract_spatial(self, history: Tensor) -> Tensor:
        if self.spatial_feature_mode == "rotunbot_maze_19d":
            # Maze frame (19-D, Euler-based):
            # 0:2 command_xy, 2:5 position_xyz, 5:8 euler_xyz, 8:11 lin_vel,
            # 11:14 ang_vel, 14 dof_pos, 15:17 dof_vel, 17:19 prev actions.
            relative_goal = history[..., 0:2] - history[..., 2:4]
            euler = history[..., 5:8]
            linear_velocity_xy = history[..., 8:10]
            angular_velocity_z = history[..., 13:14]
            return torch.cat(
                (relative_goal, euler, linear_velocity_xy, angular_velocity_z),
                dim=-1,
            )
        if self.spatial_feature_mode == "indices":
            return history.index_select(-1, self.spatial_indices)
        relative_goal = history[..., 0:2] - history[..., 2:4]
        quaternion = history[..., 4:8]
        linear_velocity_xy = history[..., 8:10]
        angular_velocity_z = history[..., 13:14]
        return torch.cat(
            (relative_goal, quaternion, linear_velocity_xy, angular_velocity_z),
            dim=-1,
        )

    def _modulation_features(self, observations: Tensor) -> Tensor:
        history = self._history(observations)
        spatial = self._extract_spatial(history)
        memory = self.memory(history, spatial)
        flat = history.reshape(history.shape[0], -1)
        short_history = flat[:, -self.num_short_obs:]
        return torch.cat((short_history, memory), dim=-1)

    # -- policy interface ----------------------------------------------------

    def _base_mean(self, observations: Tensor) -> Tensor:
        with torch.no_grad():
            if self.base_proprio_obs == self.num_proprio_obs:
                base_obs = observations
            else:
                # Base consumes only the first base_proprio_obs channels of
                # every frame (e.g. the 19-D repro layout inside a 35-D maze
                # frame with wall rays appended).
                history = self._history(observations)
                base_obs = history[..., : self.base_proprio_obs].reshape(
                    history.shape[0], -1
                )
            base_mean = self.base.act_inference(base_obs)
        return base_mean

    def _mod_gate(self, observations: Tensor) -> Tensor:
        """Front-ray gate for the residual (bug-algorithm prior).

        When ``mod_gate_distance`` is set, the last frame's wall-ray channels
        (rays appended after the base channels) gate the modulation using the
        FRONT rays (heading sector).  In a maze corridor the side walls are
        always within ~1 m, so gating on the minimum ray keeps the residual
        active everywhere and the base still drives into walls.  Gating on the
        front rays instead lets the frozen base drive straight toward the
        target while the corridor ahead is clear, and hands control to the
        SRU residual only when a wall blocks the path.  Returns ``[batch, 1]``.
        """
        if self.mod_gate_distance is None:
            return torch.ones(
                observations.shape[0], 1, device=observations.device
            )
        history = self._history(observations)
        rays = history[:, -1, self.base_proprio_obs : self.num_proprio_obs]
        # Rays 0..2 cover +/-22.5 deg around the body heading.
        front = rays[:, :3].min(dim=-1).values
        gate = torch.clamp(
            (self.mod_gate_distance - front) / self.mod_gate_distance,
            0.0,
            1.0,
        )
        return gate.unsqueeze(-1)

    def update_distribution(self, observations: Tensor) -> None:
        base_mean = self._base_mean(observations)
        delta = self.modulator(self._modulation_features(observations))
        delta = torch.tanh(delta) * self.mod_max_delta
        delta = delta * self._mod_gate(observations)
        mean = base_mean + delta
        _clamp_std(self.std, self.min_noise_std, self.max_noise_std)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations: Tensor, **kwargs) -> Tensor:
        del kwargs
        self.update_distribution(observations)
        assert self.distribution is not None
        return self.distribution.sample()

    def act_inference(self, observations: Tensor) -> Tensor:
        base_mean = self._base_mean(observations)
        delta = self.modulator(self._modulation_features(observations))
        delta = torch.tanh(delta) * self.mod_max_delta
        delta = delta * self._mod_gate(observations)
        return base_mean + delta

    def evaluate(self, critic_observations: Tensor, **kwargs) -> Tensor:
        del kwargs
        return self.critic(critic_observations)

    def get_actions_log_prob(self, actions: Tensor) -> Tensor:
        if self.distribution is None:
            raise RuntimeError("act() or update_distribution() must be called first")
        return self.distribution.log_prob(actions).sum(dim=-1)

    def reset(self, dones: Optional[Tensor] = None) -> None:
        del dones

    @property
    def action_mean(self) -> Tensor:
        if self.distribution is None:
            raise RuntimeError("distribution has not been initialized")
        return self.distribution.mean

    @property
    def action_std(self) -> Tensor:
        if self.distribution is None:
            raise RuntimeError("distribution has not been initialized")
        return self.distribution.stddev

    @property
    def entropy(self) -> Tensor:
        if self.distribution is None:
            raise RuntimeError("distribution has not been initialized")
        return self.distribution.entropy().sum(dim=-1)
