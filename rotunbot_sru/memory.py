"""High-level SRU memory encoders for fixed windows and streaming robots."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor, nn

from .sru_lstm import SRULSTM, SRUState


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"unsupported activation: {name}")


def build_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int, activation: str) -> nn.Sequential:
    layers = []
    last = input_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(last, int(width)), _activation(activation)))
        last = int(width)
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


class SRUMemoryEncoder(nn.Module):
    """Encode a fixed observation window into one memory vector.

    This mode is compatible with existing frame-stacked PPO pipelines because
    it does not require recurrent state to be stored in rollout storage.
    """

    def __init__(
        self,
        observation_size: int,
        memory_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        spatial_size: Optional[int] = None,
        projection_hidden: Sequence[int] = (128,),
        activation: str = "elu",
        dropout: float = 0.0,
        layer_norm: bool = True,
        spatial_activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.observation_size = int(observation_size)
        self.spatial_size = spatial_size
        self.memory_size = int(memory_size)
        self.rnn = SRULSTM(
            input_size=self.observation_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            spatial_size=spatial_size,
            batch_first=True,
            dropout=dropout,
            layer_norm=layer_norm,
            spatial_activation=spatial_activation,
        )
        self.projection = build_mlp(hidden_size, projection_hidden, self.memory_size, activation)

    def forward(
        self,
        observation_sequence: Tensor,
        spatial_sequence: Optional[Tensor] = None,
        reset_mask: Optional[Tensor] = None,
    ) -> Tensor:
        output, _ = self.rnn(
            observation_sequence,
            spatial=spatial_sequence,
            reset_mask=reset_mask,
        )
        return self.projection(output[:, -1])


class StreamingSRUMemory(nn.Module):
    """Stateful SRU memory for inference or recurrent PPO integrations.

    ``reset(done)`` must be called with each environment's episode termination
    mask.  During training, call ``detach_state`` at rollout boundaries unless
    the storage explicitly supports full backpropagation through time.
    """

    def __init__(
        self,
        observation_size: int,
        memory_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        spatial_size: Optional[int] = None,
        projection_hidden: Sequence[int] = (128,),
        activation: str = "elu",
        layer_norm: bool = True,
        spatial_activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.rnn = SRULSTM(
            input_size=observation_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            spatial_size=spatial_size,
            batch_first=True,
            layer_norm=layer_norm,
            spatial_activation=spatial_activation,
        )
        self.projection = build_mlp(hidden_size, projection_hidden, memory_size, activation)
        self.state: Optional[SRUState] = None

    def forward(
        self,
        observation_t: Tensor,
        spatial_t: Optional[Tensor] = None,
        reset_mask: Optional[Tensor] = None,
    ) -> Tensor:
        hidden, self.state = self.rnn.step(
            observation_t,
            state=self.state,
            spatial_t=spatial_t,
            reset_mask=reset_mask,
        )
        return self.projection(hidden)

    def reset(self, done: Optional[Tensor] = None) -> None:
        if done is None or self.state is None:
            self.state = None
            return
        if done.ndim != 1 or done.shape[0] != self.state.h.shape[1]:
            raise ValueError("done must have shape [num_envs]")
        keep = (~done.bool()).to(self.state.h.dtype).view(1, -1, 1)
        self.state = SRUState(self.state.h * keep, self.state.c * keep)

    def detach_state(self) -> None:
        if self.state is not None:
            self.state = self.state.detach()
