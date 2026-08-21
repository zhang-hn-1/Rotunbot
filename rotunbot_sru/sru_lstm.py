"""PyTorch implementation of a spatially-enhanced LSTM.

The candidate-memory update follows the SRU-LSTM formulation described by
Yang et al.:

    spatial_t = W_s s_t + b_s
    candidate_t = tanh(spatial_t * (W_xg x_t + W_hg h_{t-1} + b_g))

The input, forget, and output gates retain the standard LSTM form.  When a
separate spatial input is not supplied, x_t itself is used as s_t, matching
the core formulation.  A separate spatial stream is useful for robot
proprioception or ego-motion features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn


@dataclass
class SRUState:
    """Hidden and cell states with shape ``[num_layers, batch, hidden]``."""

    h: Tensor
    c: Tensor

    def detach(self) -> "SRUState":
        return SRUState(self.h.detach(), self.c.detach())

    def to(self, *args, **kwargs) -> "SRUState":
        return SRUState(self.h.to(*args, **kwargs), self.c.to(*args, **kwargs))


class SRULSTMCell(nn.Module):
    """One spatially-enhanced LSTM cell.

    Args:
        input_size: Dimension of the temporal observation x_t.
        hidden_size: Dimension of h_t and c_t.
        spatial_size: Dimension of the separate spatial observation s_t.  If
            omitted, x_t is used as the spatial input.
        layer_norm: Apply LayerNorm to the new cell state before tanh.
        spatial_activation: Optional bounded activation for the spatial term.
            ``"identity"`` is closest to the published formula; ``"tanh"``
            can be safer for unstable robot training.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        spatial_size: Optional[int] = None,
        layer_norm: bool = False,
        spatial_activation: str = "identity",
    ) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("input_size and hidden_size must be positive")
        if spatial_size is not None and spatial_size <= 0:
            raise ValueError("spatial_size must be positive when provided")
        if spatial_activation not in {"identity", "tanh", "sigmoid"}:
            raise ValueError("spatial_activation must be identity, tanh, or sigmoid")

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.spatial_size = int(spatial_size) if spatial_size is not None else self.input_size
        self.uses_separate_spatial_input = spatial_size is not None
        self.spatial_activation = spatial_activation

        # Three standard gates are fused; the candidate gate is kept separate
        # because it receives the multiplicative spatial transformation term.
        self.x_gates = nn.Linear(self.input_size, 3 * self.hidden_size, bias=True)
        self.h_gates = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
        self.x_candidate = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.h_candidate = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.spatial_transform = nn.Linear(self.spatial_size, self.hidden_size, bias=True)
        self.cell_norm = nn.LayerNorm(self.hidden_size) if layer_norm else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (
            self.x_gates,
            self.h_gates,
            self.x_candidate,
            self.h_candidate,
            self.spatial_transform,
        ):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        # A positive forget-gate bias is a standard recurrent-training
        # initialization and improves early long-horizon retention.
        with torch.no_grad():
            self.x_gates.bias[self.hidden_size : 2 * self.hidden_size].fill_(1.0)

    def _activate_spatial(self, spatial: Tensor) -> Tensor:
        if self.spatial_activation == "tanh":
            return torch.tanh(spatial)
        if self.spatial_activation == "sigmoid":
            return torch.sigmoid(spatial)
        return spatial

    def forward(
        self,
        x_t: Tensor,
        state: Tuple[Tensor, Tensor],
        spatial_t: Optional[Tensor] = None,
        reset_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Advance one timestep.

        Args:
            x_t: ``[batch, input_size]``.
            state: Tuple ``(h_prev, c_prev)``, each ``[batch, hidden_size]``.
            spatial_t: ``[batch, spatial_size]``. Required only when the cell
                was constructed with ``spatial_size``.
            reset_mask: Boolean or 0/1 tensor ``[batch]``. True resets that
                environment before processing x_t.
        """
        if x_t.ndim != 2 or x_t.shape[-1] != self.input_size:
            raise ValueError(
                f"x_t must have shape [batch, {self.input_size}], got {tuple(x_t.shape)}"
            )
        h_prev, c_prev = state
        expected_state = (x_t.shape[0], self.hidden_size)
        if tuple(h_prev.shape) != expected_state or tuple(c_prev.shape) != expected_state:
            raise ValueError(
                f"state tensors must have shape {expected_state}, got "
                f"{tuple(h_prev.shape)} and {tuple(c_prev.shape)}"
            )

        if reset_mask is not None:
            if reset_mask.ndim != 1 or reset_mask.shape[0] != x_t.shape[0]:
                raise ValueError("reset_mask must have shape [batch]")
            keep = (~reset_mask.bool()).to(dtype=x_t.dtype).unsqueeze(-1)
            h_prev = h_prev * keep
            c_prev = c_prev * keep

        if self.uses_separate_spatial_input:
            if spatial_t is None:
                raise ValueError("spatial_t is required because spatial_size was provided")
            if spatial_t.ndim != 2 or spatial_t.shape != (x_t.shape[0], self.spatial_size):
                raise ValueError(
                    f"spatial_t must have shape [batch, {self.spatial_size}], "
                    f"got {tuple(spatial_t.shape)}"
                )
            spatial_source = spatial_t
        else:
            if spatial_t is not None and spatial_t.shape != x_t.shape:
                raise ValueError("when spatial_size is omitted, spatial_t must match x_t")
            spatial_source = x_t if spatial_t is None else spatial_t

        gate_pre = self.x_gates(x_t) + self.h_gates(h_prev)
        input_pre, forget_pre, output_pre = gate_pre.chunk(3, dim=-1)
        input_gate = torch.sigmoid(input_pre)
        forget_gate = torch.sigmoid(forget_pre)
        output_gate = torch.sigmoid(output_pre)

        spatial_term = self._activate_spatial(self.spatial_transform(spatial_source))
        candidate_pre = self.x_candidate(x_t) + self.h_candidate(h_prev)
        candidate = torch.tanh(spatial_term * candidate_pre)

        c_t = forget_gate * c_prev + input_gate * candidate
        h_t = output_gate * torch.tanh(self.cell_norm(c_t))
        return h_t, c_t


class SRULSTM(nn.Module):
    """Multi-layer SRU-LSTM sequence module with an ``nn.LSTM``-like API."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        spatial_size: Optional[int] = None,
        batch_first: bool = True,
        dropout: float = 0.0,
        layer_norm: bool = False,
        spatial_activation: str = "identity",
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.spatial_size = spatial_size
        self.batch_first = bool(batch_first)
        self.dropout = float(dropout)

        cells = []
        for layer in range(self.num_layers):
            cells.append(
                SRULSTMCell(
                    input_size=self.input_size if layer == 0 else self.hidden_size,
                    hidden_size=self.hidden_size,
                    spatial_size=spatial_size,
                    layer_norm=layer_norm,
                    spatial_activation=spatial_activation,
                )
            )
        self.cells = nn.ModuleList(cells)
        self.dropout_layer = nn.Dropout(self.dropout)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> SRUState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        ref = next(self.parameters())
        device = ref.device if device is None else device
        dtype = ref.dtype if dtype is None else dtype
        shape = (self.num_layers, batch_size, self.hidden_size)
        return SRUState(
            h=torch.zeros(shape, device=device, dtype=dtype),
            c=torch.zeros(shape, device=device, dtype=dtype),
        )

    def forward(
        self,
        x: Tensor,
        spatial: Optional[Tensor] = None,
        state: Optional[Union[SRUState, Tuple[Tensor, Tensor]]] = None,
        reset_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, SRUState]:
        """Encode a sequence.

        Args:
            x: ``[batch, time, input]`` when batch_first, otherwise
                ``[time, batch, input]``.
            spatial: Matching sequence with last dimension ``spatial_size``.
            state: Optional initial state.
            reset_mask: Optional bool tensor ``[batch, time]`` or
                ``[time, batch]``. A true value resets state before that step.
        """
        if x.ndim != 3:
            raise ValueError("x must be a 3-D sequence tensor")
        if not self.batch_first:
            x = x.transpose(0, 1)
            if spatial is not None:
                spatial = spatial.transpose(0, 1)
            if reset_mask is not None:
                reset_mask = reset_mask.transpose(0, 1)

        batch, steps, features = x.shape
        if features != self.input_size:
            raise ValueError(f"expected input_size={self.input_size}, got {features}")
        if spatial is not None:
            if spatial.ndim != 3 or spatial.shape[:2] != (batch, steps):
                raise ValueError("spatial must match x in batch and time dimensions")
            expected_spatial = self.spatial_size if self.spatial_size is not None else self.input_size
            if spatial.shape[-1] != expected_spatial:
                raise ValueError(
                    f"expected spatial last dimension {expected_spatial}, got {spatial.shape[-1]}"
                )
        if self.spatial_size is not None and spatial is None:
            raise ValueError("spatial sequence is required because spatial_size was provided")
        if reset_mask is not None and reset_mask.shape != (batch, steps):
            raise ValueError("reset_mask must match [batch, time]")

        if state is None:
            state_obj = self.initial_state(batch, device=x.device, dtype=x.dtype)
        elif isinstance(state, SRUState):
            state_obj = state
        else:
            state_obj = SRUState(*state)
        expected = (self.num_layers, batch, self.hidden_size)
        if tuple(state_obj.h.shape) != expected or tuple(state_obj.c.shape) != expected:
            raise ValueError(f"initial state must have shape {expected}")

        h_layers = [state_obj.h[i] for i in range(self.num_layers)]
        c_layers = [state_obj.c[i] for i in range(self.num_layers)]
        outputs = []

        for t in range(steps):
            layer_input = x[:, t]
            spatial_t = None if spatial is None else spatial[:, t]
            reset_t = None if reset_mask is None else reset_mask[:, t]
            for layer, cell in enumerate(self.cells):
                h_layers[layer], c_layers[layer] = cell(
                    layer_input,
                    (h_layers[layer], c_layers[layer]),
                    spatial_t=spatial_t,
                    reset_mask=reset_t,
                )
                layer_input = h_layers[layer]
                if layer < self.num_layers - 1 and self.dropout > 0.0:
                    layer_input = self.dropout_layer(layer_input)
            outputs.append(layer_input)

        output = torch.stack(outputs, dim=1)
        final_state = SRUState(torch.stack(h_layers, dim=0), torch.stack(c_layers, dim=0))
        if not self.batch_first:
            output = output.transpose(0, 1)
        return output, final_state

    def step(
        self,
        x_t: Tensor,
        state: Optional[SRUState] = None,
        spatial_t: Optional[Tensor] = None,
        reset_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, SRUState]:
        """Streaming one-step API for deployment and recurrent rollouts."""
        batch = x_t.shape[0]
        if state is None:
            state = self.initial_state(batch, device=x_t.device, dtype=x_t.dtype)
        h_layers = [state.h[i] for i in range(self.num_layers)]
        c_layers = [state.c[i] for i in range(self.num_layers)]
        layer_input = x_t
        for layer, cell in enumerate(self.cells):
            h_layers[layer], c_layers[layer] = cell(
                layer_input,
                (h_layers[layer], c_layers[layer]),
                spatial_t=spatial_t,
                reset_mask=reset_mask,
            )
            layer_input = h_layers[layer]
            if layer < self.num_layers - 1 and self.dropout > 0.0:
                layer_input = self.dropout_layer(layer_input)
        return layer_input, SRUState(torch.stack(h_layers), torch.stack(c_layers))
