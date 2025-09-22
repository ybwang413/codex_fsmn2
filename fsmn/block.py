"""Building blocks for feedforward sequential memory networks."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class FSMNBlock(nn.Module):
    """Feedforward Sequential Memory Network (FSMN) block.

    The block consumes inputs shaped ``(batch, time, feature)`` and applies
    a learned tapped-delay memory filter with optional left/right context and
    stride. The filtered representation is added to a feedforward projection and
    returned with a residual connection to the input.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        left_context: int,
        right_context: int = 0,
        stride: int = 1,
        activation: Optional[nn.Module] = None,
        dropout: float = 0.0,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        if left_context < 0:
            raise ValueError("left_context must be >= 0")
        if right_context < 0:
            raise ValueError("right_context must be >= 0")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.left_context = left_context
        self.right_context = right_context
        self.stride = stride
        self.use_residual = use_residual

        self.ff = nn.Linear(input_size, hidden_size)
        self.activation = activation if activation is not None else nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, input_size)

        total_context = left_context + right_context + 1
        self.memory_kernel = nn.Parameter(torch.zeros(hidden_size, total_context))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the FSMN block.

        Args:
            inputs: Tensor with shape ``(batch, time, feature)``.

        Returns:
            Tensor with the same shape as ``inputs``.
        """

        if inputs.dim() != 3:
            raise ValueError("FSMNBlock expects a 3D tensor (batch, time, feature)")
        if inputs.size(-1) != self.input_size:
            raise ValueError(
                f"Expected input feature dimension {self.input_size}, "
                f"but received {inputs.size(-1)}"
            )

        residual = inputs
        hidden = self.ff(inputs)
        if self.activation is not None:
            hidden = self.activation(hidden)
        hidden = hidden + self._apply_memory(hidden)
        hidden = self.dropout(hidden)
        output = self.proj(hidden)
        if self.use_residual:
            output = output + residual
        return output

    def _apply_memory(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply the learnable memory kernel over time."""

        if self.left_context == 0 and self.right_context == 0:
            weights = self.memory_kernel[:, 0].view(1, 1, -1)
            return hidden * weights

        contexts = []
        offsets = range(-self.left_context, self.right_context + 1)
        for offset in offsets:
            shift = offset * self.stride
            if shift < 0:
                shifted = F.pad(hidden[:, : shift, :], (0, 0, -shift, 0))
            elif shift > 0:
                shifted = F.pad(hidden[:, shift:, :], (0, 0, 0, shift))
            else:
                shifted = hidden
            contexts.append(shifted)
        stacked = torch.stack(contexts, dim=-1)  # (batch, time, feat, contexts)
        memory = torch.einsum("btfc,fc->btf", stacked, self.memory_kernel)
        return memory
