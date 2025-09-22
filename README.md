# codex_fsmn2

This repository provides reference implementations and utilities for building
Feedforward Sequential Memory Network (FSMN) components in PyTorch.

## FSMN block

The :class:`fsmn.FSMNBlock` wraps the memory filter, projection, and residual
connections described in the original FSMN architecture. Inputs are shaped as
``(batch, time, feature)`` tensors and a combination of left and right context
is applied through a learnable tapped-delay line.

```python
import torch
from fsmn import FSMNBlock

block = FSMNBlock(
    input_size=64,
    hidden_size=128,
    left_context=3,
    right_context=1,
    stride=1,
)

inputs = torch.randn(8, 120, 64)
outputs = block(inputs)
assert outputs.shape == inputs.shape
```

The memory kernel can be tuned by adjusting the ``left_context``,
``right_context``, and ``stride`` arguments. Set ``use_residual=False`` when you
only need the filtered activations without adding the input back in. By default
an activation (ReLU) and dropout layer are applied between the feedforward and
projection stages, but these can be customized via the ``activation`` and
``dropout`` arguments.
