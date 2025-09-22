import pytest

torch = pytest.importorskip("torch")
from torch import nn

from fsmn import FSMNBlock


def test_forward_shape_preserved():
    block = FSMNBlock(input_size=8, hidden_size=12, left_context=2, right_context=1)
    inputs = torch.randn(3, 11, 8)
    outputs = block(inputs)
    assert outputs.shape == inputs.shape


def test_residual_connection_identity():
    block = FSMNBlock(input_size=4, hidden_size=6, left_context=1, right_context=1)
    with torch.no_grad():
        block.ff.weight.zero_()
        block.ff.bias.zero_()
        block.proj.weight.zero_()
        block.proj.bias.zero_()
        block.memory_kernel.zero_()
    inputs = torch.randn(2, 5, 4)
    outputs = block(inputs)
    assert torch.allclose(outputs, inputs)


def test_memory_kernel_matches_reference():
    block = FSMNBlock(
        input_size=3,
        hidden_size=3,
        left_context=1,
        right_context=1,
        stride=1,
        activation=nn.Identity(),
        use_residual=False,
    )
    with torch.no_grad():
        block.ff.weight.copy_(torch.eye(3))
        block.ff.bias.zero_()
        block.proj.weight.copy_(torch.eye(3))
        block.proj.bias.zero_()
        kernel = torch.tensor(
            [[0.2, 0.5, -0.1], [-0.3, 0.4, 0.6], [0.0, -0.2, 0.8]], dtype=torch.float32
        )
        block.memory_kernel.copy_(kernel)

    inputs = torch.tensor(
        [[[1.0, 0.0, -1.0], [0.5, -0.5, 0.5], [0.0, 1.0, 1.0]]], dtype=torch.float32
    )
    outputs = block(inputs)

    expected = torch.zeros_like(inputs)
    offsets = [-1, 0, 1]
    for t in range(inputs.size(1)):
        for idx, offset in enumerate(offsets):
            src = t + offset
            if 0 <= src < inputs.size(1):
                expected[:, t] += inputs[:, src] * kernel[:, idx]
    expected = inputs + expected

    assert torch.allclose(outputs, expected, atol=1e-6, rtol=1e-5)
