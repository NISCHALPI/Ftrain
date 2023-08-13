import pytest
import torch.nn as nn
import torch
from .base_gate import _Gate


@pytest.mark.parametrize(
    "batch_size, input_dim, hidden_dim, extra_input_dim, bias, globa_norm, globally_joined_norm, activation",
    [
        (5, 10, 32, None, True, False, True, None),
        (1, 10, 1, None, False, True, False, nn.ReLU()),
        (1, 10, 20, [23, 32], True, True, False, nn.Tanh()),
    ],
)
def test_base_gate(
    batch_size,
    input_dim,
    hidden_dim,
    extra_input_dim,
    bias,
    globa_norm,
    globally_joined_norm,
    activation,
):
    gate = _Gate(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        extra_input_dim=extra_input_dim,
        bias=bias,
        global_norm=globa_norm,
        globally_joined_norm=globally_joined_norm,
        activation=activation,
    )

    h_t = gate(
        torch.rand(batch_size, input_dim),
        torch.rand(batch_size, hidden_dim, requires_grad=True),
        *[torch.rand(batch_size, dim) for dim in extra_input_dim]
        if extra_input_dim
        else [],
    )

    assert h_t.shape == (batch_size, hidden_dim)
    if globally_joined_norm:
        assert hasattr(gate, "ln_global_joined_norm")
    if globa_norm:
        assert hasattr(gate, "ln_global_norm")

    if activation is None:
        assert isinstance(gate.activation, nn.Sigmoid)
    assert h_t.requires_grad

    return
