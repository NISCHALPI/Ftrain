import pytest
import torch.nn as nn
import torch
from .prediction_lstm_cell import LSTMCell


@pytest.mark.parametrize(
    "batch_size, input_dim, hidden_dim, extra_forcing_input_dim, conditional_cell_state,  bias, globa_norm, globally_joined_norm",
    [
        (5, 10, 32, [20, 30], True, False, True, False),
        (1, 10, 1, None, False, True, False, True),
        (1, 10, 20, [10, 20, 30, 40, 50], True, True, False, False),
    ],
)
def test_lstm_cell(
    batch_size,
    input_dim,
    hidden_dim,
    extra_forcing_input_dim,
    conditional_cell_state,
    bias,
    globa_norm,
    globally_joined_norm,
):
    curr_device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    cell = LSTMCell(
        input_size=input_dim,
        hidden_size=hidden_dim,
        bias=bias,
        extra_forcing_input_dim=extra_forcing_input_dim,
        conditional_cell_state=conditional_cell_state,
        global_layer_norm=globa_norm,
        global_joint_layer_norm=globally_joined_norm,
    ).to(device=curr_device)

    x_t = torch.rand(batch_size, input_dim, device=curr_device)
    h_t_prev = torch.rand(
        batch_size, hidden_dim, requires_grad=True, device=curr_device
    )
    c_t_prev = torch.rand(
        batch_size, hidden_dim, requires_grad=True, device=curr_device
    )

    # All parameters should be on the GPU
    if torch.cuda.is_available():
        assert all([param.is_cuda for param in cell.parameters()])

    if extra_forcing_input_dim is not None:
        teacher_input = [
            torch.rand(batch_size, dim, device=curr_device)
            for dim in extra_forcing_input_dim
        ]

    h_t_next, c_t_next = (
        cell(x_t, h_t_prev, c_t_prev, *teacher_input)
        if extra_forcing_input_dim is not None
        else cell(x_t, h_t_prev, c_t_prev)
    )

    assert h_t_next.shape == (batch_size, hidden_dim)
    assert c_t_next.shape == (batch_size, hidden_dim)
    assert h_t_next.requires_grad
    assert c_t_next.requires_grad

    assert hasattr(cell, "InputGate")
    assert hasattr(cell, "ForgetGate")
    assert hasattr(cell, "CellInput")
    assert hasattr(cell, "OutputGate")
    assert cell.conditonal_cell_state == conditional_cell_state
    assert cell.global_layer_norm == globa_norm
    assert cell.global_joint_layer_norm == globally_joined_norm
    pass
