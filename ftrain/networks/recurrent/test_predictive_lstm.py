import pytest
import torch.nn as nn
import torch

from .predictive_lstm import PredictiveLSTM


@pytest.mark.parametrize(
    "batch_size, seq_len , n_layer, input_size, hidden_size, bias, skipconnections, condition_cell, global_layer_norm, global_joint_norm",
    [
        (10, 10, 1, 10, 20, True, True, True, True, False),
        (1, 30, 2, 5, 14, False, True, False, False, True),
        (3, 1, 5, 7, 20, True, False, True, False, False),
    ],
)
def test_predictive_lstm(
    batch_size,
    seq_len,
    n_layer,
    input_size,
    hidden_size,
    bias,
    skipconnections,
    condition_cell,
    global_layer_norm,
    global_joint_norm,
):
    curr_device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    lstm = PredictiveLSTM(
        n_layers=n_layer,
        input_size=input_size,
        hidden_size=hidden_size,
        bias=bias,
        skip_connections=skipconnections,
        conditional_cell_state=condition_cell,
        global_joint_layer_norm=global_joint_norm,
        global_layer_norm=global_layer_norm,
    ).to(device=curr_device)

    sample_input = torch.randn(batch_size, seq_len, input_size, device=curr_device)

    # All parameters should be on the GPU
    if torch.cuda.is_available():
        assert all([param.is_cuda for param in lstm.parameters()])

    # Test forward pass
    output = lstm(sample_input)

    if skipconnections:
        assert output.shape == (batch_size, seq_len, hidden_size * n_layer)
    else:
        assert output.shape == (batch_size, seq_len, hidden_size)

    assert lstm.hidden_size == hidden_size
    assert lstm.input_size == input_size
    assert lstm.n_layers == n_layer
    assert lstm.bias == bias
    assert lstm.skip_connections == skipconnections
    assert lstm.conditional_cell_state == condition_cell
    assert lstm.global_joint_layer_norm == global_joint_norm
    assert lstm.global_layer_norm == global_layer_norm
    assert output.requires_grad
    return
