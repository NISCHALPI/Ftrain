import pytest
import torch
import torch.nn as nn

from .residual import (
    BottelNeckCNN,
    ResidualBlockCNN,
    ResidualBlockFC,
)


@pytest.fixture()
def conv_input():
    return torch.randn(5, 10, 32, 32).cuda()


@pytest.fixture()
def linear_input():
    return torch.randn(32, 10).cuda()


# Test FCResidualBlockE
@pytest.mark.parametrize("layer_size, out", [(2, 10), (4, 12), (5, 10), (7, 10)])
def test_fc_residual_block(linear_input, layer_size, out):
    activation = nn.ReLU()

    block = ResidualBlockFC(10, out, count=layer_size, activation=activation).cuda()

    output_tensor = block(linear_input)

    assert output_tensor.shape == torch.Size([32, out])
    assert output_tensor.requires_grad
    assert block.count == layer_size
    assert block.bias is True
    assert block.hidden_channel == out


@pytest.mark.parametrize("layer_size", [2, 4, 5, 7])
def test_cnn_residual_block(conv_input, layer_size):
    activation = nn.ReLU()

    block = ResidualBlockCNN(10, 12, count=layer_size, activation=activation).cuda()

    output_tensor = block(conv_input)

    assert output_tensor.shape == torch.Size([5, 12, 32, 32])
    assert output_tensor.requires_grad
    assert block.count == layer_size
    assert block.bias is True
    assert block.hidden_channel == 12


@pytest.mark.parametrize("layer_size", [2, 4, 5, 7])
def test_cnn_bottelnecl_block(conv_input, layer_size):
    activation = nn.ReLU()

    block = BottelNeckCNN(10, 12, count=layer_size, activation=activation).cuda()

    output_tensor = block(conv_input)

    assert output_tensor.shape == torch.Size([5, 12, 32, 32])
    assert output_tensor.requires_grad
    assert block.count == layer_size
    assert block.bias is True
    assert block.hidden_channel == 12
