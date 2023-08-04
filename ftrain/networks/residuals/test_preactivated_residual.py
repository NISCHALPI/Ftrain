import pytest
import torch
import torch.nn as nn
from .preactivated_residuals import (
    PreactivatedResidualBlockCNN,
    PreactivatedBottelNeckBlockCNN,
)


@pytest.fixture()
def conv_input():
    return torch.randn(5, 10, 32, 32).cuda()


@pytest.fixture()
def linear_input():
    return torch.randn(32, 10).cuda()


@pytest.mark.parametrize("layer_size", [2, 4, 5, 7])
def test_preactivated_cnn_residual_block(conv_input, layer_size):
    activation = nn.ReLU()

    block = PreactivatedResidualBlockCNN(
        10, 12, count=layer_size, activation=activation
    ).cuda()

    output_tensor = block(conv_input)

    assert output_tensor.shape == torch.Size([5, 12, 32, 32])
    assert output_tensor.requires_grad
    assert block.count == layer_size
    assert block.bias == True
    assert block.hidden_channel == 12


@pytest.mark.parametrize("layer_size", [2, 4, 5, 7])
def test_preactivated_cnn_bottelnecl_block(conv_input, layer_size):
    activation = nn.ReLU()

    block = PreactivatedBottelNeckBlockCNN(
        10, 12, count=layer_size, activation=activation
    ).cuda()

    output_tensor = block(conv_input)

    assert output_tensor.shape == torch.Size([5, 12, 32, 32])
    assert output_tensor.requires_grad
    assert block.count == layer_size
    assert block.bias == True
    assert block.hidden_channel == 12
