import pytest
import torch
import torch.nn as nn

from .residual_classifier import CNNResidualClassifier


@pytest.fixture()
def conv_input():
    return torch.randn(5, 10, 32, 32).cuda()


@pytest.fixture()
def linear_input():
    return torch.randn(32, 10).cuda()


@pytest.mark.parametrize(
    "n_layer , in_channels, out_channels, kernel_size",
    [
        (
            7,
            10,
            5,
            3,
        ),
        (
            6,
            10,
            7,
            5,
        ),
        (
            8,
            10,
            9,
            7,
        ),
        (
            9,
            10,
            12,
            9,
        ),
    ],
)
def test_cnn_residual_classifier(
    conv_input, n_layer, in_channels, out_channels, kernel_size
):

    activation = nn.ReLU()

    model = CNNResidualClassifier(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        layers=n_layer,
        activation=activation,
        insert_custom_block={
            3: nn.AdaptiveAvgPool2d((75, 75)),
            5: nn.AdaptiveAvgPool2d((25, 25)),
        },
    ).cuda()

    example_input = torch.rand(5, in_channels, 150, 150).cuda()

    output = model(example_input)

    assert output.shape == (5, out_channels)
    assert model.layers == n_layer
    assert output.requires_grad is True
    assert model.kernel_size == kernel_size

    return
