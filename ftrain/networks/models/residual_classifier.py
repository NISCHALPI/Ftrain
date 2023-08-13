"""CNN Residual Classifier Module.

This module provides a PyTorch implementation of a classifier using Convolutional Neural Networks (CNN) with residual blocks.
The `CNNResidualClassifier` class allows the user to build a deep CNN-based classifier with the flexibility to customize the number of layers,
the number of residual blocks per layer, the kernel size, and the activation function used.

The module includes the following classes:

- `CNNResidualClassifier`: A class for building a classifier using pre-activated residual blocks in a CNN.

Usage Example:
--------------
import torch
from my_module import CNNResidualClassifier

# Define the input dimensions and number of classes
in_channels = 3
out_channels = 10

# Instantiate the CNNResidualClassifier
classifier = CNNResidualClassifier(in_channels, out_channels, layers=3)

# Forward pass through the classifier
input_data = torch.randn(8, in_channels, 32, 32)  # Assuming input size is 32x32
output = classifier(input_data)
print(output.size())  # Output size will be (8, 10), representing batch_size x number of classes.

Note:
-----
- The default configuration uses 4 unique internal channel sizes: [64, 128, 256, 512].
- The number of residual blocks per layer defaults to 2.
- By default, ReLU activation is used; however, you can provide a custom activation function during instantiation.
- Custom blocks can be inserted at specific layer indices to further customize the network architecture.

For more details about the available functionalities, refer to the docstrings of individual classes and methods.

Please make sure to check the torch and torch.nn documentation for any changes in behavior or new features.
"""
import warnings
from copy import deepcopy

import torch
import torch.nn as nn

from ..residuals import (
    PreactivatedBottelNeckBlockCNN,
    PreactivatedResidualBlockCNN,
)

__all__ = ["CNNResidualClassifier"]


class CNNResidualClassifier(nn.Module):
    """CNNResidualClassifier is a PyTorch module for building a classifier using residual blocks in a Convolutional Neural Network (CNN).

    Args:
    -----
    in_channels (int): Number of input channels.
    out_channels (int): Number of output channels (classes).
    layers (int): Number of layers in the network (excluding the classification layer).
    kernel_size (int, optional): Size of the convolutional kernel. Default is 3.
    activation (nn.Module, optional): Activation function to be used. If None, ReLU is used. Default is None.
    num_residual_blocks (int, optional): Number of residual blocks per layer. Default is 2.
    insert_custom_block (Dict[int, nn.Module], optional): Dictionary containing custom blocks to be inserted
    at specific layer indices (0-indexed). Default is None.
    internal_uniques_channels (Tuple[int, int], optional): Tuple containing custom unique channels to be used.
    Default is None.

    Note:
    -----
    The network architecture is built with pre-activated residual blocks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layers: int,
        kernel_size: int = 3,
        activation: nn.Module | None = None,
        num_residual_blocks: int = 2,
        insert_custom_block: dict[int, nn.Module] = None,
        hidden_unique_channels: tuple[int, int] | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialize a CNNResidualClassifier object.

        Args:
        in_channels (int): Number of input channels in the input data.
        out_channels (int): Number of output channels (classes) for classification.
        layers (int): Number of layers in the network (excluding the classification layer).
        kernel_size (int, optional): Size of the convolutional kernel. Default is 3.
        activation (nn.Module, optional): Activation function to be used. If None, ReLU is used. Default is None.
        num_residual_blocks (int, optional): Number of residual blocks per layer. Default is 2.
        insert_custom_block (Dict[int, nn.Module], optional): Dictionary containing custom blocks to be inserted at specific layer indices (0-indexed). Default is None.
        hidden_unique_channels (Tuple[int, int], optional): Tuple containing custom unique channels to be used. Default is None.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.


        Note:
        - The `CNNResidualClassifier` uses pre-activated residual blocks for building the network.
        - If `activation` is None, ReLU activation function will be used by default.
        - If `hidden_unique_channels` is not provided, the network will use the default channel sizes: [64, 128, 256, 512].
        - If the maximum internal channel size is greater than 512 or the minimum is less than 64, warning messages will be displayed.
        - The network architecture consists of residual blocks with skip connections, making it suitable for deep networks.
        - Custom blocks can be inserted at specific layer indices to allow for further customization of the network.
        """
        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = layers
        self.kernel_size = kernel_size
        self.activation = activation if activation is not None else nn.ReLU()
        self.num_residual_blocks = num_residual_blocks

        # Default channels
        hidden_unique_channels = [64, 128, 256, 512]

        if hidden_unique_channels is not None:
            assert all(
                [
                    isinstance(channel, int)
                    for channel in hidden_unique_channels
                ]
            ), "Channels must be integers."
            if max(hidden_unique_channels) > 512:
                warnings.warn(
                    "Channels greater than 512 may cause memory issues.",
                    stacklevel=1,
                )
            if min(hidden_unique_channels) < 64:
                warnings.warn(
                    "Channels less than 64 may cause less learning.",
                    stacklevel=1,
                )

            hidden_unique_channels = hidden_unique_channels

        # Calculate Channel Propagator
        self.internal_channel_propagotar = self._get_channel_propagator(hidden_unique_channels, self.layers)

        # Default insert custom block
        if insert_custom_block is not None:
            assert all(
                [idx < self.layers for idx in insert_custom_block]
            ), "Custom block index out of range. Must be between [0,n_layers)."
            assert all(
                [
                    isinstance(block, nn.Module)
                    for block in insert_custom_block.values()
                ]
            ), "Custom block must be a nn.Module."
            self.insert_custom_block = insert_custom_block

        # BlowUp Network inputs to internal channel propagator[0] value
        self.blowup = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.internal_channel_propagotar[0],
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(self.internal_channel_propagotar[0]),
            deepcopy(self.activation),
        )

        # Build network
        self.network = self._build_network()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the CNNResidualClassifier.

        Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels).

        Note:
        - The input tensor should have dimensions (batch_size, in_channels, height, width).
        - The output tensor will have dimensions (batch_size, out_channels), representing the batch_size and the probabilities
          of each input belonging to each class (out_channels) after passing through the network.
        """
        return self.network(self.blowup(x))

    @staticmethod
    def _get_channel_propagator(
        internal_uniques_channels: list[int], n_layers: int
    ) -> torch.Tensor:
        propagotar = torch.tensor(internal_uniques_channels).repeat(
            n_layers // len(internal_uniques_channels) + 1
        )
        return propagotar[:n_layers].sort()[0]

    def _build_network(self) -> nn.Module:
        model = nn.ModuleList()


        # Build network n-1 layers
        for idx in range(0, self.layers - 1):
            model.append(
                PreactivatedResidualBlockCNN(
                    channel_in=self.internal_channel_propagotar [idx],
                    channel_out=self.internal_channel_propagotar [idx + 1],
                    kernel_size=self.kernel_size,
                    activation=self.activation,
                    count=self.num_residual_blocks,
                    padding=self.kernel_size // 2,
                )
            )
            # Add custom block if in post format
            if (
                hasattr(self, "insert_custom_block")
                and idx in self.insert_custom_block
            ):
                model.append(self.insert_custom_block[idx])

        # Add Last Layer that maps to out_channels
        model.append(
            PreactivatedBottelNeckBlockCNN(
                channel_in=self.internal_channel_propagotar [-1],
                channel_out=self.out_channels,
                kernel_size=self.kernel_size,
                activation=self.activation,
                count=self.num_residual_blocks,
                padding=self.kernel_size // 2,
            )
        )

        # Add activation after last layer
        model.append(nn.BatchNorm2d(self.out_channels))
        model.append(deepcopy(self.activation))

        # add avg pooling to map each channel to 1 * 1 tensor
        model.append(nn.AdaptiveAvgPool2d((1, 1)))
        # Add flatten layer
        model.append(nn.Flatten())

        return nn.Sequential(*model)
