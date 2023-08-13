"""Deep Residual Learning Framework as described in the paper.

Title: Deep Residual Learning for Image Recognition
Authors: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
Link: https://arxiv.org/pdf/1512.03385.pdf

This code defines two classes, ResidualBlockCNN and ResidualBlockFC, which implement the deep residual learning framework for Convolutional Neural Networks (CNNs) and Fully Connected (FC) networks, respectively. The framework is based on the concept of residual blocks that allow for easier training of very deep networks.

The _GenericResidualBlock is an abstract base class that contains common functionality for both ResidualBlockCNN and ResidualBlockFC. The residual block consists of two main parts: the main residual block and the shortcut connection.

ResidualBlockCNN:
    This class represents a residual block for CNNs. It performs convolutional operations and batch normalization to learn the residual mapping. The shortcut connection is used to match the dimensions of the input and output.

ResidualBlockFC:
    This class represents a residual block for FC networks. It performs linear transformations and batch normalization to learn the residual mapping. The shortcut connection is used to match the dimensions of the input and output.

Note: This code is designed to work with PyTorch.

Attributes:
    __all__ (List[str]): List of symbols to export when using 'from module import *'.

Classes:
    _GenericResidualBlock (nn.Module, ABC): Abstract base class for residual blocks.
    ResidualBlockCNN (nn.Module): Residual block implementation for CNNs.
    ResidualBlockFC (nn.Module): Residual block implementation for FC networks.

For more details on the implementation and usage, please refer to the corresponding class documentation.

For citation and usage of this code, please refer to the paper linked above.
"""

import copy
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

__all__ = ["ResidualBlockCNN", "ResidualBlockFC", "BottelNeckCNN"]


class _GenericResidualBlock(nn.Module, ABC):
    def __init__(
        self,
        channel_in: int,
        channel_out: int,
        activation: nn.Module,
        count: int,
        hidden_channel: int | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.activation = activation
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.hidden_channel = (
            hidden_channel if hidden_channel is not None else channel_out
        )
        assert count >= 2, "Residual block must have at least one layer."
        self.count = count
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)
        res_block = self.res_block(x)
        return self.activation(shortcut + res_block)

    @abstractmethod
    def _build_res_block(self) -> nn.Module:
        pass

    @abstractmethod
    def _build_shortcut(self) -> nn.Module:
        pass


class ResidualBlockCNN(_GenericResidualBlock):
    """Residual Block implementation for Convolutional Neural Networks (CNNs).

    This class represents a residual block for CNNs, as described in the paper
    "Deep Residual Learning for Image Recognition" by Kaiming He, Xiangyu Zhang,
    Shaoqing Ren, and Jian Sun.

    The ResidualBlockCNN performs convolutional operations and batch normalization
    to learn the residual mapping. The shortcut connection is used to match the
    dimensions of the input and output. The activation function is applied after
    each convolution.

    Args:
        channel_in (int): Number of input channels.
        channel_out (int): Number of output channels.
        activation (nn.Module): Activation function to be applied after each convolution.
        hidden_channel (Optional[int]): Number of channels in the hidden layer.
            If None, it defaults to the number of output channels.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        stride (int, optional): Stride value for the convolutional operation. Defaults to 1.
        padding (int, optional): Padding value for the convolutional operation. Defaults to 0.
        groups (int, optional): Number of groups for grouped convolution. Defaults to 1.
        bias (bool, optional): Whether to include a bias term in the convolutional operation.
            Defaults to True.
        count (int, optional): Number of convolutional layers in the residual block.
            Defaults to 2.

    Returns:
        torch.Tensor: The output tensor obtained after passing through the residual block.

    Note:
        - This class is designed to work with PyTorch.
        - The 'channel_in' and 'channel_out' parameters refer to the number of channels in
          the input and output tensors, respectively.
        - The 'activation' parameter should be a valid PyTorch activation function.

    Examples:
        # Create a ResidualBlockCNN with 64 input channels and 128 output channels
        # using ReLU as the activation function.
        residual_block = ResidualBlockCNN(64, 128, activation=nn.ReLU())

        # Pass an input tensor 'x' through the residual block.
        output = residual_block(x)
    """

    def __init__(
        self,
        channel_in: int,
        channel_out: int,
        activation: nn.Module,
        hidden_channel: int | None = None,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        count: int = 2,
        *args,
        **kwargs,
    ) -> None:
        """Constructor for the ResidualBlockCNN class.

        Args:
            channel_in (int): Number of input channels.
            channel_out (int): Number of output channels.
            activation (nn.Module): Activation function to be applied after each convolution.
            hidden_channel (Optional[int]): Number of channels in the hidden layer.
                If None, it defaults to the number of output channels.
            kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
            stride (int, optional): Stride value for the convolutional operation. Defaults to 1.
            padding (int, optional): Padding value for the convolutional operation. Defaults to 0.
            groups (int, optional): Number of groups for grouped convolution. Defaults to 1.
            bias (bool, optional): Whether to use a bias term in the convolutional layers.
                Defaults to True.
            count (int, optional): Number of convolutional layers in the residual block.
                Defaults to 2.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None

        Note:
            - This constructor initializes the ResidualBlockCNN object with the provided parameters.
            - The 'channel_in' and 'channel_out' parameters refer to the number of channels in
              the input and output tensors, respectively.
            - The 'activation' parameter should be a valid PyTorch activation function.
            - The 'hidden_channel' parameter determines the number of channels in the hidden
              convolutional layer. If not provided, it defaults to 'channel_out'.
            - The 'kernel_size', 'stride', 'padding', and 'groups' parameters control the
              behavior of the convolutional layers in the residual block.
            - The 'bias' parameter determines whether the convolutional layers use a bias term.
            - The 'count' parameter determines the number of convolutional layers in the
                residual block.
            - The 'args' and 'kwargs' parameters are used to pass any additional arguments

        Examples:
            # Create a ResidualBlockCNN with 64 input channels and 128 output channels
            # using ReLU as the activation function.
            residual_block = ResidualBlockCNN(64, 128, activation=nn.ReLU())

        """
        super().__init__(
            channel_in,
            channel_out,
            activation,
            hidden_channel=hidden_channel,
            count=count,
            *args,
            **kwargs,
        )
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.res_block = self._build_res_block()
        self.shortcut = self._build_shortcut()
        return

    def _get_conv_block(
        self,
        in_channel: int,
        out_channel: int,
        end_with_activation: bool = True,
    ) -> nn.Module:
        model = []
        model.append(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                groups=self.groups,
                bias=self.bias,
            )
        )
        model.append(nn.BatchNorm2d(out_channel))
        if end_with_activation:
            model.append(copy.deepcopy(self.activation))
        return nn.Sequential(*model)

    def _build_res_block(self) -> nn.Module:
        # Instantiate a module list
        _model = []

        # Add the first convolutional block
        _model.append(
            self._get_conv_block(self.channel_in, self.hidden_channel)
        )

        # Add the intermediate convolutional blocks
        for _ in range(self.count - 2):
            _model.append(
                self._get_conv_block(self.hidden_channel, self.hidden_channel)
            )

        # Add the last convolutional block
        _model.append(
            self._get_conv_block(
                self.hidden_channel,
                self.channel_out,
                end_with_activation=False,
            )
        )

        return nn.Sequential(*_model)

    def _build_shortcut(self) -> nn.Module:
        if self.channel_in == self.channel_out:
            return (
                nn.Identity()
            )  # Identity shortcut if channel_in == channel_out
        else:
            return nn.Sequential(  # Convolutional shortcut if channel_in != channel_out
                nn.Conv2d(
                    self.channel_in,
                    self.channel_out,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=self.groups,
                    bias=self.bias,
                ),
                nn.BatchNorm2d(self.channel_out),
            )


class ResidualBlockFC(_GenericResidualBlock):
    """Residual Block implementation for Fully Connected (FC) networks.

    This class represents a residual block for FC networks, as described in the paper
    "Deep Residual Learning for Image Recognition" by Kaiming He, Xiangyu Zhang,
    Shaoqing Ren, and Jian Sun.

    The ResidualBlockFC performs linear transformations and batch normalization
    to learn the residual mapping. The shortcut connection is used to match the
    dimensions of the input and output. The activation function is applied after
    each linear transformation.

    Args:
        channel_in (int): Number of input channels (or features).
        channel_out (int): Number of output channels (or features).
        activation (nn.Module): Activation function to be applied after each linear transformation.
        hidden_channel (Optional[int]): Number of channels in the hidden layer.
            If None, it defaults to the number of output channels.
        bias (bool, optional): If True, enables bias in linear transformations. Defaults to True.

    Returns:
        torch.Tensor: The output tensor obtained after passing through the residual block.

    Note:
        - This class is designed to work with PyTorch.
        - The 'channel_in' and 'channel_out' parameters refer to the number of channels (or features)
          in the input and output tensors, respectively.
        - The 'activation' parameter should be a valid PyTorch activation function.
    """

    def __init__(
        self,
        channel_in: int,
        channel_out: int,
        activation: nn.Module,
        hidden_channel: int | None = None,
        bias: bool = True,
        count: int = 2,
        *args,
        **kwargs,
    ) -> None:
        """Constructor for the ResidualBlockFC class.

        Args:
            channel_in (int): Number of input channels (or features).
            channel_out (int): Number of output channels (or features).
            activation (nn.Module): Activation function to be applied after each linear transformation.
            hidden_channel (Optional[int]): Number of channels in the hidden layer.
                If None, it defaults to the number of output channels.
            bias (bool, optional): If True, enables bias in linear transformations. Defaults to True.
            count (int, optional): Number of linear transformations to be applied in residual path. Defaults to 2.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None

        Note:
            - This constructor initializes the ResidualBlockFC object with the provided parameters.
            - The 'channel_in' and 'channel_out' parameters refer to the number of channels (or features)
              in the input and output tensors, respectively.
            - The 'activation' parameter should be a valid PyTorch activation function.
            - The 'hidden_channel' parameter determines the number of channels in the hidden
              fully connected layer. If not provided, it defaults to 'channel_out'.
            - The 'bias' parameter controls whether bias is enabled in linear transformations.

        Examples:
            # Create a ResidualBlockFC with 512 input channels and 256 output channels
            # using Sigmoid as the activation function.
            residual_block = ResidualBlockFC(512, 256, activation=nn.Sigmoid())

        """
        super().__init__(
            channel_in,
            channel_out,
            activation,
            hidden_channel=hidden_channel,
            count=count,
            *args,
            **kwargs,
        )
        self.bias = bias
        self.res_block = self._build_res_block()
        self.shortcut = self._build_shortcut()
        return

    def _get_linear_block(
        self,
        in_channel: int,
        out_channel: int,
        end_with_activation: bool = True,
    ) -> nn.Module:
        _model = []
        _model.append(nn.Linear(in_channel, out_channel, bias=self.bias))
        _model.append(nn.BatchNorm1d(out_channel))
        if end_with_activation:
            _model.append(copy.deepcopy(self.activation))
        return nn.Sequential(*_model)

    def _build_res_block(self) -> nn.Module:
        # Instantiate a module list
        _model = []
        # Add the first linear block
        _model.append(
            self._get_linear_block(self.channel_in, self.hidden_channel)
        )

        # Add the intermediate linear blocks
        for _ in range(self.count - 2):
            _model.append(
                self._get_linear_block(
                    self.hidden_channel, self.hidden_channel
                )
            )

        # Add the last linear block
        _model.append(
            self._get_linear_block(
                self.hidden_channel,
                self.channel_out,
                end_with_activation=False,
            )
        )

        return nn.Sequential(*_model)

    def _build_shortcut(self) -> nn.Module:
        if self.channel_in == self.channel_out:
            return (
                nn.Identity()
            )  # Identity shortcut if channel_in == channel_out
        else:
            return (
                nn.Sequential(  # Linear shortcut if channel_in != channel_out
                    nn.Linear(
                        self.channel_in, self.channel_out, bias=self.bias
                    ),
                    nn.BatchNorm1d(self.channel_out),
                )
            )


class BottelNeckCNN(_GenericResidualBlock):
    """Bottleneck CNN block, a variant of the residual block with reduced computational cost.

    This class implements a bottleneck-style convolutional neural network (CNN) block,
    which is commonly used in deep learning architectures to construct residual networks.
    The bottleneck architecture reduces computational complexity while maintaining
    the ability to learn complex features.

    This class extends the `_GenericResidualBlock` base class and can be used as a building
    block for constructing more complex neural network architectures.

    Args:
        channel_in (int): Number of input channels.
        channel_out (int): Number of output channels.
        activation (nn.Module): Activation function to be used.
        hidden_channel (Optional[int]): Number of hidden channels. Defaults to None, which means channel_in * 4.
        kernel_size (Union[int, Tuple[int, int]]): Size of the convolutional kernel. Defaults to 3.
        stride (Union[int, Tuple[int, int]]): Stride for the convolutional operation. Defaults to 1.
        padding (Union[int, Tuple[int, int]]): Padding for the convolutional operation. Defaults to 0.
        groups (int): Number of groups for the convolutional operation. Defaults to 1.
        bias (bool): If True, a bias term is added to the convolutional layers. Defaults to True.
        count (int): Number of intermediate convolutional blocks (excluding the first and last blocks). Defaults to 3.

    Attributes:
        kernel_size (Union[int, Tuple[int, int]]): Size of the convolutional kernel.
        stride (Union[int, Tuple[int, int]]): Stride for the convolutional operation.
        padding (Union[int, Tuple[int, int]]): Padding for the convolutional operation.
        groups (int): Number of groups for the convolutional operation.
        bias (bool): If True, a bias term is added to the convolutional layers.
        res_block (nn.Module): The main bottleneck residual block.
        shortcut (nn.Module): The shortcut connection for the bottleneck block.

    Methods:
        _get_conv_block(in_channel, out_channel, end_with_activation, one_by_one_conv):
            Helper function to create a convolutional block.

        _build_res_block():
            Build the main bottleneck residual block.

        _build_shortcut():
            Build the shortcut connection for the bottleneck block.

    Example:
        # Create a Bottleneck CNN block with 64 input channels and 256 output channels
        # using the ReLU activation function and default values for other parameters.
        block = BottelNeckCNN(64, 256, nn.ReLU())
    """

    def __init__(
        self,
        channel_in: int,
        channel_out: int,
        activation: nn.Module,
        hidden_channel: int | None = None,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        count: int = 3,
        *args,
        **kwargs,
    ) -> None:
        """Initialize a Bottleneck CNN block, a variant of the residual block with reduced computational cost.

        Args:
            channel_in (int): Number of input channels.
            channel_out (int): Number of output channels.
            activation (nn.Module): Activation function to be used.
            hidden_channel (Optional[int]): Number of hidden channels. Defaults to None, which means channel_in * 4.
            kernel_size (Union[int, Tuple[int, int]]): Size of the convolutional kernel. Defaults to 3.
            stride (Union[int, Tuple[int, int]]): Stride for the convolutional operation. Defaults to 1.
            padding (Union[int, Tuple[int, int]]): Padding for the convolutional operation. Defaults to 0.
            groups (int): Number of groups for the convolutional operation. Defaults to 1.
            bias (bool): If True, a bias term is added to the convolutional layers. Defaults to True.
            count (int): Number of intermediate convolutional blocks (excluding the first and last blocks). Defaults to 3.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(
            channel_in,
            channel_out,
            activation,
            count,
            hidden_channel,
            *args,
            **kwargs,
        )
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.res_block = self._build_res_block()
        self.shortcut = self._build_shortcut()
        return

    def _get_conv_block(
        self,
        in_channel: int,
        out_channel: int,
        end_with_activation: bool = True,
        one_by_one_conv: bool = False,
    ) -> nn.Module:
        _model = []

        if one_by_one_conv:
            _model.append(
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=self.groups,
                    bias=self.bias,
                )
            )
        else:
            _model.append(
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    groups=self.groups,
                    bias=self.bias,
                )
            )

        _model.append(nn.BatchNorm2d(out_channel))
        if end_with_activation:
            _model.append(copy.deepcopy(self.activation))
        return nn.Sequential(*_model)

    def _build_res_block(self) -> nn.Module:
        _model = []
        # Add the first convolutional block
        _model.append(
            self._get_conv_block(
                self.channel_in, self.hidden_channel, one_by_one_conv=True
            )
        )

        # Add the intermediate convolutional blocks
        for _ in range(self.count - 2):
            _model.append(
                self._get_conv_block(self.hidden_channel, self.hidden_channel)
            )

        # Add the last convolutional block
        _model.append(
            self._get_conv_block(
                self.hidden_channel,
                self.channel_out,
                end_with_activation=False,
                one_by_one_conv=True,
            )
        )
        return nn.Sequential(*_model)

    def _build_shortcut(self) -> nn.Module:
        if self.channel_in == self.channel_out:
            return nn.Identity()

        return nn.Sequential(
            nn.Conv2d(
                self.channel_in,
                self.channel_out,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=self.groups,
                bias=self.bias,
            ),
            nn.BatchNorm2d(self.channel_out),
        )
