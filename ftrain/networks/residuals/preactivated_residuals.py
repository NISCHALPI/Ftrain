"""This module implements the preactivated residual network architecture as described in the following paper.

Identity Mappings in Deep Residual Networks" by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. (Published in 2016)
https://arxiv.org/pdf/1603.05027.pdf.

Classes:
    - _GenericPreactivatedResidualBlock: A generic preactivated residual block.
    - PreactivatedResidualBlockCNN: A preactivated residual block for CNNs.
    - PreactivatedBottelNeckBlockCNN: A preactivated bottleneck block for CNNs.
"""

import typing as tp
from copy import deepcopy

import torch
import torch.nn as nn

from .residual import _GenericResidualBlock


__all__ = ["PreactivatedResidualBlockCNN", "PreactivatedBottelNeckBlockCNN"]


class _GenericPreactivatedResidualBlock(_GenericResidualBlock):
    """A generic preactivated residual block that inherits from `_GenericResidualBlock`.

    This class implements the preactivated residual block, which is used as a building block
    in preactivated residual networks. It overrides the `forward` method to define the forward
    pass through the preactivated residual block.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Perform the forward pass through the preactivated residual block.

    Parameters:
        Inherits all parameters from `_GenericResidualBlock`.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x) + self.res_block(x)


class PreactivatedResidualBlockCNN(_GenericPreactivatedResidualBlock):
    """A preactivated residual block for Convolutional Neural Networks (CNNs).

    This class implements a preactivated residual block suitable for CNN architectures. It is
    based on the `_GenericPreactivatedResidualBlock` class and extends it with specific settings
    for convolutional layers.

    Methods:
        __init__(
            channel_in: int,
            channel_out: int,
            activation: nn.Module,
            hidden_channel: tp.Optional[int] = None,
            kernel_size: tp.Union[int, tp.Tuple[int, int]] = 3,
            stride: tp.Union[int, tp.Tuple[int, int]] = 1,
            padding: tp.Union[int, tp.Tuple[int, int]] = 1,
            groups: int = 1,
            bias: bool = True,
            count: int = 2,
            *args,
            **kwargs,
        ) -> None:
            Initialize the PreactivatedResidualBlockCNN.

        _get_conv_block(in_channel: int, out_channel: int) -> nn.Module:
            Create a convolutional block with a batch normalization layer and specified activation function.

        _build_res_block() -> nn.Module:
            Build the entire preactivated residual block.

        _build_shortcut() -> nn.Module:
            Build the shortcut connection for the preactivated residual block.

    Parameters:
        Inherits all parameters from `_GenericPreactivatedResidualBlock`.

    Attributes:
        kernel_size: The size of the convolutional kernel.
        stride: The stride of the convolutional operation.
        padding: The padding applied to the input during the convolution.
        groups: The number of groups for grouped convolution.
        bias: Whether to include bias in the convolutional layer.
    """

    def __init__(
        self,
        channel_in: int,
        channel_out: int,
        activation: nn.Module,
        hidden_channel: tp.Optional[int] = None,
        kernel_size: tp.Union[int, tp.Tuple[int, int]] = 3,
        stride: tp.Union[int, tp.Tuple[int, int]] = 1,
        padding: tp.Union[int, tp.Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        count: int = 2,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the PreactivatedResidualBlockCNN.

        Parameters:
            channel_in (int): The number of input channels to the block.
            channel_out (int): The number of output channels from the block.
            activation (nn.Module): The activation function to be used in the block.
            hidden_channel (int, optional): The number of hidden channels inside the block.
                Defaults to None, in which case it is set to `channel_out`.
            kernel_size (int or tuple, optional): The size of the convolutional kernel.
                Defaults to 3.
            stride (int or tuple, optional): The stride of the convolutional operation.
                Defaults to 1.
            padding (int or tuple, optional): The padding applied to the input during the convolution.
                Defaults to 1.
            groups (int, optional): The number of groups for grouped convolution.
                Defaults to 1.
            bias (bool, optional): Whether to include bias in the convolutional layer.
                Defaults to True.
            count (int, optional): The number of convolutional layers inside the block.
                Defaults to 2.
            *args, **kwargs: Additional arguments to be passed to the parent class constructor.

        Returns:
            None
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

        self.shortcut = self._build_shortcut()
        self.res_block = self._build_res_block()

    def _get_conv_block(self, in_channel: int, out_channel: int) -> nn.Module:
        return nn.Sequential(
            nn.BatchNorm2d(in_channel),
            deepcopy(self.activation),
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                groups=self.groups,
                bias=self.bias,
            ),
        )

    def _build_res_block(self) -> nn.Module:
        _model = []
        # Build the first conv block
        _model.append(
            self._get_conv_block(self.channel_in, self.hidden_channel)
        )

        # Build the middle conv blocks
        for _ in range(self.count - 2):
            _model.append(
                self._get_conv_block(self.hidden_channel, self.hidden_channel)
            )

        # Build the last conv block
        _model.append(
            self._get_conv_block(self.hidden_channel, self.channel_out)
        )

        return nn.Sequential(*_model)

    def _build_shortcut(self) -> nn.Module:
        if self.channel_in == self.channel_out:
            return nn.Identity()

        return nn.Sequential(
            nn.BatchNorm2d(self.channel_in),
            nn.Conv2d(
                self.channel_in,
                self.channel_out,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
        )


class PreactivatedBottelNeckBlockCNN(_GenericPreactivatedResidualBlock):
    """A preactivated bottleneck block for Convolutional Neural Networks (CNNs).

    This class implements a preactivated bottleneck block suitable for CNN architectures. It is
    based on the `_GenericPreactivatedResidualBlock` class and extends it with specific settings
    for bottleneck convolutional layers.

    Methods:
        __init__(
            channel_in: int,
            channel_out: int,
            activation: nn.Module,
            hidden_channel: tp.Optional[int] = None,
            kernel_size: tp.Union[int, tp.Tuple[int, int]] = 3,
            stride: tp.Union[int, tp.Tuple[int, int]] = 1,
            padding: tp.Union[int, tp.Tuple[int, int]] = 1,
            groups: int = 1,
            bias: bool = True,
            count: int = 3,
            *args,
            **kwargs,
        ) -> None:
            Initialize the PreactivatedBottelNeckBlockCNN.

        _get_conv_block(in_channel: int, out_channel: int, one_by_one: bool = False) -> nn.Module:
            Create a convolutional block with a batch normalization layer and specified activation function.

        _build_res_block() -> nn.Module:
            Build the entire preactivated bottleneck block.

    Parameters:
        Inherits all parameters from `_GenericPreactivatedResidualBlock`.

    Attributes:
        kernel_size: The size of the convolutional kernel.
        stride: The stride of the convolutional operation.
        padding: The padding applied to the input during the convolution.
        groups: The number of groups for grouped convolution.
        bias: Whether to include bias in the convolutional layer.
    """

    def __init__(
        self,
        channel_in: int,
        channel_out: int,
        activation: nn.Module,
        hidden_channel: tp.Optional[int] = None,
        kernel_size: tp.Union[int, tp.Tuple[int, int]] = 3,
        stride: tp.Union[int, tp.Tuple[int, int]] = 1,
        padding: tp.Union[int, tp.Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        count: int = 3,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the PreactivatedBottelNeckBlockCNN.

        Parameters:
            channel_in (int): The number of input channels to the block.
            channel_out (int): The number of output channels from the block.
            activation (nn.Module): The activation function to be used in the block.
            hidden_channel (int, optional): The number of hidden channels inside the block.
                Defaults to None, in which case it is set to `channel_out`.
            kernel_size (int or tuple, optional): The size of the convolutional kernel.
                Defaults to 3.
            stride (int or tuple, optional): The stride of the convolutional operation.
                Defaults to 1.
            padding (int or tuple, optional): The padding applied to the input during the convolution.
                Defaults to 1.
            groups (int, optional): The number of groups for grouped convolution.
                Defaults to 1.
            bias (bool, optional): Whether to include bias in the convolutional layer.
                Defaults to True.
            count (int, optional): The number of convolutional layers inside the block.
                Defaults to 3.
            *args, **kwargs: Additional arguments to be passed to the parent class constructor.

        Returns:
            None
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

        self.shortcut = self._build_shortcut()
        self.res_block = self._build_res_block()
        return

    def _get_conv_block(
        self, in_channel: int, out_channel: int, one_by_one: bool = False
    ) -> nn.Module:
        if one_by_one:
            return nn.Sequential(
                nn.BatchNorm2d(in_channel),
                deepcopy(self.activation),
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=self.groups,
                    bias=self.bias,
                ),
            )
        return nn.Sequential(
            nn.BatchNorm2d(in_channel),
            deepcopy(self.activation),
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                groups=self.groups,
                bias=self.bias,
            ),
        )

    def _build_res_block(self) -> nn.Module:
        _model = []
        # Build the first conv block
        _model.append(
            self._get_conv_block(
                self.channel_in, self.hidden_channel, one_by_one=True
            )
        )

        # Build the middle conv blocks
        for _ in range(self.count - 2):
            _model.append(
                self._get_conv_block(self.hidden_channel, self.hidden_channel)
            )

        # Build the last conv block
        _model.append(
            self._get_conv_block(
                self.hidden_channel, self.channel_out, one_by_one=True
            )
        )

        return nn.Sequential(*_model)

    def _build_shortcut(self) -> nn.Module:
        return nn.Sequential(
            nn.BatchNorm2d(self.channel_in),
            nn.Conv2d(
                self.channel_in,
                self.channel_out,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
        )
