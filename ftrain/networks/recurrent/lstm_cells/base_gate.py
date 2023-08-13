"""Module: Gates Base LTSM.

Usage:
from layer_normalized_lstm import _Gate.

References:
[1] "LAYER-NORMALIZED LSTM FOR HYBRID-HMM AND END-TO-END ASR"
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9053635

"""


import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["_Gate"]


class _Gate(nn.Module):
    """Custom LSTM gate with layer normalization and optional gating mechanisms.

    Args:
        input_dim (int): Input dimension.
            The size of the input tensor along the input feature dimension.
        hidden_dim (int): Hidden state dimension.
            The size of the hidden state tensor along the hidden feature dimension.
        extra_input_dim (list[int] | None, optional): List of dimensions for extra input tensors. Defaults to None.
            If provided, a list of dimensions for additional input tensors.
        globally_joined_norm (bool, optional): Apply globally joined layer normalization. Defaults to True.
            Whether to apply globally joined layer normalization.
        global_norm (bool, optional): Apply global layer normalization. Defaults to False.
            Whether to apply global layer normalization.
        bias (bool, optional): Use bias terms. Defaults to True.
            Whether to include bias terms.
        activation (nn.Module | None, optional): Activation function. Defaults to Sigmoid.
            Activation function applied to the gate output. If None, Sigmoid is used.

    References:
        [1] "LAYER-NORMALIZED LSTM FOR HYBRID-HMM AND END-TO-END ASR"
            https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9053635


    Raises:
        AssertionError: If input_dim or hidden_dim is not greater than 0.
                        If extra_input_dim is provided but contains non-positive dimensions.
                        If both global_norm and globally_joined_norm are set to True.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        extra_input_dim: list[int] | None = None,
        globally_joined_norm: bool = True,
        global_norm: bool = False,
        bias: bool = True,
        activation: nn.Module | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the custom LSTM gate with layer normalization and optional gating mechanisms.

        Args:
            input_dim (int): Input dimension.
            hidden_dim (int): Hidden state dimension.
            extra_input_dim (list[int] | None, optional): List of dimensions for extra input tensors. Defaults to None.
            globally_joined_norm (bool, optional): Apply globally joined layer normalization. Defaults to True.
            global_norm (bool, optional): Apply global layer normalization. Defaults to False.
            bias (bool, optional): Use bias terms. Defaults to True.
            activation (nn.Module | None, optional): Activation function. Defaults to Sigmoid.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.


        Returns:
            None

        Raises:
            AssertionError: If input_dim or hidden_dim is not greater than 0.
                            If extra_input_dim is provided but contains non-positive dimensions.
                            If both global_norm and globally_joined_norm are set to True.
        """
        super().__init__(*args, **kwargs)

        assert (
            input_dim > 0 and hidden_dim > 0
        ), "input_dim must be > 0 and hidden_dim must be > 0"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if extra_input_dim is not None and len(extra_input_dim) > 0:
            assert all(
                [dim > 0 for dim in extra_input_dim]
            ), "extra_input_dim must be > 0"
            self.extra_input_dim = extra_input_dim

        assert not (
            global_norm and globally_joined_norm
        ), "globa_norm and globally_joined_norm cannot be True at the same time"

        self.bias = bias

        # Globally Joined Norm
        if globally_joined_norm and bias:
            warnings.warn(
                "The bias term is redundant when globally_joined_norm is True.",
                stacklevel=1,
            )
            self.bias = False

        self.globally_joined_norm = globally_joined_norm
        self.globa_norm = global_norm

        # Register Weight Matrix
        # Register the IH weight matrix
        self.register_parameter(
            "weight_ih", nn.Parameter(torch.randn(self.hidden_dim, self.input_dim))
        )

        # Register the HH weight matrix
        self.register_parameter(
            "weight_hh", nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        )

        # Register Extra Input Weight Matrix
        if hasattr(self, 'extra_input_dim'):
            self.extra_parameter_list = nn.ParameterList(
                [
                    nn.Parameter(torch.randn(hidden_dim, dim))
                    for dim in self.extra_input_dim
                ]
            )

        # Register bias
        if self.bias:
            self.register_parameter(
                "bias_vector", nn.Parameter(torch.randn(hidden_dim))
            )

        # Set Activations
        if activation is not None:
            self.activation = activation
        else:
            self.activation = nn.Sigmoid()

        # Register Norms
        if self.globa_norm:
            # If global norm is True, then we need to register the global norm for every input.
            self.register_module(
                "ln_global_norm",
                nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(2)]),
            )

            # Extend global Norm if extra input is provided
            if hasattr(self, 'extra_input_dim'):
                self.ln_global_norm.extend([nn.LayerNorm(hidden_dim) for _ in range(len(self.extra_input_dim))])  # type: ignore[union-attr, operator]

        elif self.globally_joined_norm:
            self.register_module("ln_global_joined_norm", nn.LayerNorm(hidden_dim))

        return

    def _global_sum(
        self, x_t: torch.Tensor, h_t: torch.Tensor, *extra_input: torch.Tensor
    ) -> torch.Tensor:
        preactivated_mask = self.ln_global_norm[0](  # type: ignore[operator, index]
            F.linear(x_t, self.weight_ih)  # type: ignore[arg-type]
        ) + self.ln_global_norm[
            1
        ](  # type: ignore[index]
            F.linear(h_t, self.weight_hh)  # type: ignore[arg-type]
        )  # type: ignore[operator, index, arg-type]

        if hasattr(self, 'extra_input_dim'):
            for id, extra_input_t in enumerate(extra_input):
                preactivated_mask += self.ln_global_norm[id + 2](  # type: ignore[operator, index]
                    F.linear(extra_input_t, self.extra_parameter_list[id])
                )

        if self.bias:
            preactivated_mask += self.bias_vector

        return preactivated_mask

    def _naive_sum(
        self,
        x_t: torch.Tensor,
        h_t: torch.Tensor,
        *extra_input,
        exclude_bias: bool = False,
    ) -> torch.Tensor:
        preactivated_mask = F.linear(x_t, self.weight_ih) + F.linear(  # type: ignore[arg-type]
            h_t, self.weight_hh  # type: ignore[arg-type]
        )
        if hasattr(self, 'extra_input_dim'):
            for i, extra_input_t in enumerate(extra_input):
                preactivated_mask += F.linear(
                    extra_input_t, self.extra_parameter_list[i]
                )

            if self.bias and not exclude_bias:
                preactivated_mask += self.bias_vector

        return preactivated_mask

    def forward(
        self, x_t: torch.Tensor, h_t: torch.Tensor, *extra_input: torch.Tensor
    ) -> torch.Tensor:
        """Perform the forward pass through the custom LSTM gate.

        Args:
            x_t (torch.Tensor): Input tensor.
            h_t (torch.Tensor): Hidden state tensor.
            extra_input (torch.Tensor): Extra input tensors.

        Returns:
            torch.Tensor: Output tensor after passing through the LSTM gate.
        """
        # See the LSTM paper for the equations : https://www.bioinf.jku.at/publications/older/2604.pdf
        # Compute the mask for the input
        # This is used to mask the input to the LSTM

        # See the Globally Joined Norm - Layer Norm application
        if self.globally_joined_norm:
            return self.activation(
                self.ln_global_joined_norm(  # type: ignore[operator, index]
                    self._naive_sum(x_t, h_t, *extra_input, exclude_bias=True)
                )
            )

        if self.globa_norm:
            return self.activation(self._global_sum(x_t, h_t, *extra_input))

        # If none, just return the naive sum
        # Outshape : (batch_size, hidden_dim)
        return self.activation(
            self._naive_sum(x_t, h_t, *extra_input, exclude_bias=False)
        )
