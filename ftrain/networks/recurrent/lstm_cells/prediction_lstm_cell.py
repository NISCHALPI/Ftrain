"""Module: LSTM Cell with Layer Normalization and Gating Mechanisms.

This module defines an LSTM cell implementation enriched with optional layer normalization and gating mechanisms. The LSTM cell is a fundamental component of sequence modeling tasks, capable of capturing long-term dependencies within sequential data.

References:
[2] "Generating Sequences With Recurrent Neural Network"
    https://arxiv.org/abs/1308.0850

Classes:
- LSTMCell: A customizable LSTM cell with support for teacher forcing, conditional cell state, and layer normalization.

Usage Example:
```python
import torch
import torch.nn as nn
from custom_lstm_module import LSTMCell

input_size = 32
hidden_size = 64

# Create an LSTM cell instance
lstm_cell = LSTMCell(
    input_size=input_size,
    hidden_size=hidden_size,
    teacher_forcing=True,
    global_joint_layer_norm=True,
)

# Forward pass through the LSTM cell
x_t = torch.rand(batch_size, input_size)
h_t_prev = torch.rand(batch_size, hidden_size)
c_t_prev = torch.rand(batch_size, hidden_size)
h_t_next, c_t_next = lstm_cell(x_t, h_t_prev, c_t_prev)

"""

import warnings

import torch
import torch.nn as nn

from .base_gate import _Gate

__all__ = ["LSTMCell"]


class LSTMCell(nn.Module):
    """LSTM Cell with Optional Layer Normalization and Gating Mechanisms.

    This class defines an LSTM cell enriched with advanced features including optional layer normalization and gating mechanisms. The LSTM cell is a core building block for modeling sequential data, excelling at capturing long-range dependencies and patterns.

    Args:
        input_size (int): Size of the input tensor along the input feature dimension.
        hidden_size (int): Size of the hidden state tensor along the hidden feature dimension.
        bias (bool, optional): Whether to include bias terms. Defaults to True.
        teacher_forcing (bool, optional): Whether to use teacher forcing during training. Defaults to False.
        teacher_input_dim (torch.Size | tuple | int | None, optional): Dimension of the teacher input tensor. Defaults to None.
        conditional_cell_state (bool, optional): Whether to condition on the previous cell state. Defaults to False.
        global_layer_norm (bool, optional): Apply global layer normalization. Defaults to False.
        global_joint_layer_norm (bool, optional): Apply globally joined layer normalization. Defaults to True.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Raises:
        ValueError: If teacher_forcing is True but teacher_input_dim is not provided.

    References:
        [2] "Generating Sequences With Recurrent Neural Network"
            https://arxiv.org/abs/1308.0850

    Example:
        ```python
        import torch
        import torch.nn as nn
        from custom_lstm_module import LSTMCell

        input_size = 32
        hidden_size = 64

        # Create an LSTM cell instance
        lstm_cell = LSTMCell(
            input_size=input_size,
            hidden_size=hidden_size,
            teacher_forcing=True,
            global_joint_layer_norm=True,
        )

        # Forward pass through the LSTM cell
        x_t = torch.rand(batch_size, input_size)
        h_t_prev = torch.rand(batch_size, hidden_size)
        c_t_prev = torch.rand(batch_size, hidden_size)
        h_t_next, c_t_next = lstm_cell(x_t, h_t_prev, c_t_prev)
        ```
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        conditional_cell_state: bool = False,
        extra_forcing_input_dim: list[int] | None = None,
        global_layer_norm: bool = False,
        global_joint_layer_norm: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the LSTM Cell with optional layer normalization and gating mechanisms.

        Args:
            input_size (int): Size of the input tensor along the input feature dimension.
            hidden_size (int): Size of the hidden state tensor along the hidden feature dimension.
            bias (bool, optional): Whether to include bias terms. Defaults to True.
            extra_forcing_input_dim (list[int] | None, optional): Dimension of the extra forcing input tensor.
              Defaults to None. Can be used for teacher forcing and skipconnections.
            conditional_cell_state (bool, optional): Whether to condition on the previous cell state. Defaults to False.
            global_layer_norm (bool, optional): Apply global layer normalization. Defaults to False.
            global_joint_layer_norm (bool, optional): Apply globally joined layer normalization. Defaults to True.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            ValueError: If teacher_forcing is True but teacher_input_dim is not provided.
        """
        super().__init__(*args, **kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # Can be generalized to teacher forcing with multiple inputs
        if extra_forcing_input_dim is not None and len(extra_forcing_input_dim) > 0:
            assert all(
                [dim > 0 for dim in extra_forcing_input_dim]
            ), "extra_forcing_input_dim must be a list of positive integers"
            self.extra_forcing_input_dim = extra_forcing_input_dim
            warnings.warn(
                "Forward method expects following arguments since extra_forcing_input_dim is not None: x_t, h_t , c_t, *extra_forcing_input[ordered same as extra_forcing_input_dim]",
                stacklevel=1,
            )

        # Norms initialization
        self.global_layer_norm = global_layer_norm
        self.global_joint_layer_norm = global_joint_layer_norm

        # Contition on Previous Cell State
        self.conditonal_cell_state = conditional_cell_state

        # Initialize Gates
        self._gate_init()

        return

    def _gate_init(self) -> None:
        """Initialize the LSTM gates."""
        # Set Forget Gate

        # Set Extra Input Per Gate | condition on extra inputs
        extra_input_per_gate = []
        if self.conditonal_cell_state:
            extra_input_per_gate.append(self.hidden_size)
        if hasattr(self, 'extra_forcing_input_dim'):
            extra_input_per_gate.extend(self.extra_forcing_input_dim)

        self.ForgetGate = _Gate(
            input_dim=self.input_size,
            hidden_dim=self.hidden_size,
            bias=self.bias,
            global_norm=self.global_layer_norm,
            globally_joined_norm=self.global_joint_layer_norm,
            activation=nn.Sigmoid(),
            extra_input_dim=extra_input_per_gate,
        )

        # Set Input Gate
        self.InputGate = _Gate(
            input_dim=self.input_size,
            hidden_dim=self.hidden_size,
            bias=self.bias,
            global_norm=self.global_layer_norm,
            globally_joined_norm=self.global_joint_layer_norm,
            activation=nn.Sigmoid(),
            extra_input_dim=extra_input_per_gate,
        )

        # Set Output Gate
        self.OutputGate = _Gate(
            input_dim=self.input_size,
            hidden_dim=self.hidden_size,
            bias=self.bias,
            global_norm=self.global_layer_norm,
            globally_joined_norm=self.global_joint_layer_norm,
            activation=nn.Sigmoid(),
            extra_input_dim=extra_input_per_gate,
        )

        # Do Not Condition on Previous Cell State On This Cell Input
        # Do Not Condition on Extra Inputs On This Cell Input
        self.CellInput = _Gate(
            input_dim=self.input_size,
            hidden_dim=self.hidden_size,
            bias=self.bias,
            global_norm=self.global_layer_norm,
            globally_joined_norm=self.global_joint_layer_norm,
            activation=nn.Tanh(),
            extra_input_dim=None,
        )

        return

    def _conditionally_forward(
        self,
        gate: nn.Module,
        x_t: torch.Tensor,
        h_t: torch.Tensor,
        c_t: torch.Tensor,
        *extra_inputs,
    ) -> torch.Tensor:
        if self.conditonal_cell_state:
            return gate(x_t, h_t, c_t, *extra_inputs)
        return gate(x_t, h_t, *extra_inputs)

    def forward(
        self, x_t: torch.Tensor, h_t: torch.Tensor, c_t: torch.Tensor, *extra_inputs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform a forward pass through the LSTM cell.

        Args:
            x_t (torch.Tensor): Input tensor for the current time step.
                Shape should be (batch_size, input_size).
            h_t (torch.Tensor): Previous hidden state tensor.
                Shape should be (batch_size, hidden_size).
            c_t (torch.Tensor): Previous cell state tensor.
                Shape should be (batch_size, hidden_size).
            *extra_inputs: Variable length argument list for additional input tensors. Must have same dimension as (BatchSize, extra_forcing_input_dim[i])

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing the next hidden state and cell state.

        Expected Shapes:
            x_t: (batch_size, input_size)
            h_t: (batch_size, hidden_size)
            c_t: (batch_size, hidden_size)
            extra_inputs: Varying shapes based on conditional teacher forcing settings.
        """
        i = self._conditionally_forward(self.InputGate, x_t, h_t, c_t, *extra_inputs)
        f = self._conditionally_forward(self.ForgetGate, x_t, h_t, c_t, *extra_inputs)
        o = self._conditionally_forward(self.OutputGate, x_t, h_t, c_t, *extra_inputs)

        # next cell state
        c_t_next = f * c_t + i * self.CellInput(x_t, h_t)

        # next hidden state
        h_t_next = o * torch.tanh(c_t_next)

        return h_t_next, c_t_next
