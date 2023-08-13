"""Predictive LSTM Network Module.

This module implements a predictive LSTM network, a deep architecture designed to capture complex temporal patterns in sequential data. It supports optional features such as skip connections, layer normalization, and conditional cell state.

References:
    [2] "Generating Sequences With Recurrent Neural Network"
        https://arxiv.org/abs/1308.0850

Example:
    ```python
    import torch
    import torch.nn as nn
    from predictive_lstm_module import PredictiveLSTM

    input_size = 32
    hidden_size = 64
    num_layers = 4

    # Create a predictive LSTM instance
    predictive_lstm = PredictiveLSTM(
        n_layers=num_layers,
        input_size=input_size,
        hidden_size=hidden_size,
        skip_connections=True,
        global_joint_layer_norm=True,
    )

    # Forward pass through the predictive LSTM
    seq_length = 10
    batch_size = 16
    input_sequence = torch.rand(batch_size, seq_length, input_size)
    output_sequence = predictive_lstm(input_sequence)
    ```
"""


import torch
import torch.nn as nn

from .lstm_cells import LSTMCell

__all__ = ['PredictiveLSTM']


class PredictiveLSTM(nn.Module):
    """Predictive LSTM Network.

    This class implements a predictive LSTM network, a deep architecture designed to capture complex temporal patterns in sequential data. It supports optional features such as skip connections, layer normalization, and conditional cell state.

    Args:
        n_layers (int): Number of LSTM layers in the network.
        input_size (int): Size of the input tensor along the input feature dimension.
        hidden_size (int): Size of the hidden state tensor along the hidden feature dimension.
        bias (bool, optional): Whether to include bias terms. Defaults to True.
        skip_connections (bool, optional): Whether to use skip connections between layers. Defaults to False.
        conditional_cell_state (bool, optional): Whether to condition on the previous cell state. Defaults to False.
        global_layer_norm (bool, optional): Whether to apply global layer normalization. Defaults to False.
        global_joint_layer_norm (bool, optional): Whether to apply globally joined layer normalization. Defaults to True.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Raises:
        ValueError: If n_layers is not a positive integer.

    References:
        [2] "Generating Sequences With Recurrent Neural Network"
            https://arxiv.org/abs/1308.0850

    Example:
        ```python
        import torch
        import torch.nn as nn
        from predictive_lstm_module import PredictiveLSTM

        input_size = 32
        hidden_size = 64
        num_layers = 4

        # Create a predictive LSTM instance
        predictive_lstm = PredictiveLSTM(
            n_layers=num_layers,
            input_size=input_size,
            hidden_size=hidden_size,
            skip_connections=True,
            global_joint_layer_norm=True,
        )

        # Forward pass through the predictive LSTM
        seq_length = 10
        batch_size = 16
        input_sequence = torch.rand(batch_size, seq_length, input_size)
        output_sequence = predictive_lstm(input_sequence)
        ```
    """

    def __init__(
        self,
        n_layers: int,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        skip_connections: bool = False,
        conditional_cell_state: bool = False,
        global_layer_norm: bool = False,
        global_joint_layer_norm: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the PredictiveLSTM network.

        Args:
            n_layers (int): Number of LSTM layers in the network.
            input_size (int): Size of the input tensor along the input feature dimension.
            hidden_size (int): Size of the hidden state tensor along the hidden feature dimension.
            bias (bool, optional): Whether to include bias terms. Defaults to True.
            skip_connections (bool, optional): Whether to use skip connections between layers. Defaults to False.
            conditional_cell_state (bool, optional): Whether to condition on the previous cell state. Defaults to False.
            global_layer_norm (bool, optional): Whether to apply global layer normalization. Defaults to False.
            global_joint_layer_norm (bool, optional): Whether to apply globally joined layer normalization. Defaults to True.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            ValueError: If n_layers is not a positive integer.
        """
        super().__init__(*args, **kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size

        assert n_layers > 0, "Number of layers must be greater than 0."
        self.n_layers = n_layers
        self.bias = bias
        self.conditional_cell_state = conditional_cell_state
        self.global_layer_norm = global_layer_norm
        self.skip_connections = skip_connections
        self.global_joint_layer_norm = global_joint_layer_norm

        # Instantiate LSTM cells
        self.lstm_cells = self._build_lstm_cells()

    def _build_lstm_cells(self) -> nn.ModuleList:
        lstm_cells = nn.ModuleList()

        # Add first LSTM cell | No skip connections in the first layer
        lstm_cells.append(
            LSTMCell(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                bias=self.bias,
                conditional_cell_state=self.conditional_cell_state,
                global_layer_norm=self.global_layer_norm,
                global_joint_layer_norm=self.global_joint_layer_norm,
                extra_forcing_input_dim=None,
            )
        )

        # Add skip connections  from input to each layer
        for idx in range(self.n_layers - 1):
            if self.skip_connections:
                lstm_cells.append(
                    LSTMCell(
                        input_size=self.hidden_size,
                        hidden_size=self.hidden_size,
                        bias=self.bias,
                        conditional_cell_state=self.conditional_cell_state,
                        global_layer_norm=self.global_layer_norm,
                        global_joint_layer_norm=self.global_joint_layer_norm,
                        extra_forcing_input_dim=[self.input_size],
                    )
                )
            else:
                lstm_cells.append(
                    LSTMCell(
                        input_size=self.hidden_size,
                        hidden_size=self.hidden_size,
                        bias=self.bias,
                        conditional_cell_state=self.conditional_cell_state,
                        global_layer_norm=self.global_layer_norm,
                        global_joint_layer_norm=self.global_joint_layer_norm,
                        extra_forcing_input_dim=None,
                    )
                )

        return lstm_cells

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the PredictiveLSTM network.

        Args:
            x_t (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_size * n_layers) if skip_connections are enabled,
                          or (batch_size, seq_len, hidden_size) if skip_connections are disabled.
        """
        return self._condition_forward(x_t)

    def _condition_forward(self, x_t: torch.Tensor) -> torch.Tensor:
        # Reocrd lstm outputs across space and time
        # Where 2 is for hidden[0] and cell state[1]
        lstm_intermediate_outputs = torch.zeros(
            self.n_layers, 2, x_t.shape[0], x_t.shape[1], self.hidden_size
        )

        # Double loop across layer and seq to compute hidden and cell states
        for idx, lstm_cell in enumerate(self.lstm_cells):
            for time_step in range(x_t.shape[1]):
                # Time step 0 is special case for every layer sicnce it overwrites the initital hidden and cell states
                if time_step == 0:
                    # The first layer doesn't take input X_t but rest take h_t of previous layer.
                    if idx == 0:
                        (
                            lstm_intermediate_outputs[idx, 0, :, time_step, :],
                            lstm_intermediate_outputs[idx, 1, :, time_step, :],
                        ) = lstm_cell(
                            x_t[:, time_step, :],
                            lstm_intermediate_outputs[0, 0, :, time_step, :],
                            lstm_intermediate_outputs[0, 1, :, time_step, :],
                        )
                    # The rest of the layers takes the previous layer's hidden as input
                    else:
                        # Pass X_t along with h_t of previous layer if skip connections are enabled
                        if self.skip_connections:
                            (
                                lstm_intermediate_outputs[idx, 0, :, time_step, :],
                                lstm_intermediate_outputs[idx, 1, :, time_step, :],
                            ) = lstm_cell(
                                lstm_intermediate_outputs[
                                    idx - 1, 0, :, time_step, :
                                ],  # previous layer's hidden state as input
                                lstm_intermediate_outputs[
                                    idx, 0, :, time_step, :
                                ],  # current layer's hidden state as input
                                lstm_intermediate_outputs[
                                    idx, 1, :, time_step, :
                                ],  # current layer's cell state as input
                                x_t[:, time_step, :],
                            )

                        else:
                            # Do not pass X_t if skip connections are disabled
                            (
                                lstm_intermediate_outputs[idx, 0, :, time_step, :],
                                lstm_intermediate_outputs[idx, 1, :, time_step, :],
                            ) = lstm_cell(
                                lstm_intermediate_outputs[
                                    idx - 1, 0, :, time_step, :
                                ],  # previous layer's hidden state as input
                                lstm_intermediate_outputs[
                                    idx, 0, :, time_step, :
                                ],  # current layer's hidden state as input
                                lstm_intermediate_outputs[
                                    idx, 1, :, time_step, :
                                ],  # current layer's cell state as input
                            )
                # For all other time steps, the previous layer's hidden and cell states are used as input
                else:
                    # If  first layer, pass X_t as input for other time steps
                    # No skips
                    if idx == 0:
                        (
                            lstm_intermediate_outputs[idx, 0, :, time_step, :],
                            lstm_intermediate_outputs[idx, 1, :, time_step, :],
                        ) = lstm_cell(
                            x_t[:, time_step, :],  # First Layer takes X_t as input
                            lstm_intermediate_outputs[
                                idx, 0, :, time_step - 1, :
                            ],  # previous timestep hidden state as input hidden state
                            lstm_intermediate_outputs[
                                idx, 1, :, time_step - 1, :
                            ],  # previous layer timestep cell state as input
                        )
                    else:
                        if self.skip_connections:
                            # Pass X_t along with h_t of previous layer if skip connections are enabled
                            (
                                lstm_intermediate_outputs[idx, 0, :, time_step, :],
                                lstm_intermediate_outputs[idx, 1, :, time_step, :],
                            ) = lstm_cell(
                                lstm_intermediate_outputs[
                                    idx - 1, 0, :, time_step, :
                                ],  # previous layer's hidden state as input
                                lstm_intermediate_outputs[
                                    idx, 0, :, time_step - 1, :
                                ],  # previous timestep hidden state as input hidden state
                                lstm_intermediate_outputs[
                                    idx, 1, :, time_step - 1, :
                                ],  # previous layer timestep cell state as input
                                x_t[:, time_step, :],
                            )
                        else:
                            # Do not pass X_t if skip connections are disabled
                            (
                                lstm_intermediate_outputs[idx, 0, :, time_step, :],
                                lstm_intermediate_outputs[idx, 1, :, time_step, :],
                            ) = lstm_cell(
                                lstm_intermediate_outputs[
                                    idx - 1, 0, :, time_step, :
                                ],  # previous layer's hidden state as input
                                lstm_intermediate_outputs[
                                    idx, 0, :, time_step - 1, :
                                ],  # previous timestep hidden state as input hidden state
                                lstm_intermediate_outputs[
                                    idx, 1, :, time_step - 1, :
                                ],  # previous layer timestep cell state as input
                            )

        if self.skip_connections:
            # Return layer hidden state as (B ,T , H * n_layer)
            return (
                lstm_intermediate_outputs[:, 0, :, :, :]
                .permute(2, 3, 1, 0)
                .reshape(x_t.shape[0], x_t.shape[1], self.hidden_size * self.n_layers)
            )

        # Return last layer hidden state as (B ,T , H) if not skip connections
        return lstm_intermediate_outputs[-1, 0, :, :, :].reshape(
            x_t.shape[0], x_t.shape[1], self.hidden_size
        )
