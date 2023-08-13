"""This module implements the predictive LSTM network.

References:
[2] "Generating Sequences With Recurrent Neural Network"
    https://arxiv.org/abs/1308.0850
"""


import torch.nn as nn
from .lstm_cells import LSTMCell

__all__ = ['PredictiveLSTM']


class PredictiveLSTM(nn.Module):
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
        super().__init__(*args, **kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size

        assert self.n_layers > 0, "Number of layers must be greater than 0."
        self.n_layers = n_layers
        self.bias = bias
        self.conditional_cell_state = conditional_cell_state
        self.global_layer_norm = global_layer_norm
        self.skip_connections = skip_connections

        # Instantiate LSTM cells
        self.lstm_cells = self._build_lstm_cells()

    def _build_lstm_cells(self) -> nn.ModuleList:
        lstm_cells = nn.ModuleList()

        # Add first LSTM cell
        lstm_cells.append()

        pass
