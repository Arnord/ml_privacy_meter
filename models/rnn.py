import torch
from torch import nn
from torch.nn import functional as F


class LSTM(nn.Module):
    """Simple LSTM for time series classification."""

    def __init__(self, input_size, output_size, hidden_size=32 ,num_layers=1):
        """
        :param input_size: Number of features in input (e.g., channels per time step)
        :param hidden_size: Number of features in hidden state
        :param num_layers: Number of stacked LSTM layers
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass for LSTM.

        :param x: Input tensor of shape (batch_size, seq_len, input_size)
        :return: Output tensor of shape (batch_size, num_classes)
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_len, hidden_size)
        out = out[:, -1, :]              # take last time step
        out = self.fc(out)               # (batch_size, output_size)
        return out
