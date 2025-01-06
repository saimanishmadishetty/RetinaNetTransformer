# model_components.py

import torch.nn as nn


class StockPredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(StockPredictionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Fully connected on last time step
        return out
