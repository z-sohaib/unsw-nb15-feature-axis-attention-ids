from __future__ import annotations

import torch
from torch import nn


class CNNClassifier(nn.Module):
    def __init__(
        self,
        input_features: int,
        num_classes: int,
        conv_channels: int,
        kernel_size: int,
        dense_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=input_features,
                out_channels=conv_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(conv_channels, dense_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x.transpose(1, 2)).squeeze(-1)
        return self.classifier(h)


class RecurrentClassifier(nn.Module):
    def __init__(
        self,
        input_features: int,
        num_classes: int,
        lstm_hidden: int,
        lstm_layers: int,
        dense_hidden: int,
        dropout: float,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        output_features = lstm_hidden * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(output_features),
            nn.Linear(output_features, dense_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        return self.classifier(h[:, -1, :])


class CNNRecurrentClassifier(nn.Module):
    def __init__(
        self,
        input_features: int,
        num_classes: int,
        conv_channels: int,
        kernel_size: int,
        lstm_hidden: int,
        lstm_layers: int,
        dense_hidden: int,
        dropout: float,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=input_features,
                out_channels=conv_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        output_features = lstm_hidden * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(output_features),
            nn.Linear(output_features, dense_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.lstm(h)
        return self.classifier(h[:, -1, :])
