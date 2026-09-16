from __future__ import annotations

import torch
from torch import nn


class FeatureSequenceMultiScaleBiLSTMAttention(nn.Module):
    """Multi-scale CNN-BiLSTM-attention over feature positions.

    The tabular loader emits tensors shaped as batch, 1, features. This model
    treats each encoded feature position as one sequence step with one channel,
    so convolution and recurrent layers operate over the feature axis instead of
    over a length-1 pseudo-time axis.
    """

    def __init__(
        self,
        input_features: int,
        num_classes: int,
        conv_channels: int,
        kernel_sizes: tuple[int, ...],
        lstm_hidden: int,
        lstm_layers: int,
        attention_heads: int,
        dense_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if not kernel_sizes:
            raise ValueError("At least one convolution kernel size is required.")

        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=1,
                        out_channels=conv_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.BatchNorm1d(conv_channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for kernel_size in kernel_sizes
            ]
        )
        cnn_features = conv_channels * len(kernel_sizes)
        self.lstm = nn.LSTM(
            input_size=cnn_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        attention_dim = lstm_hidden * 2
        if attention_dim % attention_heads != 0:
            raise ValueError("BiLSTM output dimension must be divisible by attention_heads.")
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(attention_dim),
            nn.Linear(attention_dim, dense_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        branch_outputs = [branch(h) for branch in self.branches]
        h = torch.cat(branch_outputs, dim=1)
        h = h.transpose(1, 2)
        h, _ = self.lstm(h)
        attended, _ = self.attention(h, h, h, need_weights=False)
        pooled = attended.mean(dim=1)
        return self.classifier(pooled)


class FTTransformerClassifier(nn.Module):
    """FT-Transformer-style classifier for tabular UNSW-NB15 features."""

    def __init__(
        self,
        input_features: int,
        num_classes: int,
        transformer_dim: int,
        transformer_layers: int,
        attention_heads: int,
        dense_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if transformer_dim % attention_heads != 0:
            raise ValueError("transformer_dim must be divisible by attention_heads.")

        self.value_embedding = nn.Linear(1, transformer_dim)
        self.feature_embedding = nn.Parameter(torch.zeros(1, input_features, transformer_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=attention_heads,
            dim_feedforward=dense_hidden * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, dense_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, num_classes),
        )
        nn.init.normal_(self.feature_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = x.transpose(1, 2).unsqueeze(-1)
        tokens = self.value_embedding(values.squeeze(2)) + self.feature_embedding
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        h = torch.cat([cls, tokens], dim=1)
        h = self.encoder(h)
        return self.classifier(h[:, 0])


class SimpleFeatureTransformerClassifier(nn.Module):
    """Vanilla Transformer encoder over encoded feature positions."""

    def __init__(
        self,
        input_features: int,
        num_classes: int,
        transformer_dim: int,
        transformer_layers: int,
        attention_heads: int,
        dense_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if transformer_dim % attention_heads != 0:
            raise ValueError("transformer_dim must be divisible by attention_heads.")

        self.value_embedding = nn.Linear(1, transformer_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, input_features, transformer_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=attention_heads,
            dim_feedforward=dense_hidden,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, dense_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, num_classes),
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = x.transpose(1, 2).unsqueeze(-1)
        h = self.value_embedding(values.squeeze(2)) + self.position_embedding
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        return self.classifier(pooled)
