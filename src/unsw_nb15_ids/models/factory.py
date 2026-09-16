from __future__ import annotations

from torch import nn

from unsw_nb15_ids.config import ModelConfig
from unsw_nb15_ids.models.baselines import (
    CNNClassifier,
    CNNRecurrentClassifier,
    RecurrentClassifier,
)
from unsw_nb15_ids.models.feature_sequence import (
    FTTransformerClassifier,
    FeatureSequenceMultiScaleBiLSTMAttention,
    SimpleFeatureTransformerClassifier,
)
from unsw_nb15_ids.models.multiscale_bilstm_attention import MultiScaleBiLSTMAttention


def build_model(
    config: ModelConfig,
    input_features: int,
    num_classes: int,
) -> nn.Module:
    primary_kernel = config.kernel_sizes[0]

    if config.name == "cnn":
        return CNNClassifier(
            input_features=input_features,
            num_classes=num_classes,
            conv_channels=config.conv_channels,
            kernel_size=primary_kernel,
            dense_hidden=config.dense_hidden,
            dropout=config.dropout,
        )
    if config.name == "lstm":
        return RecurrentClassifier(
            input_features=input_features,
            num_classes=num_classes,
            lstm_hidden=config.lstm_hidden,
            lstm_layers=config.lstm_layers,
            dense_hidden=config.dense_hidden,
            dropout=config.dropout,
            bidirectional=False,
        )
    if config.name == "bilstm":
        return RecurrentClassifier(
            input_features=input_features,
            num_classes=num_classes,
            lstm_hidden=config.lstm_hidden,
            lstm_layers=config.lstm_layers,
            dense_hidden=config.dense_hidden,
            dropout=config.dropout,
            bidirectional=True,
        )
    if config.name == "cnn_lstm":
        return CNNRecurrentClassifier(
            input_features=input_features,
            num_classes=num_classes,
            conv_channels=config.conv_channels,
            kernel_size=primary_kernel,
            lstm_hidden=config.lstm_hidden,
            lstm_layers=config.lstm_layers,
            dense_hidden=config.dense_hidden,
            dropout=config.dropout,
            bidirectional=False,
        )
    if config.name == "cnn_bilstm":
        return CNNRecurrentClassifier(
            input_features=input_features,
            num_classes=num_classes,
            conv_channels=config.conv_channels,
            kernel_size=primary_kernel,
            lstm_hidden=config.lstm_hidden,
            lstm_layers=config.lstm_layers,
            dense_hidden=config.dense_hidden,
            dropout=config.dropout,
            bidirectional=True,
        )
    if config.name == "ms_cnn_bilstm_attention":
        return MultiScaleBiLSTMAttention(
            input_features=input_features,
            num_classes=num_classes,
            conv_channels=config.conv_channels,
            kernel_sizes=config.kernel_sizes,
            lstm_hidden=config.lstm_hidden,
            lstm_layers=config.lstm_layers,
            attention_heads=config.attention_heads,
            dense_hidden=config.dense_hidden,
            dropout=config.dropout,
        )
    if config.name == "feature_ms_cnn_bilstm_attention":
        return FeatureSequenceMultiScaleBiLSTMAttention(
            input_features=input_features,
            num_classes=num_classes,
            conv_channels=config.conv_channels,
            kernel_sizes=config.kernel_sizes,
            lstm_hidden=config.lstm_hidden,
            lstm_layers=config.lstm_layers,
            attention_heads=config.attention_heads,
            dense_hidden=config.dense_hidden,
            dropout=config.dropout,
        )
    if config.name == "ft_transformer":
        return FTTransformerClassifier(
            input_features=input_features,
            num_classes=num_classes,
            transformer_dim=config.transformer_dim or config.dense_hidden,
            transformer_layers=config.transformer_layers,
            attention_heads=config.attention_heads,
            dense_hidden=config.dense_hidden,
            dropout=config.dropout,
        )
    if config.name == "simple_transformer":
        return SimpleFeatureTransformerClassifier(
            input_features=input_features,
            num_classes=num_classes,
            transformer_dim=config.transformer_dim or config.dense_hidden,
            transformer_layers=config.transformer_layers,
            attention_heads=config.attention_heads,
            dense_hidden=config.dense_hidden,
            dropout=config.dropout,
        )

    raise ValueError(f"Unsupported model name: {config.name}")
