"""
Metrics encoders for time series data.

Implements two approaches for ablation:
1. Chronos-Bolt-Tiny: Foundation model, zero-shot, 100MB
2. TCN: Temporal Convolutional Network, trained, parallelizable
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class ChronosEncoder(nn.Module):
    """
    Chronos-Bolt-Tiny encoder for zero-shot metrics encoding.

    Features:
    - 20M parameters, 100MB memory
    - Zero-shot anomaly detection
    - 250x faster than LSTM
    - Direct multi-step forecasting

    Reference: Amazon Chronos (November 2024)
    HuggingFace: amazon/chronos-bolt-tiny
    """

    def __init__(
        self,
        model_name: str = "amazon/chronos-bolt-tiny",
        embedding_dim: int = 256,
        freeze_backbone: bool = True
    ):
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.freeze_backbone = freeze_backbone

        # TODO: Load Chronos model from HuggingFace
        # self.chronos = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode metrics time series.

        Args:
            x: (batch, seq_len, features) metrics tensor

        Returns:
            (batch, embedding_dim) encoded representation
        """
        # TODO: Implement forward pass
        raise NotImplementedError("Phase 3 implementation")


class TCNEncoder(nn.Module):
    """
    Temporal Convolutional Network encoder.

    Features:
    - Dilated causal convolutions
    - Parallelizable training (3-5x faster than LSTM)
    - Exponentially large receptive field
    - Lightweight (efficient for edge deployment)

    Architecture:
    - 7 layers with exponential dilation [1,2,4,8,16,32,64]
    - Receptive field: 381 timesteps
    - Parameters: <10M

    Reference: "SDN Anomalous Traffic Detection Based on TCN" (2024)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        embedding_dim: int = 256,
        num_layers: int = 7,
        kernel_size: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        # TODO: Build TCN layers with dilated convolutions
        # Dilation pattern: [1, 2, 4, 8, 16, 32, 64]
        # Receptive field = kernel_size * (2^num_layers - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode metrics time series with TCN.

        Args:
            x: (batch, seq_len, features) metrics tensor

        Returns:
            (batch, embedding_dim) encoded representation
        """
        # TODO: Implement TCN forward pass
        raise NotImplementedError("Phase 4 implementation")


class TemporalBlock(nn.Module):
    """
    Single TCN temporal block with dilated convolutions.

    Components:
    - Dilated causal conv1d
    - Weight normalization
    - ReLU activation
    - Dropout
    - Residual connection
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.3
    ):
        super().__init__()
        # TODO: Implement temporal block
        pass


# TODO: Add TCN Autoencoder for unsupervised anomaly detection
# TODO: Add evaluation utilities (inference time, memory usage)
