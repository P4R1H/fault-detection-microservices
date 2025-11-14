"""
Multimodal fusion with cross-modal attention.

Architecture:
1. Separate encoders per modality (Chronos/TCN, Logs, GCN)
2. Cross-modal attention for dynamic importance weighting
3. Combined representation for RCA

Reference:
- "FAMOS" (ICSE 2025): Gaussian-attention multimodal fusion
- "MULAN" (WWW 2024): Log-tailored LM with contrastive learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism.

    Learns to weight different modalities based on:
    - Inter-modality relationships
    - Task-specific importance
    - Dynamic contextual relevance

    Features:
    - Multi-head attention (8 heads typical)
    - Scaled dot-product attention
    - Dropout for regularization
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply cross-modal attention.

        Args:
            query: (batch, seq_len, embed_dim) query modality
            key: (batch, seq_len, embed_dim) key modality
            value: (batch, seq_len, embed_dim) value modality
            return_attention: Return attention weights for visualization

        Returns:
            (batch, seq_len, embed_dim) attended output
            Optional: attention weights
        """
        attn_output, attn_weights = self.multihead_attn(
            query, key, value,
            need_weights=return_attention
        )

        if return_attention:
            return attn_output, attn_weights
        return attn_output, None


class MultimodalFusion(nn.Module):
    """
    Complete multimodal fusion pipeline.

    Architecture:
    ```
    Metrics → Encoder → 256-dim ─┐
                                   │
    Logs → Encoder → 256-dim ──────┼→ Cross-Attention → 512-dim → RCA Head
                                   │
    Traces → Encoder → 128-dim ───┘
    ```

    Fusion Strategies:
    - Early: Concatenate raw features (baseline)
    - Late: Separate predictions + ensemble (baseline)
    - Intermediate: This approach (SOTA)
    """

    def __init__(
        self,
        metrics_encoder: nn.Module,
        logs_encoder: nn.Module,
        traces_encoder: nn.Module,
        fusion_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        fusion_strategy: str = 'intermediate'
    ):
        super().__init__()

        self.metrics_encoder = metrics_encoder
        self.logs_encoder = logs_encoder
        self.traces_encoder = traces_encoder
        self.fusion_strategy = fusion_strategy

        if fusion_strategy == 'intermediate':
            # Cross-modal attention
            self.cross_attention = CrossModalAttention(
                embed_dim=fusion_dim,
                num_heads=num_heads,
                dropout=dropout
            )

            # Projection layers to common dimension
            self.metrics_proj = nn.Linear(256, fusion_dim)
            self.logs_proj = nn.Linear(256, fusion_dim)
            self.traces_proj = nn.Linear(128, fusion_dim)

        elif fusion_strategy == 'early':
            # Simple concatenation
            total_dim = 256 + 256 + 128
            self.fusion_layer = nn.Linear(total_dim, fusion_dim)

        elif fusion_strategy == 'late':
            # Separate heads
            self.metrics_head = nn.Linear(256, fusion_dim)
            self.logs_head = nn.Linear(256, fusion_dim)
            self.traces_head = nn.Linear(128, fusion_dim)

        # Final layers
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        metrics: torch.Tensor,
        logs: Dict,
        traces: Tuple[torch.Tensor, torch.Tensor],
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse multimodal data.

        Args:
            metrics: (batch, seq_len, features) metrics tensor
            logs: Dictionary with log data
            traces: (node_features, edge_index) graph data
            return_attention: Return attention weights for visualization

        Returns:
            Dictionary with:
            - 'fused': (batch, fusion_dim) fused representation
            - 'attention': Attention weights if requested
            - 'modality_embeddings': Individual embeddings
        """
        # TODO: Implement multimodal fusion
        raise NotImplementedError("Phase 8 implementation")


# TODO: Add early fusion baseline
# TODO: Add late fusion baseline
# TODO: Add attention visualization utilities
# TODO: Add modality dropout for robustness
