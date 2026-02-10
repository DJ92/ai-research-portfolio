"""Scaled dot-product attention and multi-head attention.

Implements the core attention mechanism from "Attention is All You Need" (Vaswani et al., 2017).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    This is the fundamental building block of transformers. The scaling by sqrt(d_k)
    prevents dot products from growing too large in magnitude, which would push the
    softmax into regions with extremely small gradients.
    """

    def __init__(self, dropout: float = 0.1):
        """Initialize scaled dot-product attention.

        Args:
            dropout: Dropout probability applied to attention weights
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention.

        Args:
            query: Query tensor (batch, seq_len, d_k)
            key: Key tensor (batch, seq_len, d_k)
            value: Value tensor (batch, seq_len, d_v)
            mask: Optional mask tensor (batch, seq_len, seq_len)
                  True values are masked out (set to -inf before softmax)

        Returns:
            output: Attention output (batch, seq_len, d_v)
            attention_weights: Attention weights (batch, seq_len, seq_len)
        """
        d_k = query.size(-1)

        # Compute attention scores: QK^T / sqrt(d_k)
        # Shape: (batch, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask if provided (e.g., for causal/decoder self-attention)
        if mask is not None:
            scores = scores.masked_fill(mask == True, float('-inf'))

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights)

        # Compute weighted sum of values
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism.

    Instead of performing a single attention function with d_model-dimensional keys,
    values, and queries, we linearly project them h times with different learned
    projections to d_k, d_k, and d_v dimensions respectively.

    This allows the model to jointly attend to information from different representation
    subspaces at different positions.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        """Initialize multi-head attention.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        # Scaled dot-product attention
        self.attention = ScaledDotProductAttention(dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply multi-head attention.

        Args:
            query: Query tensor (batch, seq_len, d_model)
            key: Key tensor (batch, seq_len, d_model)
            value: Value tensor (batch, seq_len, d_model)
            mask: Optional mask (batch, 1, seq_len, seq_len) or (batch, seq_len, seq_len)

        Returns:
            output: Multi-head attention output (batch, seq_len, d_model)
            attention_weights: Average attention weights across heads (batch, seq_len, seq_len)
        """
        batch_size = query.size(0)

        # 1. Linear projections in batch: (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2. Split into multiple heads: (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Adjust mask dimensions if needed
        if mask is not None and mask.dim() == 3:
            # Add head dimension: (batch, seq_len, seq_len) -> (batch, 1, seq_len, seq_len)
            mask = mask.unsqueeze(1)

        # 3. Apply attention on all heads in parallel
        # Q, K, V shape: (batch, num_heads, seq_len, d_k)
        # Reshape for attention: (batch * num_heads, seq_len, d_k)
        Q_reshape = Q.contiguous().view(batch_size * self.num_heads, -1, self.d_k)
        K_reshape = K.contiguous().view(batch_size * self.num_heads, -1, self.d_k)
        V_reshape = V.contiguous().view(batch_size * self.num_heads, -1, self.d_k)

        # Expand mask for all heads
        if mask is not None:
            mask_reshape = mask.expand(batch_size, self.num_heads, mask.size(-2), mask.size(-1))
            mask_reshape = mask_reshape.contiguous().view(batch_size * self.num_heads, mask.size(-2), mask.size(-1))
        else:
            mask_reshape = None

        # Apply attention
        attn_output, attn_weights = self.attention(Q_reshape, K_reshape, V_reshape, mask_reshape)

        # 4. Concatenate heads: (batch * num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        attn_output = attn_output.view(batch_size, self.num_heads, -1, self.d_k)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)

        # 5. Apply output projection
        output = self.W_o(attn_output)
        output = self.dropout(output)

        # Average attention weights across heads for visualization
        attn_weights = attn_weights.view(batch_size, self.num_heads, attn_weights.size(-2), attn_weights.size(-1))
        attn_weights = attn_weights.mean(dim=1)

        return output, attn_weights

    def get_attention_maps(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get attention weights for all heads separately (for visualization).

        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Optional mask

        Returns:
            Attention weights per head (batch, num_heads, seq_len, seq_len)
        """
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        d_k = self.d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == True, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)

        return attention_weights
