"""Transformer building blocks: encoder and decoder blocks.

Implements the core transformer architecture components with both
Pre-LN (modern) and Post-LN (original) variants.
"""

import torch
import torch.nn as nn
from typing import Optional

from ..attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Position-wise feed-forward network.

    FFN(x) = GELU(x W_1 + b_1) W_2 + b_2

    Typical expansion ratio: 4x (e.g., 512 -> 2048 -> 512)

    Why GELU over ReLU?
    - Smoother gradients (not piecewise linear)
    - Better empirical performance
    - Stochastic regularization effect
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """Initialize feed-forward network.

        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension (typically 4 * d_model)
            dropout: Dropout probability
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward network.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block (BERT-style).

    Architecture:
    x = x + MultiHeadAttention(LayerNorm(x))  # Pre-LN
    x = x + FFN(LayerNorm(x))

    Uses Pre-LN (modern) by default for better training stability.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        pre_norm: bool = True
    ):
        """Initialize encoder block.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            pre_norm: Use Pre-LN (True) or Post-LN (False)
        """
        super().__init__()
        self.pre_norm = pre_norm

        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply encoder block.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        if self.pre_norm:
            # Pre-LN: x = x + Sublayer(LayerNorm(x))
            # Self-attention
            normed = self.norm1(x)
            attn_output, _ = self.attention(normed, normed, normed, mask)
            x = x + self.dropout(attn_output)

            # Feed-forward
            normed = self.norm2(x)
            ff_output = self.feed_forward(normed)
            x = x + ff_output

        else:
            # Post-LN: x = LayerNorm(x + Sublayer(x))
            # Self-attention
            attn_output, _ = self.attention(x, x, x, mask)
            x = self.norm1(x + self.dropout(attn_output))

            # Feed-forward
            ff_output = self.feed_forward(x)
            x = self.norm2(x + ff_output)

        return x


class TransformerDecoderBlock(nn.Module):
    """Transformer decoder block (GPT-style, decoder-only).

    Architecture:
    x = x + MaskedSelfAttention(LayerNorm(x))  # Causal masking
    x = x + FFN(LayerNorm(x))

    For encoder-decoder models (like T5), would include cross-attention.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        pre_norm: bool = True
    ):
        """Initialize decoder block.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            pre_norm: Use Pre-LN (True) or Post-LN (False)
        """
        super().__init__()
        self.pre_norm = pre_norm

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply decoder block with causal masking.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Causal mask (lower triangular)

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        if self.pre_norm:
            # Pre-LN
            # Masked self-attention
            normed = self.norm1(x)
            attn_output, _ = self.self_attention(normed, normed, normed, mask)
            x = x + self.dropout(attn_output)

            # Feed-forward
            normed = self.norm2(x)
            ff_output = self.feed_forward(normed)
            x = x + ff_output

        else:
            # Post-LN
            # Masked self-attention
            attn_output, _ = self.self_attention(x, x, x, mask)
            x = self.norm1(x + self.dropout(attn_output))

            # Feed-forward
            ff_output = self.feed_forward(x)
            x = self.norm2(x + ff_output)

        return x


class TransformerEncoderDecoderBlock(nn.Module):
    """Transformer encoder-decoder block (T5-style).

    Used in seq2seq models. Includes:
    1. Masked self-attention (causal, on decoder input)
    2. Cross-attention (decoder attends to encoder output)
    3. Feed-forward network
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        pre_norm: bool = True
    ):
        """Initialize encoder-decoder block.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            pre_norm: Use Pre-LN (True) or Post-LN (False)
        """
        super().__init__()
        self.pre_norm = pre_norm

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply encoder-decoder block.

        Args:
            x: Decoder input (batch, tgt_len, d_model)
            encoder_output: Encoder output (batch, src_len, d_model)
            self_mask: Causal mask for self-attention
            cross_mask: Mask for cross-attention (padding)

        Returns:
            Output tensor (batch, tgt_len, d_model)
        """
        if self.pre_norm:
            # Masked self-attention
            normed = self.norm1(x)
            self_attn_output, _ = self.self_attention(normed, normed, normed, self_mask)
            x = x + self.dropout(self_attn_output)

            # Cross-attention (Q from decoder, K/V from encoder)
            normed = self.norm2(x)
            cross_attn_output, _ = self.cross_attention(
                normed, encoder_output, encoder_output, cross_mask
            )
            x = x + self.dropout(cross_attn_output)

            # Feed-forward
            normed = self.norm3(x)
            ff_output = self.feed_forward(normed)
            x = x + ff_output

        else:
            # Post-LN
            # Masked self-attention
            self_attn_output, _ = self.self_attention(x, x, x, self_mask)
            x = self.norm1(x + self.dropout(self_attn_output))

            # Cross-attention
            cross_attn_output, _ = self.cross_attention(
                x, encoder_output, encoder_output, cross_mask
            )
            x = self.norm2(x + self.dropout(cross_attn_output))

            # Feed-forward
            ff_output = self.feed_forward(x)
            x = self.norm3(x + ff_output)

        return x
