"""Positional encoding implementations for transformers.

Implements three variants:
1. Sinusoidal (Vaswani et al., 2017) - Original transformer
2. Learned (GPT-style) - Trainable positional embeddings
3. Rotary (RoPE) - Modern relative position encoding
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from 'Attention is All You Need'.

    Uses sine and cosine functions of different frequencies to encode position:

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Advantages:
    - No learned parameters (zero-shot generalization)
    - Can extrapolate to longer sequences than seen during training
    - Different frequencies encode position at different scales

    Why it works:
    - Each dimension uses a different frequency (geometric progression)
    - Allows model to learn relative positions (PE(pos+k) is linear function of PE(pos))
    - Smooth variation prevents sharp discontinuities
    """

    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        """Initialize sinusoidal positional encoding.

        Args:
            d_model: Model dimension (must be even)
            max_seq_len: Maximum sequence length to pre-compute
            dropout: Dropout probability
        """
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for sinusoidal encoding"

        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

        # Pre-compute positional encodings for efficiency
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # Compute the division term: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (max_seq_len, d_model) -> (1, max_seq_len, d_model)
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: Input embeddings (batch, seq_len, d_model)

        Returns:
            Embeddings with positional encoding added (batch, seq_len, d_model)
        """
        seq_len = x.size(1)

        # Add positional encoding (broadcast across batch)
        x = x + self.pe[:, :seq_len, :]

        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embeddings (GPT-style).

    Instead of fixed sinusoidal patterns, learns position embeddings from data.

    Advantages:
    - Can learn task-specific positional patterns
    - Often works better in practice for fixed-length tasks
    - Simple to implement and understand

    Disadvantages:
    - Fixed maximum sequence length (can't extrapolate)
    - Requires training data to learn patterns
    - More parameters to train
    """

    def __init__(self, d_model: int, max_seq_len: int = 1024, dropout: float = 0.1):
        """Initialize learned positional encoding.

        Args:
            d_model: Model dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Learnable position embeddings
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add learned positional embeddings to input.

        Args:
            x: Input embeddings (batch, seq_len, d_model)

        Returns:
            Embeddings with positional encoding added (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.size()

        assert seq_len <= self.max_seq_len, \
            f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"

        # Create position indices: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        # Add position embeddings
        x = x + self.position_embeddings(positions)

        return self.dropout(x)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) from RoFormer (Su et al., 2021).

    Used in modern models like LLaMA, GPT-NeoX, and PaLM.

    Key idea: Encode relative position by rotating the query and key vectors.
    Instead of adding position to embeddings, rotates them in 2D subspaces.

    Advantages:
    - Encodes relative positions naturally (attention sees relative distances)
    - Better extrapolation to longer sequences
    - No additional parameters
    - Efficient implementation

    Math:
    For position m, rotate [q_{2i}, q_{2i+1}] by angle m*θ_i where θ_i = 10000^(-2i/d)
    This creates rotation matrix that depends on relative position difference.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        """Initialize RoPE.

        Args:
            dim: Dimension of each head (d_k)
            max_seq_len: Maximum sequence length
            base: Base for frequency computation (default 10000)
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Pre-compute frequency inverse: θ_i = base^(-2i/dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Pre-compute cos and sin for all positions
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Build cache of cos and sin values for all positions."""
        # Create position indices: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, dtype=torch.float)

        # Compute angles: position * inv_freq
        # Shape: (seq_len, dim/2)
        freqs = torch.outer(positions, self.inv_freq)

        # Concatenate to match full dimension
        # Shape: (seq_len, dim)
        emb = torch.cat([freqs, freqs], dim=-1)

        # Pre-compute cos and sin
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input.

        For vector [x1, x2, x3, x4], returns [-x3, -x4, x1, x2]
        This implements the rotation in 2D subspaces.
        """
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embedding to queries and keys.

        Args:
            q: Query tensor (batch, num_heads, seq_len, d_k)
            k: Key tensor (batch, num_heads, seq_len, d_k)
            seq_len: Sequence length (if None, uses q.size(2))

        Returns:
            Rotated query and key tensors
        """
        if seq_len is None:
            seq_len = q.size(2)

        # Rebuild cache if sequence is longer than cached
        if seq_len > self.max_seq_len:
            self.max_seq_len = seq_len
            self._build_cache(seq_len)

        # Get cos and sin for current sequence length
        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)

        # Apply rotation: x * cos + rotate_half(x) * sin
        q_embed = q * cos + self._rotate_half(q) * sin
        k_embed = k * cos + self._rotate_half(k) * sin

        return q_embed, k_embed


def compare_positional_encodings(d_model: int = 512, seq_len: int = 100):
    """Compare different positional encoding methods (for debugging/visualization).

    Args:
        d_model: Model dimension
        seq_len: Sequence length to compare
    """
    # Create dummy input
    x = torch.randn(1, seq_len, d_model)

    # Sinusoidal
    sin_pe = SinusoidalPositionalEncoding(d_model)
    sin_output = sin_pe(x)

    # Learned (requires initialization)
    learned_pe = LearnedPositionalEncoding(d_model, max_seq_len=seq_len)
    learned_output = learned_pe(x)

    # RoPE (applied to Q, K in attention, not embeddings)
    rope = RotaryPositionalEmbedding(dim=d_model // 8)  # dim per head

    print("Positional Encoding Comparison:")
    print(f"  Sinusoidal - Shape: {sin_output.shape}, Params: 0")
    print(f"  Learned    - Shape: {learned_output.shape}, Params: {seq_len * d_model:,}")
    print(f"  RoPE       - Applied in attention, Params: 0")

    return {
        'sinusoidal': sin_output,
        'learned': learned_output,
        'rope': rope
    }


if __name__ == "__main__":
    # Quick test
    compare_positional_encodings()
