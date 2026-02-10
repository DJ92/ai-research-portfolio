"""Attention mechanism variants (local, sparse, etc.).

These are advanced variants of attention for future work.
Placeholders for now to maintain clean module structure.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class LocalAttention(nn.Module):
    """Local attention with fixed window size.

    TODO: Implement sliding window attention.
    Each token attends only to neighbors within a fixed window.

    Reduces complexity from O(n²) to O(n * window_size).
    """

    def __init__(self, d_model: int, num_heads: int, window_size: int = 256):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        raise NotImplementedError("LocalAttention not yet implemented")


class SparseAttention(nn.Module):
    """Sparse attention with fixed patterns.

    TODO: Implement sparse attention (e.g., strided, fixed patterns).
    Uses sparse attention patterns to reduce computational cost.

    Examples: Longformer, BigBird attention patterns.
    """

    def __init__(self, d_model: int, num_heads: int, sparsity_pattern: str = "strided"):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.sparsity_pattern = sparsity_pattern
        raise NotImplementedError("SparseAttention not yet implemented")
