"""Tests for attention mechanisms."""

import pytest
import torch
import math

from src.attention import ScaledDotProductAttention, MultiHeadAttention


class TestScaledDotProductAttention:
    """Test scaled dot-product attention."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        batch_size, seq_len, d_k = 2, 10, 64
        attn = ScaledDotProductAttention(dropout=0.0)

        q = torch.randn(batch_size, seq_len, d_k)
        k = torch.randn(batch_size, seq_len, d_k)
        v = torch.randn(batch_size, seq_len, d_k)

        output, weights = attn(q, k, v)

        assert output.shape == (batch_size, seq_len, d_k)
        assert weights.shape == (batch_size, seq_len, seq_len)

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1 for each query."""
        batch_size, seq_len, d_k = 2, 10, 64
        attn = ScaledDotProductAttention(dropout=0.0)

        q = torch.randn(batch_size, seq_len, d_k)
        k = torch.randn(batch_size, seq_len, d_k)
        v = torch.randn(batch_size, seq_len, d_k)

        _, weights = attn(q, k, v)

        # Weights should sum to 1 along last dimension (over keys)
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6)

    def test_causal_masking(self):
        """Test that causal mask prevents attending to future positions."""
        batch_size, seq_len, d_k = 2, 5, 64
        attn = ScaledDotProductAttention(dropout=0.0)

        q = torch.randn(batch_size, seq_len, d_k)
        k = torch.randn(batch_size, seq_len, d_k)
        v = torch.randn(batch_size, seq_len, d_k)

        # Create causal mask (upper triangular)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

        _, weights = attn(q, k, v, mask)

        # Check that future positions have zero weight
        for i in range(seq_len):
            future_weights = weights[:, i, i+1:]
            if i < seq_len - 1:
                assert torch.allclose(
                    future_weights,
                    torch.zeros_like(future_weights),
                    atol=1e-6
                )

    def test_scaling_prevents_saturation(self):
        """Test that scaling by sqrt(d_k) prevents softmax saturation."""
        seq_len, d_k = 10, 512  # Large d_k
        attn = ScaledDotProductAttention(dropout=0.0)

        # Create queries and keys (not identical to avoid diagonal dominance)
        torch.manual_seed(42)
        q = torch.randn(1, seq_len, d_k)
        k = torch.randn(1, seq_len, d_k)
        v = torch.randn(1, seq_len, d_k)

        _, weights = attn(q, k, v)

        # With proper scaling, weights should not be extremely peaked
        # Average maximum weight across queries should be reasonable
        max_weights = weights.max(dim=-1).values
        avg_max_weight = max_weights.mean()

        # Should not be too uniform (> 0.15) or too peaked (> 0.8)
        assert 0.15 < avg_max_weight < 0.8, f"Unexpected attention distribution: {avg_max_weight:.3f}"


class TestMultiHeadAttention:
    """Test multi-head attention."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        batch_size, seq_len, d_model = 2, 10, 512
        num_heads = 8
        mha = MultiHeadAttention(d_model, num_heads, dropout=0.0)

        q = torch.randn(batch_size, seq_len, d_model)
        k = torch.randn(batch_size, seq_len, d_model)
        v = torch.randn(batch_size, seq_len, d_model)

        output, weights = mha(q, k, v)

        assert output.shape == (batch_size, seq_len, d_model)
        assert weights.shape == (batch_size, seq_len, seq_len)

    def test_dimension_divisibility(self):
        """Test that d_model must be divisible by num_heads."""
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=512, num_heads=7)  # 512 % 7 != 0

    def test_get_attention_maps(self):
        """Test getting per-head attention maps."""
        batch_size, seq_len, d_model = 2, 10, 512
        num_heads = 8
        mha = MultiHeadAttention(d_model, num_heads, dropout=0.0)

        q = torch.randn(batch_size, seq_len, d_model)
        k = torch.randn(batch_size, seq_len, d_model)
        v = torch.randn(batch_size, seq_len, d_model)

        head_weights = mha.get_attention_maps(q, k, v)

        assert head_weights.shape == (batch_size, num_heads, seq_len, seq_len)

        # Each head's weights should sum to 1
        weight_sums = head_weights.sum(dim=-1)
        assert torch.allclose(
            weight_sums,
            torch.ones_like(weight_sums),
            atol=1e-6
        )

    def test_different_heads_different_patterns(self):
        """Test that different heads can learn different attention patterns."""
        # This is more of a sanity check - they should be able to diverge
        batch_size, seq_len, d_model = 1, 10, 512
        num_heads = 8
        mha = MultiHeadAttention(d_model, num_heads, dropout=0.0)

        q = torch.randn(batch_size, seq_len, d_model)
        k = torch.randn(batch_size, seq_len, d_model)
        v = torch.randn(batch_size, seq_len, d_model)

        head_weights = mha.get_attention_maps(q, k, v)

        # Check that not all heads have identical patterns
        # (would be highly unlikely with random initialization)
        head_0 = head_weights[0, 0]
        head_1 = head_weights[0, 1]

        assert not torch.allclose(head_0, head_1, atol=0.01)

    def test_gradient_flow(self):
        """Test that gradients flow through attention."""
        batch_size, seq_len, d_model = 2, 10, 512
        num_heads = 8
        mha = MultiHeadAttention(d_model, num_heads, dropout=0.0)

        q = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        k = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        v = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

        output, _ = mha(q, k, v)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None

        # Check that gradients are non-zero
        assert q.grad.abs().sum() > 0
        assert k.grad.abs().sum() > 0
        assert v.grad.abs().sum() > 0
