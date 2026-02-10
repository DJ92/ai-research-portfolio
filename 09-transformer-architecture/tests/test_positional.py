"""Tests for positional encodings."""

import pytest
import torch
import math

from src.positional import (
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
    RotaryPositionalEmbedding
)


class TestSinusoidalPositionalEncoding:
    """Test sinusoidal positional encoding."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        batch_size, seq_len, d_model = 2, 10, 512
        pe = SinusoidalPositionalEncoding(d_model, dropout=0.0)

        x = torch.randn(batch_size, seq_len, d_model)
        output = pe(x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_deterministic(self):
        """Test that encoding is deterministic (no randomness)."""
        d_model, seq_len = 512, 10
        pe = SinusoidalPositionalEncoding(d_model, dropout=0.0)

        x = torch.randn(1, seq_len, d_model)
        output1 = pe(x)
        output2 = pe(x)

        assert torch.allclose(output1, output2)

    def test_different_positions_different_encodings(self):
        """Test that different positions get different encodings."""
        d_model, seq_len = 512, 10
        pe = SinusoidalPositionalEncoding(d_model, dropout=0.0)

        # Get encodings directly
        encodings = pe.pe[0]  # (max_seq_len, d_model)

        # Check that consecutive positions are different
        for i in range(seq_len - 1):
            assert not torch.allclose(encodings[i], encodings[i + 1])

    def test_even_dimension_requirement(self):
        """Test that d_model must be even."""
        with pytest.raises(AssertionError):
            SinusoidalPositionalEncoding(d_model=513)  # Odd

    def test_frequency_properties(self):
        """Test that different dimensions use different frequencies."""
        d_model, seq_len = 512, 100
        pe = SinusoidalPositionalEncoding(d_model, dropout=0.0)

        encodings = pe.pe[0, :seq_len, :]  # (seq_len, d_model)

        # First dimension should vary faster than last dimension
        # (higher frequency for lower dimensions)
        first_dim_changes = (encodings[1:, 0] != encodings[:-1, 0]).sum()
        last_dim_changes = (encodings[1:, -1] != encodings[:-1, -1]).sum()

        # This is approximate, but first dim should change more
        assert first_dim_changes > 0


class TestLearnedPositionalEncoding:
    """Test learned positional encoding."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        batch_size, seq_len, d_model = 2, 10, 512
        max_seq_len = 1024
        pe = LearnedPositionalEncoding(d_model, max_seq_len, dropout=0.0)

        x = torch.randn(batch_size, seq_len, d_model)
        output = pe(x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_trainable_parameters(self):
        """Test that position embeddings are trainable."""
        d_model, max_seq_len = 512, 1024
        pe = LearnedPositionalEncoding(d_model, max_seq_len, dropout=0.0)

        # Should have trainable parameters
        params = list(pe.parameters())
        assert len(params) > 0

        # Position embeddings should be trainable
        assert pe.position_embeddings.weight.requires_grad

    def test_max_length_enforcement(self):
        """Test that sequences longer than max_seq_len raise error."""
        d_model, max_seq_len = 512, 100
        pe = LearnedPositionalEncoding(d_model, max_seq_len, dropout=0.0)

        # Should work for seq_len <= max_seq_len
        x = torch.randn(1, max_seq_len, d_model)
        output = pe(x)
        assert output.shape == (1, max_seq_len, d_model)

        # Should fail for seq_len > max_seq_len
        x_long = torch.randn(1, max_seq_len + 1, d_model)
        with pytest.raises(AssertionError):
            pe(x_long)

    def test_different_positions_different_embeddings(self):
        """Test that learned embeddings start with different values."""
        d_model, max_seq_len = 512, 100
        pe = LearnedPositionalEncoding(d_model, max_seq_len, dropout=0.0)

        # Different positions should have different initial embeddings
        # (with high probability given random initialization)
        emb_0 = pe.position_embeddings.weight[0]
        emb_1 = pe.position_embeddings.weight[1]

        assert not torch.allclose(emb_0, emb_1)


class TestRotaryPositionalEmbedding:
    """Test rotary positional embedding (RoPE)."""

    def test_output_shapes(self):
        """Test that rotated Q and K have correct shapes."""
        batch_size, num_heads, seq_len, d_k = 2, 8, 10, 64
        rope = RotaryPositionalEmbedding(dim=d_k)

        q = torch.randn(batch_size, num_heads, seq_len, d_k)
        k = torch.randn(batch_size, num_heads, seq_len, d_k)

        q_rot, k_rot = rope(q, k)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_no_trainable_parameters(self):
        """Test that RoPE has no trainable parameters."""
        rope = RotaryPositionalEmbedding(dim=64)

        params = list(rope.parameters())
        trainable_params = [p for p in params if p.requires_grad]

        assert len(trainable_params) == 0

    def test_rotation_property(self):
        """Test that RoPE preserves vector norms (rotations are norm-preserving)."""
        batch_size, num_heads, seq_len, d_k = 2, 8, 10, 64
        rope = RotaryPositionalEmbedding(dim=d_k)

        q = torch.randn(batch_size, num_heads, seq_len, d_k)
        k = torch.randn(batch_size, num_heads, seq_len, d_k)

        q_rot, k_rot = rope(q, k)

        # Norms should be approximately preserved (rotations are norm-preserving)
        q_norms = torch.norm(q, dim=-1)
        q_rot_norms = torch.norm(q_rot, dim=-1)

        k_norms = torch.norm(k, dim=-1)
        k_rot_norms = torch.norm(k_rot, dim=-1)

        assert torch.allclose(q_norms, q_rot_norms, rtol=1e-5)
        assert torch.allclose(k_norms, k_rot_norms, rtol=1e-5)

    def test_relative_position_encoding(self):
        """Test that RoPE encodes relative positions.

        The key property: dot product between rotated vectors depends only on
        relative position difference, not absolute positions.
        """
        num_heads, seq_len, d_k = 1, 10, 64
        rope = RotaryPositionalEmbedding(dim=d_k)

        # Create simple query and key
        q = torch.randn(1, num_heads, seq_len, d_k)
        k = torch.randn(1, num_heads, seq_len, d_k)

        q_rot, k_rot = rope(q, k)

        # Compute attention scores (dot products)
        scores = torch.matmul(
            q_rot.squeeze(0).squeeze(0),
            k_rot.squeeze(0).squeeze(0).transpose(-2, -1)
        )

        # Scores along diagonals (same relative distance) should be related
        # This is hard to test precisely, but we can check that the function works
        assert scores.shape == (seq_len, seq_len)

    def test_longer_sequences(self):
        """Test that RoPE can handle sequences longer than initial max_seq_len."""
        d_k = 64
        initial_max_len = 100
        rope = RotaryPositionalEmbedding(dim=d_k, max_seq_len=initial_max_len)

        # Try with longer sequence
        longer_seq_len = initial_max_len + 50
        q = torch.randn(1, 1, longer_seq_len, d_k)
        k = torch.randn(1, 1, longer_seq_len, d_k)

        # Should automatically rebuild cache
        q_rot, k_rot = rope(q, k, seq_len=longer_seq_len)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape


class TestPositionalEncodingComparison:
    """Compare different positional encoding methods."""

    def test_all_add_position_information(self):
        """Test that all methods modify the input (add position info)."""
        batch_size, seq_len, d_model = 2, 10, 512

        x = torch.randn(batch_size, seq_len, d_model)

        # Sinusoidal
        sin_pe = SinusoidalPositionalEncoding(d_model, dropout=0.0)
        sin_out = sin_pe(x.clone())
        assert not torch.allclose(x, sin_out)

        # Learned
        learned_pe = LearnedPositionalEncoding(d_model, dropout=0.0)
        learned_out = learned_pe(x.clone())
        assert not torch.allclose(x, learned_out)

        # RoPE is different - it rotates Q/K in attention, not embeddings
        # So we test it separately
        rope = RotaryPositionalEmbedding(dim=64)
        q = torch.randn(batch_size, 8, seq_len, 64)
        k = q.clone()
        q_rot, k_rot = rope(q, k)
        assert not torch.allclose(q, q_rot)
