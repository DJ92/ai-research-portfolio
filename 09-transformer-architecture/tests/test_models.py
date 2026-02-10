"""Tests for transformer models."""

import pytest
import torch

from src.models import (
    GPTModel,
    create_gpt_small,
    create_gpt_medium,
    TransformerDecoderBlock,
    FeedForward
)


class TestFeedForward:
    """Test feed-forward network."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        batch_size, seq_len, d_model = 2, 10, 512
        d_ff = 2048
        ffn = FeedForward(d_model, d_ff, dropout=0.0)

        x = torch.randn(batch_size, seq_len, d_model)
        output = ffn(x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_expansion_and_contraction(self):
        """Test that FFN expands then contracts dimension."""
        d_model, d_ff = 512, 2048
        ffn = FeedForward(d_model, d_ff, dropout=0.0)

        # Check layer dimensions
        assert ffn.linear1.out_features == d_ff
        assert ffn.linear2.out_features == d_model

    def test_gradient_flow(self):
        """Test that gradients flow through FFN."""
        batch_size, seq_len, d_model = 2, 10, 512
        ffn = FeedForward(d_model, 2048, dropout=0.0)

        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        output = ffn(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestTransformerDecoderBlock:
    """Test transformer decoder block."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        batch_size, seq_len, d_model = 2, 10, 512
        block = TransformerDecoderBlock(
            d_model=d_model,
            num_heads=8,
            d_ff=2048,
            dropout=0.0
        )

        x = torch.randn(batch_size, seq_len, d_model)
        output = block(x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_causal_masking(self):
        """Test that decoder block respects causal mask."""
        batch_size, seq_len, d_model = 1, 5, 512
        block = TransformerDecoderBlock(
            d_model=d_model,
            num_heads=8,
            d_ff=2048,
            dropout=0.0
        )

        x = torch.randn(batch_size, seq_len, d_model)

        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

        # Run with and without mask
        output_no_mask = block(x, mask=None)
        output_with_mask = block(x, mask=mask)

        # Outputs should be different
        assert not torch.allclose(output_no_mask, output_with_mask)

    def test_residual_connections(self):
        """Test that residual connections are present."""
        batch_size, seq_len, d_model = 2, 10, 512
        block = TransformerDecoderBlock(
            d_model=d_model,
            num_heads=8,
            d_ff=2048,
            dropout=0.0,
            pre_norm=True
        )

        x = torch.randn(batch_size, seq_len, d_model)
        output = block(x)

        # Output should be different from input (transformation applied)
        # but should contain some information from input (residual connection)
        assert not torch.allclose(x, output)

        # Gradient should flow to input (residual connection)
        x_grad = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        output_grad = block(x_grad)
        loss = output_grad.sum()
        loss.backward()

        assert x_grad.grad is not None


class TestGPTModel:
    """Test GPT model."""

    def test_forward_shape(self):
        """Test that forward pass produces correct shape."""
        vocab_size = 1000
        batch_size, seq_len = 2, 10

        model = GPTModel(
            vocab_size=vocab_size,
            d_model=256,
            num_layers=4,
            num_heads=4,
            d_ff=1024,
            max_seq_len=512,
            dropout=0.0
        )

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        logits = model(input_ids)

        assert logits.shape == (batch_size, seq_len, vocab_size)

    def test_causal_generation(self):
        """Test that model can generate autoregressively."""
        vocab_size = 1000
        model = GPTModel(
            vocab_size=vocab_size,
            d_model=256,
            num_layers=4,
            num_heads=4,
            d_ff=1024,
            max_seq_len=512,
            dropout=0.0
        )

        # Generate from short prompt
        prompt = torch.randint(0, vocab_size, (1, 5))
        generated = model.generate(
            prompt,
            max_new_tokens=10,
            do_sample=False  # Greedy for determinism
        )

        assert generated.shape == (1, 15)  # 5 + 10
        # Check that prompt is unchanged
        assert torch.equal(generated[:, :5], prompt)

    def test_temperature_sampling(self):
        """Test that temperature affects generation diversity."""
        vocab_size = 1000
        model = GPTModel(
            vocab_size=vocab_size,
            d_model=256,
            num_layers=4,
            num_heads=4,
            d_ff=1024,
            max_seq_len=512,
            dropout=0.0
        )

        prompt = torch.randint(0, vocab_size, (1, 5))

        # Low temperature should be more deterministic
        torch.manual_seed(42)
        gen_low_1 = model.generate(prompt.clone(), max_new_tokens=10, temperature=0.1)
        torch.manual_seed(42)
        gen_low_2 = model.generate(prompt.clone(), max_new_tokens=10, temperature=0.1)

        # Should be identical with same seed
        assert torch.equal(gen_low_1, gen_low_2)

    def test_overfit_small_dataset(self):
        """Test that model can overfit a tiny dataset (sanity check)."""
        vocab_size = 100
        model = GPTModel(
            vocab_size=vocab_size,
            d_model=128,
            num_layers=2,
            num_heads=4,
            d_ff=512,
            max_seq_len=32,
            dropout=0.0
        )

        # Create tiny dataset (single sequence repeated)
        seq_len = 10
        data = torch.randint(0, vocab_size, (1, seq_len))

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        # Train for a few steps
        initial_loss = None
        final_loss = None

        for step in range(50):
            optimizer.zero_grad()

            # Forward pass
            logits = model(data)

            # Compute loss (predict next token)
            # Shift so that tokens < n predict n
            loss = criterion(
                logits[:, :-1, :].reshape(-1, vocab_size),
                data[:, 1:].reshape(-1)
            )

            if step == 0:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

            if step == 49:
                final_loss = loss.item()

        # Loss should decrease significantly
        assert final_loss < initial_loss * 0.5, \
            f"Model didn't learn: initial={initial_loss:.4f}, final={final_loss:.4f}"

    def test_attention_mask(self):
        """Test that padding mask works correctly."""
        vocab_size = 1000
        batch_size = 2
        seq_len = 10

        model = GPTModel(
            vocab_size=vocab_size,
            d_model=256,
            num_layers=2,
            num_heads=4,
            d_ff=1024,
            max_seq_len=512,
            dropout=0.0
        )

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Create padding mask (1 = valid, 0 = padding)
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, 7:] = 0  # Mask last 3 tokens of first example

        logits_with_mask = model(input_ids, attention_mask=attention_mask)
        logits_no_mask = model(input_ids)

        # Should produce different results
        assert not torch.allclose(logits_with_mask, logits_no_mask)

    def test_model_sizes(self):
        """Test that different model sizes have expected parameter counts."""
        vocab_size = 50000

        small = create_gpt_small(vocab_size)
        medium = create_gpt_medium(vocab_size)

        small_params = small.get_num_params(non_embedding=True)
        medium_params = medium.get_num_params(non_embedding=True)

        # Medium should have more parameters than small
        assert medium_params > small_params

        # Rough ballpark checks (adjusted for our smaller models)
        assert 10e6 < small_params < 30e6  # ~18M
        assert 50e6 < medium_params < 100e6  # ~85M

    def test_weight_tying(self):
        """Test that input and output embeddings can be tied."""
        vocab_size = 1000

        # With weight tying
        model_tied = GPTModel(
            vocab_size=vocab_size,
            d_model=256,
            num_layers=2,
            num_heads=4,
            d_ff=1024,
            tie_weights=True
        )

        # Embeddings should be tied (same object)
        assert model_tied.token_embedding.weight is model_tied.lm_head.weight

        # Without weight tying
        model_untied = GPTModel(
            vocab_size=vocab_size,
            d_model=256,
            num_layers=2,
            num_heads=4,
            d_ff=1024,
            tie_weights=False
        )

        # Embeddings should be different
        assert model_untied.token_embedding.weight is not model_untied.lm_head.weight

    def test_hidden_states_output(self):
        """Test that model can return intermediate hidden states."""
        vocab_size = 1000
        num_layers = 4

        model = GPTModel(
            vocab_size=vocab_size,
            d_model=256,
            num_layers=num_layers,
            num_heads=4,
            d_ff=1024,
            dropout=0.0
        )

        input_ids = torch.randint(0, vocab_size, (1, 10))
        logits, hidden_states = model(input_ids, return_hidden_states=True)

        # Should have one hidden state per layer
        assert len(hidden_states) == num_layers

        # Each should have correct shape
        for hidden in hidden_states:
            assert hidden.shape == (1, 10, 256)
