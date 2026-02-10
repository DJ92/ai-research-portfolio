"""Tests for pre-training objectives."""

import pytest
import torch
import torch.nn as nn

import sys
sys.path.insert(0, '/Users/djoshi/Desktop/Codebase/ai-research-portfolio/09-transformer-architecture')

from src.objectives import CausalLanguageModeling, MaskedLanguageModeling, verify_masking_strategy


# Dummy model for testing
class DummyModel(nn.Module):
    def __init__(self, vocab_size, d_model=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True)
        self.output = nn.Linear(d_model, vocab_size)
        self.max_seq_len = 512

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        return self.output(x)

    def generate(self, input_ids, max_new_tokens=10, **kwargs):
        # Simple greedy generation for testing
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


class TestCausalLanguageModeling:
    """Test CLM objective."""

    def test_loss_computation(self):
        """Test that CLM loss can be computed."""
        vocab_size = 1000
        model = DummyModel(vocab_size)
        clm = CausalLanguageModeling(model)

        batch_size, seq_len = 4, 32
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        loss, metrics = clm.compute_loss(input_ids)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() > 0

        assert 'loss' in metrics
        assert 'perplexity' in metrics
        assert 'accuracy' in metrics

    def test_perplexity_calculation(self):
        """Test that perplexity is exp(loss)."""
        vocab_size = 1000
        model = DummyModel(vocab_size)
        clm = CausalLanguageModeling(model)

        input_ids = torch.randint(0, vocab_size, (2, 16))

        loss, metrics = clm.compute_loss(input_ids)

        # Perplexity should be exp(loss)
        expected_ppl = torch.exp(loss).item()
        assert abs(metrics['perplexity'] - expected_ppl) < 1e-5

    def test_padding_mask(self):
        """Test that padding is handled correctly."""
        vocab_size = 1000
        pad_token_id = 0
        model = DummyModel(vocab_size)
        clm = CausalLanguageModeling(model, ignore_index=pad_token_id)

        batch_size, seq_len = 2, 20
        input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))

        # Add padding to second sequence
        input_ids[1, 15:] = pad_token_id
        attention_mask = (input_ids != pad_token_id).long()

        loss_with_mask, _ = clm.compute_loss(input_ids, attention_mask)

        # Should not crash and should compute loss
        assert loss_with_mask.item() > 0

    def test_generation(self):
        """Test that generation works."""
        vocab_size = 100
        model = DummyModel(vocab_size)
        clm = CausalLanguageModeling(model)

        prompt = torch.randint(0, vocab_size, (1, 5))
        generated = clm.generate(prompt, max_new_tokens=10)

        assert generated.shape == (1, 15)  # 5 + 10
        # Prompt should be unchanged
        assert torch.equal(generated[:, :5], prompt)


class TestMaskedLanguageModeling:
    """Test MLM objective."""

    def test_masking_creates_labels(self):
        """Test that masking creates correct labels."""
        vocab_size = 1000
        mask_token_id = 999
        model = DummyModel(vocab_size)
        mlm = MaskedLanguageModeling(model, mask_token_id, vocab_size, mask_prob=0.15)

        input_ids = torch.randint(0, vocab_size - 1, (2, 20))

        masked_input, labels = mlm.create_masked_input(input_ids)

        # Shapes should match
        assert masked_input.shape == input_ids.shape
        assert labels.shape == input_ids.shape

        # Labels should be mostly ignore_index
        assert (labels == mlm.ignore_index).sum() > 0

        # Some tokens should be masked
        num_masked = (labels != mlm.ignore_index).sum()
        assert num_masked > 0

    def test_masking_strategy_proportions(self):
        """Test that masking follows 80/10/10 split."""
        vocab_size = 1000
        mask_token_id = 999
        model = DummyModel(vocab_size)
        mlm = MaskedLanguageModeling(
            model,
            mask_token_id,
            vocab_size,
            mask_prob=0.15,
            mask_token_prob=0.8,
            random_token_prob=0.1
        )

        # Verify with many samples
        stats = verify_masking_strategy(mlm, num_samples=1000)

        if stats and stats['total_samples'] > 100:
            # Should be approximately 80/10/10
            assert 75 < stats['mask_pct'] < 85
            assert 5 < stats['random_pct'] < 15
            assert 5 < stats['unchanged_pct'] < 15

    def test_mlm_loss_computation(self):
        """Test that MLM loss can be computed."""
        vocab_size = 1000
        mask_token_id = 999
        model = DummyModel(vocab_size)
        mlm = MaskedLanguageModeling(model, mask_token_id, vocab_size)

        batch_size, seq_len = 4, 32
        input_ids = torch.randint(0, vocab_size - 1, (batch_size, seq_len))

        loss, metrics = mlm.compute_loss(input_ids)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0

        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 'num_masked' in metrics

    def test_special_tokens_not_masked(self):
        """Test that special tokens are never masked."""
        vocab_size = 1000
        mask_token_id = 999
        cls_token_id = 998
        sep_token_id = 997

        model = DummyModel(vocab_size)
        mlm = MaskedLanguageModeling(
            model,
            mask_token_id,
            vocab_size,
            special_tokens=[cls_token_id, sep_token_id, mask_token_id]
        )

        # Create input with special tokens
        batch_size, seq_len = 2, 20
        input_ids = torch.randint(0, vocab_size - 3, (batch_size, seq_len))
        input_ids[:, 0] = cls_token_id  # CLS at start
        input_ids[:, -1] = sep_token_id  # SEP at end

        masked_input, labels = mlm.create_masked_input(input_ids)

        # Special tokens should never be in labels (always ignore_index)
        assert labels[:, 0].eq(mlm.ignore_index).all()
        assert labels[:, -1].eq(mlm.ignore_index).all()

    def test_mask_percentage(self):
        """Test that approximately 15% of tokens are masked."""
        vocab_size = 1000
        mask_token_id = 999
        model = DummyModel(vocab_size)
        mlm = MaskedLanguageModeling(model, mask_token_id, vocab_size, mask_prob=0.15)

        # Test on multiple batches
        total_tokens = 0
        total_masked = 0

        for _ in range(10):
            input_ids = torch.randint(0, vocab_size - 1, (4, 100))
            masked_input, labels = mlm.create_masked_input(input_ids)

            num_masked = (labels != mlm.ignore_index).sum().item()
            total_tokens += labels.numel()
            total_masked += num_masked

        mask_percentage = total_masked / total_tokens * 100

        # Should be approximately 15%
        assert 10 < mask_percentage < 20


class TestObjectiveComparison:
    """Compare CLM and MLM objectives."""

    def test_both_compute_loss(self):
        """Test that both objectives can compute loss."""
        vocab_size = 1000
        mask_token_id = 999
        model = DummyModel(vocab_size)

        clm = CausalLanguageModeling(model)
        mlm = MaskedLanguageModeling(model, mask_token_id, vocab_size)

        input_ids = torch.randint(0, vocab_size - 1, (2, 32))

        # Both should compute loss without error
        clm_loss, clm_metrics = clm.compute_loss(input_ids)
        mlm_loss, mlm_metrics = mlm.compute_loss(input_ids)

        assert clm_loss.item() > 0
        assert mlm_loss.item() > 0

    def test_gradient_flow(self):
        """Test that gradients flow through both objectives."""
        vocab_size = 1000
        mask_token_id = 999

        # CLM
        model_clm = DummyModel(vocab_size)
        clm = CausalLanguageModeling(model_clm)

        input_ids = torch.randint(0, vocab_size, (2, 16))
        loss, _ = clm.compute_loss(input_ids)
        loss.backward()

        # Check that gradients exist
        has_grad = False
        for param in model_clm.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad

        # MLM
        model_mlm = DummyModel(vocab_size)
        mlm = MaskedLanguageModeling(model_mlm, mask_token_id, vocab_size)

        input_ids = torch.randint(0, vocab_size - 1, (2, 16))
        loss, _ = mlm.compute_loss(input_ids)
        loss.backward()

        # Check that gradients exist
        has_grad = False
        for param in model_mlm.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad
