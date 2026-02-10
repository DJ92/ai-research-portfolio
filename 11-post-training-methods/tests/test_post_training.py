"""Tests for post-training methods."""

import pytest
import torch
import torch.nn as nn

import sys
sys.path.insert(0, '/Users/djoshi/Desktop/Codebase/ai-research-portfolio/11-post-training-methods')

from src.sft import SupervisedFineTuner, InstructionDataset
from src.rlhf import RewardModel, RewardModelTrainer, PreferenceDataset
from src.dpo import DirectPreferenceOptimization, DPOPreferenceDataset


# Dummy models and tokenizer for testing
class DummyModel(nn.Module):
    def __init__(self, vocab_size=1000, d_model=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True)
        self.output = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        return self.output(x)


class DummyTokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size

    def __call__(self, text, max_length=512, padding='max_length',
                 truncation=True, return_tensors=None, add_special_tokens=True):
        # Simple tokenization: hash text to get token IDs
        if isinstance(text, list):
            text = text[0]

        tokens = [hash(text) % (self.vocab_size - 100) + 1 for _ in range(min(20, max_length))]
        tokens += [0] * (max_length - len(tokens))  # Pad

        result = {
            'input_ids': tokens,
            'attention_mask': [1] * min(20, max_length) + [0] * (max_length - min(20, max_length))
        }

        if return_tensors == 'pt':
            result = {k: torch.tensor([v]) for k, v in result.items()}

        return result


class TestSupervisedFineTuning:
    """Test SFT trainer."""

    def test_loss_computation(self):
        """Test that SFT can compute loss."""
        model = DummyModel()
        tokenizer = DummyTokenizer()
        sft = SupervisedFineTuner(model, tokenizer)

        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = input_ids.clone()
        labels[:, :10] = -100  # Mask first 10 tokens (prompt)

        loss, metrics = sft.compute_loss(input_ids, attention_mask, labels)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0

        assert 'loss' in metrics
        assert 'accuracy' in metrics

    def test_instruction_dataset(self):
        """Test instruction dataset creation."""
        tokenizer = DummyTokenizer()

        instructions = ["What is 2+2?", "Explain gravity"]
        responses = ["2+2 equals 4", "Gravity is a force"]

        dataset = InstructionDataset(
            instructions,
            responses,
            tokenizer,
            max_length=128
        )

        assert len(dataset) == 2

        example = dataset[0]
        assert 'input_ids' in example
        assert 'attention_mask' in example
        assert 'labels' in example

        # Labels should have some masked tokens (prompt)
        assert (example['labels'] == -100).sum() > 0

    def test_gradient_flow(self):
        """Test that gradients flow through SFT."""
        model = DummyModel()
        tokenizer = DummyTokenizer()
        sft = SupervisedFineTuner(model, tokenizer)

        input_ids = torch.randint(0, 1000, (2, 32))
        attention_mask = torch.ones(2, 32)
        labels = input_ids.clone()
        labels[:, :10] = -100

        loss, _ = sft.compute_loss(input_ids, attention_mask, labels)
        loss.backward()

        # Check gradients exist
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad


class TestRewardModel:
    """Test reward model."""

    def test_forward_pass(self):
        """Test reward model forward pass."""
        base_model = DummyModel()
        reward_model = RewardModel(base_model, d_model=128)

        batch_size, seq_len = 4, 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        rewards = reward_model(input_ids, attention_mask)

        assert rewards.shape == (batch_size,)
        assert rewards.dtype == torch.float32

    def test_pairwise_loss(self):
        """Test pairwise ranking loss."""
        base_model = DummyModel()
        reward_model = RewardModel(base_model, d_model=128)
        trainer = RewardModelTrainer(reward_model)

        batch_size, seq_len = 2, 32
        chosen_ids = torch.randint(0, 1000, (batch_size, seq_len))
        chosen_mask = torch.ones(batch_size, seq_len)
        rejected_ids = torch.randint(0, 1000, (batch_size, seq_len))
        rejected_mask = torch.ones(batch_size, seq_len)

        loss, metrics = trainer.compute_loss(
            chosen_ids, chosen_mask,
            rejected_ids, rejected_mask
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0

        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 'margin' in metrics

    def test_preference_dataset(self):
        """Test preference dataset."""
        tokenizer = DummyTokenizer()

        prompts = ["What is AI?"]
        chosen = ["AI is artificial intelligence"]
        rejected = ["AI is magic"]

        dataset = PreferenceDataset(
            prompts, chosen, rejected,
            tokenizer,
            max_length=128
        )

        assert len(dataset) == 1

        example = dataset[0]
        assert 'chosen_input_ids' in example
        assert 'rejected_input_ids' in example

    def test_chosen_gets_higher_reward_after_training(self):
        """Test that reward model learns to prefer chosen responses."""
        base_model = DummyModel()
        reward_model = RewardModel(base_model, d_model=128)
        trainer = RewardModelTrainer(reward_model, learning_rate=0.01)

        # Create simple synthetic data where chosen is always longer
        batch_size = 8
        chosen_ids = torch.randint(1, 1000, (batch_size, 32))
        chosen_mask = torch.ones(batch_size, 32)
        rejected_ids = torch.randint(1, 1000, (batch_size, 20))
        rejected_mask = torch.ones(batch_size, 20)

        # Pad rejected to same length
        rejected_ids = torch.cat([
            rejected_ids,
            torch.zeros(batch_size, 12, dtype=torch.long)
        ], dim=1)
        rejected_mask = torch.cat([
            rejected_mask,
            torch.zeros(batch_size, 12)
        ], dim=1)

        # Train for a few steps
        for _ in range(10):
            batch = {
                'chosen_input_ids': chosen_ids,
                'chosen_attention_mask': chosen_mask,
                'rejected_input_ids': rejected_ids,
                'rejected_attention_mask': rejected_mask
            }
            trainer.train_step(batch)

        # Check that margin improved
        with torch.no_grad():
            reward_chosen = reward_model(chosen_ids, chosen_mask)
            reward_rejected = reward_model(rejected_ids, rejected_mask)
            accuracy = (reward_chosen > reward_rejected).float().mean().item()

        # Should prefer chosen more often than random (>0.5)
        assert accuracy > 0.5


class TestDirectPreferenceOptimization:
    """Test DPO trainer."""

    def test_logprob_computation(self):
        """Test log probability computation."""
        model = DummyModel()
        ref_model = DummyModel()
        dpo = DirectPreferenceOptimization(model, ref_model)

        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = input_ids.clone()

        logprobs = dpo.get_logprobs(model, input_ids, attention_mask, labels)

        assert logprobs.shape == (batch_size,)
        assert logprobs.dtype == torch.float32

    def test_dpo_loss_computation(self):
        """Test DPO loss computation."""
        model = DummyModel()
        ref_model = DummyModel()
        dpo = DirectPreferenceOptimization(model, ref_model, beta=0.1)

        batch_size, seq_len = 2, 32
        chosen_ids = torch.randint(0, 1000, (batch_size, seq_len))
        chosen_mask = torch.ones(batch_size, seq_len)
        chosen_labels = chosen_ids.clone()

        rejected_ids = torch.randint(0, 1000, (batch_size, seq_len))
        rejected_mask = torch.ones(batch_size, seq_len)
        rejected_labels = rejected_ids.clone()

        loss, metrics = dpo.compute_loss(
            chosen_ids, chosen_mask, chosen_labels,
            rejected_ids, rejected_mask, rejected_labels
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0

        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 'reward_chosen' in metrics
        assert 'reward_rejected' in metrics
        assert 'kl_chosen' in metrics

    def test_dpo_preference_dataset(self):
        """Test DPO preference dataset with labels."""
        tokenizer = DummyTokenizer()

        prompts = ["What is ML?"]
        chosen = ["ML is machine learning"]
        rejected = ["ML is mindless logic"]

        dataset = DPOPreferenceDataset(
            prompts, chosen, rejected,
            tokenizer,
            max_length=128
        )

        assert len(dataset) == 1

        example = dataset[0]
        assert 'chosen_input_ids' in example
        assert 'chosen_labels' in example
        assert 'rejected_input_ids' in example
        assert 'rejected_labels' in example

        # Labels should have masked prompt tokens
        assert (example['chosen_labels'] == -100).sum() > 0
        assert (example['rejected_labels'] == -100).sum() > 0

    def test_gradient_flow(self):
        """Test that gradients flow through DPO."""
        model = DummyModel()
        ref_model = DummyModel()
        dpo = DirectPreferenceOptimization(model, ref_model)

        batch_size, seq_len = 2, 32
        chosen_ids = torch.randint(0, 1000, (batch_size, seq_len))
        chosen_mask = torch.ones(batch_size, seq_len)
        chosen_labels = chosen_ids.clone()

        rejected_ids = torch.randint(0, 1000, (batch_size, seq_len))
        rejected_mask = torch.ones(batch_size, seq_len)
        rejected_labels = rejected_ids.clone()

        loss, _ = dpo.compute_loss(
            chosen_ids, chosen_mask, chosen_labels,
            rejected_ids, rejected_mask, rejected_labels
        )
        loss.backward()

        # Check gradients exist for policy model
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad

        # Check no gradients for reference model
        for param in ref_model.parameters():
            assert param.grad is None or param.grad.abs().sum() == 0


class TestMethodComparison:
    """Compare different post-training methods."""

    def test_all_methods_compute_loss(self):
        """Test that all methods can compute loss."""
        model = DummyModel()
        ref_model = DummyModel()
        tokenizer = DummyTokenizer()

        # SFT
        sft = SupervisedFineTuner(model, tokenizer)
        input_ids = torch.randint(0, 1000, (2, 32))
        attention_mask = torch.ones(2, 32)
        labels = input_ids.clone()
        labels[:, :10] = -100
        sft_loss, _ = sft.compute_loss(input_ids, attention_mask, labels)

        # Reward Model
        reward_model = RewardModel(model, d_model=128)
        trainer = RewardModelTrainer(reward_model)
        chosen_ids = torch.randint(0, 1000, (2, 32))
        rejected_ids = torch.randint(0, 1000, (2, 32))
        masks = torch.ones(2, 32)
        rm_loss, _ = trainer.compute_loss(chosen_ids, masks, rejected_ids, masks)

        # DPO
        dpo = DirectPreferenceOptimization(model, ref_model)
        dpo_loss, _ = dpo.compute_loss(
            chosen_ids, masks, chosen_ids,
            rejected_ids, masks, rejected_ids
        )

        # All should compute loss
        assert sft_loss.item() > 0
        assert rm_loss.item() > 0
        assert dpo_loss.item() > 0
