"""Reward model for RLHF.

Learns to predict human preferences from pairwise comparisons.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class RewardModel(nn.Module):
    """Reward model that scores model outputs.

    Takes a sequence and outputs a scalar reward.
    Trained on pairwise preferences using Bradley-Terry model.

    Architecture:
    - Base language model (frozen or fine-tuned)
    - Reward head (linear layer) on top of last token

    Training objective:
    - Maximize P(y_chosen > y_rejected) = σ(r_chosen - r_rejected)
    - Loss = -log σ(r_chosen - r_rejected)
    """

    def __init__(
        self,
        base_model: nn.Module,
        d_model: int,
        freeze_base: bool = False
    ):
        """Initialize reward model.

        Args:
            base_model: Pre-trained language model
            d_model: Model dimension
            freeze_base: Whether to freeze base model weights
        """
        super().__init__()
        self.base_model = base_model
        self.d_model = d_model

        # Freeze base model if specified
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Reward head: outputs scalar reward
        self.reward_head = nn.Linear(d_model, 1)

        # Initialize reward head
        nn.init.zeros_(self.reward_head.weight)
        nn.init.zeros_(self.reward_head.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute reward for sequences.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            attention_mask: Attention mask

        Returns:
            Rewards (batch,) - scalar reward per sequence
        """
        # Get hidden states from base model
        # For a proper model, this would return hidden states
        # For our dummy model, it returns logits, so we'll use embedding instead
        hidden_states = self.base_model.embedding(input_ids)  # (batch, seq, d_model)

        # Get last token representation
        if attention_mask is not None:
            # Find last non-padding token for each sequence
            last_token_indices = attention_mask.sum(dim=1).long() - 1
            batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
            last_hidden = hidden_states[batch_indices, last_token_indices, :]
        else:
            # Use last token
            last_hidden = hidden_states[:, -1, :]

        # Compute reward
        rewards = self.reward_head(last_hidden).squeeze(-1)  # (batch,)

        return rewards


class RewardModelTrainer:
    """Trainer for reward model using preference data.

    Trains reward model to predict which response humans prefer.
    Uses pairwise ranking loss (Bradley-Terry model).
    """

    def __init__(
        self,
        reward_model: RewardModel,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0
    ):
        """Initialize trainer.

        Args:
            reward_model: Reward model to train
            learning_rate: Learning rate
            weight_decay: Weight decay
            max_grad_norm: Gradient clipping threshold
        """
        self.reward_model = reward_model
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.AdamW(
            reward_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.device = next(reward_model.parameters()).device

    def compute_loss(
        self,
        chosen_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute pairwise ranking loss.

        Loss = -log σ(r_chosen - r_rejected)

        Args:
            chosen_ids: Chosen response IDs (batch, seq_len)
            chosen_mask: Chosen attention mask
            rejected_ids: Rejected response IDs (batch, seq_len)
            rejected_mask: Rejected attention mask

        Returns:
            loss: Scalar loss
            metrics: Dictionary of metrics
        """
        # Compute rewards for both responses
        reward_chosen = self.reward_model(chosen_ids, chosen_mask)
        reward_rejected = self.reward_model(rejected_ids, rejected_mask)

        # Bradley-Terry loss
        # P(chosen > rejected) = σ(r_chosen - r_rejected)
        # Loss = -log P(chosen > rejected)
        logits = reward_chosen - reward_rejected
        loss = -F.logsigmoid(logits).mean()

        # Metrics
        with torch.no_grad():
            # Accuracy: how often does chosen get higher reward?
            accuracy = (reward_chosen > reward_rejected).float().mean().item()

            # Margin: average difference in rewards
            margin = (reward_chosen - reward_rejected).mean().item()

        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy,
            'margin': margin,
            'reward_chosen_mean': reward_chosen.mean().item(),
            'reward_rejected_mean': reward_rejected.mean().item()
        }

        return loss, metrics

    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Single training step.

        Args:
            batch: Batch with chosen and rejected responses

        Returns:
            Metrics dictionary
        """
        # Move to device
        chosen_ids = batch['chosen_input_ids'].to(self.device)
        chosen_mask = batch['chosen_attention_mask'].to(self.device)
        rejected_ids = batch['rejected_input_ids'].to(self.device)
        rejected_mask = batch['rejected_attention_mask'].to(self.device)

        # Forward pass
        loss, metrics = self.compute_loss(
            chosen_ids, chosen_mask,
            rejected_ids, rejected_mask
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.reward_model.parameters(),
            self.max_grad_norm
        )

        # Update weights
        self.optimizer.step()

        return metrics


class PreferenceDataset(torch.utils.data.Dataset):
    """Dataset for preference pairs (chosen vs rejected)."""

    def __init__(
        self,
        prompts: list[str],
        chosen_responses: list[str],
        rejected_responses: list[str],
        tokenizer,
        max_length: int = 512
    ):
        """Initialize dataset.

        Args:
            prompts: List of prompts
            chosen_responses: List of preferred responses
            rejected_responses: List of rejected responses
            tokenizer: Tokenizer
            max_length: Maximum sequence length
        """
        assert len(prompts) == len(chosen_responses) == len(rejected_responses)
        self.prompts = prompts
        self.chosen_responses = chosen_responses
        self.rejected_responses = rejected_responses
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single preference pair.

        Args:
            idx: Index

        Returns:
            Dictionary with chosen and rejected inputs
        """
        prompt = self.prompts[idx]
        chosen = self.chosen_responses[idx]
        rejected = self.rejected_responses[idx]

        # Tokenize chosen
        chosen_text = f"{prompt}\n\n{chosen}"
        chosen_encoding = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize rejected
        rejected_text = f"{prompt}\n\n{rejected}"
        rejected_encoding = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'chosen_input_ids': chosen_encoding['input_ids'].squeeze(0),
            'chosen_attention_mask': chosen_encoding['attention_mask'].squeeze(0),
            'rejected_input_ids': rejected_encoding['input_ids'].squeeze(0),
            'rejected_attention_mask': rejected_encoding['attention_mask'].squeeze(0)
        }
