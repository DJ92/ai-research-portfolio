"""Direct Preference Optimization (DPO).

Simpler alternative to RLHF that optimizes policy directly from preferences
without needing a reward model or RL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class DirectPreferenceOptimization:
    """Direct Preference Optimization trainer.

    Key insight: RLHF's optimal policy has a closed form that depends on
    the reward and reference policy. We can rearrange this to express the
    reward in terms of the optimal and reference policies, then substitute
    back into the preference loss.

    Result: Directly optimize policy on preferences, no reward model needed!

    Objective:
    maximize E[log σ(β * log π_θ(y_w|x) / π_ref(y_w|x)
                        - β * log π_θ(y_l|x) / π_ref(y_l|x))]

    Where:
    - y_w = preferred (winning) response
    - y_l = rejected (losing) response
    - β = temperature parameter (controls KL penalty strength)
    - π_θ = policy being trained
    - π_ref = reference policy (frozen SFT model)
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        beta: float = 0.1,
        learning_rate: float = 5e-7,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0
    ):
        """Initialize DPO trainer.

        Args:
            model: Policy model to train
            ref_model: Reference model (frozen, typically SFT model)
            beta: Temperature parameter (higher = stronger KL penalty)
            learning_rate: Learning rate (very low for stability)
            weight_decay: Weight decay
            max_grad_norm: Gradient clipping threshold
        """
        self.model = model
        self.ref_model = ref_model
        self.beta = beta
        self.max_grad_norm = max_grad_norm

        # Freeze reference model
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model.eval()

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.device = next(model.parameters()).device

    def get_logprobs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probabilities for sequences.

        Args:
            model: Language model
            input_ids: Input IDs (batch, seq_len)
            attention_mask: Attention mask
            labels: Target labels (batch, seq_len)

        Returns:
            Log probabilities (batch,) - sum of log probs for each sequence
        """
        # Forward pass
        logits = model(input_ids, attention_mask=attention_mask)

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather log probs for actual tokens
        batch_size, seq_len, vocab_size = shift_logits.shape
        gathered_log_probs = log_probs.gather(
            dim=2,
            index=shift_labels.unsqueeze(2)
        ).squeeze(2)  # (batch, seq_len-1)

        # Sum log probs (only over valid tokens, not padding)
        valid_mask = (shift_labels != -100)  # Assuming -100 is ignore_index
        sequence_log_probs = (gathered_log_probs * valid_mask).sum(dim=1)

        return sequence_log_probs

    def compute_loss(
        self,
        chosen_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_mask: torch.Tensor,
        rejected_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute DPO loss.

        Loss = -E[log σ(β * log_ratio_chosen - β * log_ratio_rejected)]

        Where log_ratio = log π_θ(y|x) - log π_ref(y|x)

        Args:
            chosen_ids: Chosen response IDs
            chosen_mask: Chosen attention mask
            chosen_labels: Chosen labels
            rejected_ids: Rejected response IDs
            rejected_mask: Rejected attention mask
            rejected_labels: Rejected labels

        Returns:
            loss: Scalar loss
            metrics: Dictionary of metrics
        """
        # Compute log probs under policy model
        policy_chosen_logps = self.get_logprobs(
            self.model, chosen_ids, chosen_mask, chosen_labels
        )
        policy_rejected_logps = self.get_logprobs(
            self.model, rejected_ids, rejected_mask, rejected_labels
        )

        # Compute log probs under reference model
        with torch.no_grad():
            ref_chosen_logps = self.get_logprobs(
                self.ref_model, chosen_ids, chosen_mask, chosen_labels
            )
            ref_rejected_logps = self.get_logprobs(
                self.ref_model, rejected_ids, rejected_mask, rejected_labels
            )

        # Compute log ratios: log π_θ / π_ref
        chosen_log_ratio = policy_chosen_logps - ref_chosen_logps
        rejected_log_ratio = policy_rejected_logps - ref_rejected_logps

        # DPO loss
        logits = self.beta * (chosen_log_ratio - rejected_log_ratio)
        loss = -F.logsigmoid(logits).mean()

        # Metrics
        with torch.no_grad():
            # Accuracy: how often is chosen preferred?
            accuracy = (chosen_log_ratio > rejected_log_ratio).float().mean().item()

            # Implicit rewards (for monitoring)
            # r(x,y) = β * log π(y|x) / π_ref(y|x)
            implicit_reward_chosen = self.beta * chosen_log_ratio
            implicit_reward_rejected = self.beta * rejected_log_ratio

            # KL divergence from reference (approximate)
            kl_chosen = (policy_chosen_logps - ref_chosen_logps).mean().item()
            kl_rejected = (policy_rejected_logps - ref_rejected_logps).mean().item()

        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy,
            'reward_chosen': implicit_reward_chosen.mean().item(),
            'reward_rejected': implicit_reward_rejected.mean().item(),
            'reward_margin': (implicit_reward_chosen - implicit_reward_rejected).mean().item(),
            'kl_chosen': kl_chosen,
            'kl_rejected': kl_rejected
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
        chosen_labels = batch['chosen_labels'].to(self.device)

        rejected_ids = batch['rejected_input_ids'].to(self.device)
        rejected_mask = batch['rejected_attention_mask'].to(self.device)
        rejected_labels = batch['rejected_labels'].to(self.device)

        # Forward pass
        loss, metrics = self.compute_loss(
            chosen_ids, chosen_mask, chosen_labels,
            rejected_ids, rejected_mask, rejected_labels
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.max_grad_norm
        )

        # Update weights
        self.optimizer.step()

        return metrics

    def train(
        self,
        train_dataset,
        epochs: int = 1,
        batch_size: int = 16,
        eval_dataset: Optional = None,
        eval_every: int = 100
    ):
        """Train with DPO.

        Args:
            train_dataset: Preference dataset
            epochs: Number of epochs (typically 1 for DPO)
            batch_size: Batch size
            eval_dataset: Optional evaluation dataset
            eval_every: Evaluate every N steps

        Returns:
            Trained model
        """
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        self.model.train()
        global_step = 0

        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch in pbar:
                metrics = self.train_step(batch)

                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'acc': f"{metrics['accuracy']:.3f}",
                    'margin': f"{metrics['reward_margin']:.3f}"
                })

                global_step += 1

        return self.model


class DPOPreferenceDataset(torch.utils.data.Dataset):
    """Dataset for DPO preference pairs with labels."""

    def __init__(
        self,
        prompts: list[str],
        chosen_responses: list[str],
        rejected_responses: list[str],
        tokenizer,
        max_length: int = 512,
        ignore_index: int = -100
    ):
        """Initialize dataset.

        Args:
            prompts: List of prompts
            chosen_responses: List of preferred responses
            rejected_responses: List of rejected responses
            tokenizer: Tokenizer
            max_length: Maximum sequence length
            ignore_index: Index for prompt tokens (not trained on)
        """
        assert len(prompts) == len(chosen_responses) == len(rejected_responses)
        self.prompts = prompts
        self.chosen_responses = chosen_responses
        self.rejected_responses = rejected_responses
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.prompts)

    def _prepare_example(
        self,
        prompt: str,
        response: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare a single example with labels.

        Args:
            prompt: Prompt text
            response: Response text

        Returns:
            input_ids, attention_mask, labels
        """
        # Combine
        full_text = f"{prompt}\n\n{response}"

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Find prompt boundary to create labels
        prompt_text = f"{prompt}\n\n"
        prompt_encoding = self.tokenizer(
            prompt_text,
            add_special_tokens=False
        )
        prompt_len = len(prompt_encoding['input_ids'])

        # Create labels (mask prompt, train on response)
        labels = encoding['input_ids'].clone().squeeze(0)
        labels[:prompt_len] = self.ignore_index

        return (
            encoding['input_ids'].squeeze(0),
            encoding['attention_mask'].squeeze(0),
            labels
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single preference pair.

        Args:
            idx: Index

        Returns:
            Dictionary with chosen and rejected inputs with labels
        """
        prompt = self.prompts[idx]
        chosen = self.chosen_responses[idx]
        rejected = self.rejected_responses[idx]

        # Prepare chosen
        chosen_ids, chosen_mask, chosen_labels = self._prepare_example(prompt, chosen)

        # Prepare rejected
        rejected_ids, rejected_mask, rejected_labels = self._prepare_example(prompt, rejected)

        return {
            'chosen_input_ids': chosen_ids,
            'chosen_attention_mask': chosen_mask,
            'chosen_labels': chosen_labels,
            'rejected_input_ids': rejected_ids,
            'rejected_attention_mask': rejected_mask,
            'rejected_labels': rejected_labels
        }
