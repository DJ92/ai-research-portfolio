"""Simple pre-training trainer."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict
from tqdm import tqdm
import math


class PreTrainer:
    """Simple trainer for language model pre-training.

    Handles training loop, evaluation, and checkpointing for both CLM and MLM objectives.
    """

    def __init__(
        self,
        model: nn.Module,
        objective,  # CLM or MLM objective
        train_dataset,
        val_dataset: Optional = None,
        batch_size: int = 32,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        max_steps: int = 10000,
        eval_every: int = 500,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 500
    ):
        """Initialize trainer.

        Args:
            model: Language model to train
            objective: Pre-training objective (CLM or MLM)
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            batch_size: Batch size
            learning_rate: Peak learning rate
            weight_decay: Weight decay for AdamW
            max_steps: Maximum training steps
            eval_every: Evaluate every N steps
            device: Device to train on
            gradient_accumulation_steps: Gradient accumulation
            max_grad_norm: Maximum gradient norm for clipping
            warmup_steps: Linear warmup steps
        """
        self.model = model.to(device)
        self.objective = objective
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.eval_every = eval_every
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0  # Simple for now
        )

        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )
        else:
            self.val_loader = None

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)  # GPT-style betas
        )

        # Learning rate scheduler (linear warmup + cosine decay)
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate

        # Tracking
        self.global_step = 0
        self.training_metrics = []

    def get_lr(self, step: int) -> float:
        """Get learning rate for current step (warmup + cosine decay).

        Args:
            step: Current training step

        Returns:
            Learning rate
        """
        if step < self.warmup_steps:
            # Linear warmup
            return self.learning_rate * step / self.warmup_steps
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            return self.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step.

        Args:
            batch: Batch of data

        Returns:
            Metrics dictionary
        """
        # Move to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Forward pass
        loss, metrics = self.objective.compute_loss(input_ids, attention_mask)

        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps

        # Backward pass
        loss.backward()

        return metrics

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set.

        Returns:
            Average metrics
        """
        if self.val_loader is None:
            return {}

        self.model.eval()

        total_loss = 0.0
        total_perplexity = 0.0
        total_accuracy = 0.0
        num_batches = 0

        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            _, metrics = self.objective.compute_loss(input_ids, attention_mask)

            total_loss += metrics['loss']
            total_perplexity += metrics.get('perplexity', 0.0)
            total_accuracy += metrics.get('accuracy', 0.0)
            num_batches += 1

        avg_metrics = {
            'val_loss': total_loss / num_batches,
            'val_perplexity': total_perplexity / num_batches,
            'val_accuracy': total_accuracy / num_batches
        }

        self.model.train()

        return avg_metrics

    def train(self):
        """Main training loop."""
        self.model.train()

        train_iter = iter(self.train_loader)
        pbar = tqdm(total=self.max_steps, desc="Training")

        while self.global_step < self.max_steps:
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                # Restart iterator
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            # Training step
            metrics = self.train_step(batch)

            # Update weights every gradient_accumulation_steps
            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )

                # Update weights
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Update learning rate
                lr = self.get_lr(self.global_step)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

            # Logging
            if self.global_step % 10 == 0:
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'lr': f"{self.get_lr(self.global_step):.2e}"
                })

            # Evaluation
            if self.global_step % self.eval_every == 0 and self.global_step > 0:
                eval_metrics = self.evaluate()
                print(f"\nStep {self.global_step}: {eval_metrics}")

                self.training_metrics.append({
                    'step': self.global_step,
                    **metrics,
                    **eval_metrics
                })

            self.global_step += 1
            pbar.update(1)

        pbar.close()

        # Final evaluation
        if self.val_loader is not None:
            final_metrics = self.evaluate()
            print(f"\nFinal evaluation: {final_metrics}")

        return self.training_metrics
