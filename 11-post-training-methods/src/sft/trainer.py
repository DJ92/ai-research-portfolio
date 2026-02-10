"""Supervised Fine-Tuning (SFT) for instruction following.

Teaches pre-trained models to follow instructions by fine-tuning on
high-quality instruction-response pairs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from tqdm import tqdm


class SupervisedFineTuner:
    """Supervised fine-tuning on instruction-response pairs.

    Fine-tunes a pre-trained language model on demonstrations of desired behavior.
    This is the first stage of alignment, providing basic instruction-following.

    Why SFT works:
    - Shifts distribution toward helpful, formatted responses
    - Conditions model on instruction-following examples
    - Provides foundation for further alignment (RLHF/DPO)

    Limitations:
    - No notion of "better" vs "worse" (just imitation)
    - Can't optimize beyond training data
    - May hallucinate plausible-sounding errors
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        ignore_index: int = -100
    ):
        """Initialize SFT trainer.

        Args:
            model: Pre-trained language model
            tokenizer: Tokenizer for encoding text
            learning_rate: Learning rate (lower than pre-training)
            weight_decay: Weight decay for regularization
            max_grad_norm: Gradient clipping threshold
            ignore_index: Index to ignore in loss (for padding)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.ignore_index = ignore_index

        # Optimizer (AdamW with lower LR than pre-training)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )

        self.device = next(model.parameters()).device

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute SFT loss for a batch.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            labels: Target labels (batch, seq_len), with ignore_index for prompt tokens

        Returns:
            loss: Scalar loss
            metrics: Dictionary of metrics
        """
        # Forward pass
        logits = self.model(input_ids, attention_mask=attention_mask)

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Compute cross-entropy only on response tokens (not prompt)
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
            reduction='mean'
        )

        # Compute metrics
        with torch.no_grad():
            # Accuracy on response tokens
            predictions = shift_logits.argmax(dim=-1)
            response_mask = (shift_labels != self.ignore_index)
            if response_mask.sum() > 0:
                correct = (predictions == shift_labels) & response_mask
                accuracy = correct.sum().item() / response_mask.sum().item()
            else:
                accuracy = 0.0

        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy
        }

        return loss, metrics

    def prepare_training_batch(
        self,
        instruction: str,
        response: str,
        max_length: int = 512
    ) -> Dict[str, torch.Tensor]:
        """Prepare a training example.

        Format: <instruction>\n\n<response>
        Labels: Mask instruction tokens, train only on response

        Args:
            instruction: User instruction
            response: Model response
            max_length: Maximum sequence length

        Returns:
            Dictionary with input_ids, attention_mask, labels
        """
        # Combine instruction and response
        full_text = f"{instruction}\n\n{response}"

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize instruction separately to find boundary
        instruction_encoding = self.tokenizer(
            f"{instruction}\n\n",
            add_special_tokens=False
        )
        instruction_len = len(instruction_encoding['input_ids'])

        # Create labels (mask instruction, train on response)
        labels = encoding['input_ids'].clone()
        labels[:, :instruction_len] = self.ignore_index

        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': labels
        }

    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Single training step.

        Args:
            batch: Batch of data

        Returns:
            Metrics dictionary
        """
        # Move to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)

        # Forward pass
        loss, metrics = self.compute_loss(input_ids, attention_mask, labels)

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
        epochs: int = 3,
        batch_size: int = 16,
        eval_dataset: Optional = None,
        eval_every: int = 100
    ):
        """Train with supervised fine-tuning.

        Args:
            train_dataset: Training dataset (instruction-response pairs)
            epochs: Number of epochs
            batch_size: Batch size
            eval_dataset: Optional evaluation dataset
            eval_every: Evaluate every N steps

        Returns:
            Trained model
        """
        from torch.utils.data import DataLoader

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
                    'acc': f"{metrics['accuracy']:.3f}"
                })

                global_step += 1

                # Evaluation
                if eval_dataset is not None and global_step % eval_every == 0:
                    eval_metrics = self.evaluate(eval_dataset, batch_size)
                    print(f"\nStep {global_step}: {eval_metrics}")
                    self.model.train()

        return self.model

    @torch.no_grad()
    def evaluate(
        self,
        dataset,
        batch_size: int = 16
    ) -> Dict[str, float]:
        """Evaluate on dataset.

        Args:
            dataset: Evaluation dataset
            batch_size: Batch size

        Returns:
            Average metrics
        """
        from torch.utils.data import DataLoader

        self.model.eval()

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        for batch in loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            _, metrics = self.compute_loss(input_ids, attention_mask, labels)

            total_loss += metrics['loss']
            total_accuracy += metrics['accuracy']
            num_batches += 1

        return {
            'eval_loss': total_loss / num_batches,
            'eval_accuracy': total_accuracy / num_batches
        }


class InstructionDataset(torch.utils.data.Dataset):
    """Simple dataset for instruction-response pairs."""

    def __init__(
        self,
        instructions: list[str],
        responses: list[str],
        tokenizer,
        max_length: int = 512,
        ignore_index: int = -100
    ):
        """Initialize dataset.

        Args:
            instructions: List of instructions
            responses: List of corresponding responses
            tokenizer: Tokenizer
            max_length: Maximum sequence length
            ignore_index: Index for masked tokens
        """
        assert len(instructions) == len(responses)
        self.instructions = instructions
        self.responses = responses
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.instructions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example.

        Args:
            idx: Index

        Returns:
            Dictionary with input_ids, attention_mask, labels
        """
        instruction = self.instructions[idx]
        response = self.responses[idx]

        # Combine
        full_text = f"{instruction}\n\n{response}"

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Find instruction boundary
        instruction_text = f"{instruction}\n\n"
        instruction_encoding = self.tokenizer(
            instruction_text,
            add_special_tokens=False
        )
        instruction_len = len(instruction_encoding['input_ids'])

        # Create labels (mask instruction)
        labels = encoding['input_ids'].clone().squeeze(0)
        labels[:instruction_len] = self.ignore_index

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels
        }
