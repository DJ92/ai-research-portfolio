"""Masked Language Modeling (MLM) objective.

Used in BERT-style models for bidirectional pre-training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import random


class MaskedLanguageModeling:
    """Masked Language Modeling objective.

    Randomly masks tokens and trains model to predict them from bidirectional context.

    Masking strategy (from BERT):
    - 15% of tokens are selected for masking
    - Of those 15%:
      - 80% are replaced with [MASK]
      - 10% are replaced with random token
      - 10% are kept unchanged

    Why this strategy:
    - 80% [MASK]: Main learning signal
    - 10% random: Prevents model from relying solely on [MASK]
    - 10% unchanged: Forces model to learn from all tokens
    """

    def __init__(
        self,
        model: nn.Module,
        mask_token_id: int,
        vocab_size: int,
        mask_prob: float = 0.15,
        mask_token_prob: float = 0.8,
        random_token_prob: float = 0.1,
        ignore_index: int = -100,
        special_tokens: list[int] | None = None
    ):
        """Initialize MLM objective.

        Args:
            model: Language model (should output logits)
            mask_token_id: ID of [MASK] token
            vocab_size: Size of vocabulary
            mask_prob: Probability of masking each token (default 15%)
            mask_token_prob: Probability of replacing with [MASK] (default 80% of masked)
            random_token_prob: Probability of replacing with random token (default 10% of masked)
            ignore_index: Index to ignore in loss
            special_tokens: Special tokens to never mask (e.g., [CLS], [SEP], [PAD])
        """
        self.model = model
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.mask_prob = mask_prob
        self.mask_token_prob = mask_token_prob
        self.random_token_prob = random_token_prob
        self.ignore_index = ignore_index
        self.special_tokens = set(special_tokens or [])

        # Validate probabilities
        assert 0 <= mask_prob <= 1
        assert 0 <= mask_token_prob <= 1
        assert 0 <= random_token_prob <= 1
        assert mask_token_prob + random_token_prob <= 1

    def create_masked_input(
        self,
        input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create masked input and labels.

        Args:
            input_ids: Original input IDs (batch, seq_len)

        Returns:
            masked_input: Input with some tokens masked (batch, seq_len)
            labels: Original tokens, with non-masked positions set to ignore_index
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Create labels (start with ignore_index everywhere)
        labels = torch.full_like(input_ids, self.ignore_index)

        # Clone input for masking
        masked_input = input_ids.clone()

        # Process each sequence in the batch
        for i in range(batch_size):
            # Determine which tokens to mask
            mask_indices = []
            for j in range(seq_len):
                token_id = input_ids[i, j].item()

                # Skip special tokens
                if token_id in self.special_tokens:
                    continue

                # Randomly select for masking
                if random.random() < self.mask_prob:
                    mask_indices.append(j)

            # Apply masking strategy to selected tokens
            for j in mask_indices:
                original_token = input_ids[i, j].item()

                # Save original token in labels
                labels[i, j] = original_token

                # Determine masking strategy
                rand = random.random()

                if rand < self.mask_token_prob:
                    # 80%: Replace with [MASK]
                    masked_input[i, j] = self.mask_token_id

                elif rand < self.mask_token_prob + self.random_token_prob:
                    # 10%: Replace with random token
                    random_token = random.randint(0, self.vocab_size - 1)
                    # Avoid replacing with special tokens
                    while random_token in self.special_tokens:
                        random_token = random.randint(0, self.vocab_size - 1)
                    masked_input[i, j] = random_token

                # else: 10%: Keep original (no change to masked_input)

        return masked_input, labels

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute MLM loss for a batch.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            attention_mask: Optional mask for padding

        Returns:
            loss: Scalar loss tensor
            metrics: Dictionary of metrics (loss, accuracy, num_masked)
        """
        # Create masked input and labels
        masked_input, labels = self.create_masked_input(input_ids)

        # Forward pass with bidirectional attention
        logits = self.model(masked_input, attention_mask=attention_mask)

        # Compute loss only on masked positions
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),  # (batch*seq_len, vocab)
            labels.view(-1),                    # (batch*seq_len,)
            ignore_index=self.ignore_index,
            reduction='mean'
        )

        # Compute metrics
        with torch.no_grad():
            # Count masked tokens
            num_masked = (labels != self.ignore_index).sum().item()

            # Accuracy on masked positions
            predictions = logits.argmax(dim=-1)
            masked_positions = labels != self.ignore_index
            if num_masked > 0:
                correct = (predictions == labels) & masked_positions
                accuracy = correct.sum().item() / num_masked
            else:
                accuracy = 0.0

        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy,
            'num_masked': num_masked,
            'mask_percentage': num_masked / labels.numel() * 100
        }

        return loss, metrics


def verify_masking_strategy(mlm: MaskedLanguageModeling, num_samples: int = 10000):
    """Verify that masking strategy follows 80/10/10 split.

    Args:
        mlm: MLM objective instance
        num_samples: Number of samples to test

    Returns:
        Statistics about masking strategy
    """
    mask_count = 0
    random_count = 0
    unchanged_count = 0

    # Create dummy input
    input_ids = torch.randint(0, mlm.vocab_size, (1, 100))

    for _ in range(num_samples):
        masked_input, labels = mlm.create_masked_input(input_ids)

        # Check masked positions
        masked_positions = labels[0] != mlm.ignore_index

        for j in range(100):
            if masked_positions[j]:
                original = labels[0, j].item()
                masked = masked_input[0, j].item()

                if masked == mlm.mask_token_id:
                    mask_count += 1
                elif masked == original:
                    unchanged_count += 1
                else:
                    random_count += 1

    total = mask_count + random_count + unchanged_count

    if total == 0:
        return None

    return {
        'mask_pct': mask_count / total * 100,
        'random_pct': random_count / total * 100,
        'unchanged_pct': unchanged_count / total * 100,
        'total_samples': total
    }


if __name__ == "__main__":
    # Quick test
    print("Testing MLM objective...")

    # Create dummy model (for testing)
    class DummyModel(nn.Module):
        def __init__(self, vocab_size, d_model=256):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.output = nn.Linear(d_model, vocab_size)

        def forward(self, input_ids, attention_mask=None):
            x = self.embedding(input_ids)
            return self.output(x)

    vocab_size = 1000
    mask_token_id = vocab_size - 1  # Use last token as [MASK]

    model = DummyModel(vocab_size)

    # Create MLM objective
    mlm = MaskedLanguageModeling(
        model=model,
        mask_token_id=mask_token_id,
        vocab_size=vocab_size
    )

    # Test on batch
    batch_size, seq_len = 4, 32
    input_ids = torch.randint(0, vocab_size - 1, (batch_size, seq_len))

    # Compute loss
    loss, metrics = mlm.compute_loss(input_ids)

    print(f"MLM Loss: {metrics['loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Masked tokens: {metrics['num_masked']} ({metrics['mask_percentage']:.1f}%)")

    # Verify masking strategy
    print("\nVerifying masking strategy (should be ~80/10/10)...")
    stats = verify_masking_strategy(mlm)
    if stats:
        print(f"[MASK]: {stats['mask_pct']:.1f}%")
        print(f"Random: {stats['random_pct']:.1f}%")
        print(f"Unchanged: {stats['unchanged_pct']:.1f}%")
