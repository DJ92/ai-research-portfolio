"""Simple dataset for pre-training."""

import torch
from torch.utils.data import Dataset
from typing import List, Dict


class SimpleTextDataset(Dataset):
    """Simple dataset that yields tokenized sequences.

    For real pre-training, use datasets library with streaming.
    This is a simplified version for demonstration.
    """

    def __init__(
        self,
        token_ids: List[List[int]],
        seq_len: int = 512,
        pad_token_id: int = 0
    ):
        """Initialize dataset.

        Args:
            token_ids: List of tokenized sequences
            seq_len: Target sequence length
            pad_token_id: ID for padding token
        """
        self.token_ids = token_ids
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example.

        Args:
            idx: Index

        Returns:
            Dictionary with input_ids and attention_mask
        """
        tokens = self.token_ids[idx]

        # Truncate or pad to seq_len
        if len(tokens) > self.seq_len:
            tokens = tokens[:self.seq_len]
        else:
            # Pad
            padding_len = self.seq_len - len(tokens)
            tokens = tokens + [self.pad_token_id] * padding_len

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1 if t != self.pad_token_id else 0 for t in tokens]

        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }


def create_dummy_dataset(
    num_examples: int = 1000,
    seq_len: int = 128,
    vocab_size: int = 1000
) -> SimpleTextDataset:
    """Create dummy dataset for testing.

    Args:
        num_examples: Number of examples
        seq_len: Sequence length
        vocab_size: Vocabulary size

    Returns:
        Dataset instance
    """
    # Generate random token sequences
    token_ids = [
        torch.randint(1, vocab_size, (seq_len,)).tolist()
        for _ in range(num_examples)
    ]

    return SimpleTextDataset(token_ids, seq_len=seq_len)


if __name__ == "__main__":
    # Quick test
    dataset = create_dummy_dataset(num_examples=10, seq_len=32, vocab_size=100)

    print(f"Dataset size: {len(dataset)}")

    # Get first example
    example = dataset[0]
    print(f"Input IDs shape: {example['input_ids'].shape}")
    print(f"Attention mask shape: {example['attention_mask'].shape}")
    print(f"First 10 tokens: {example['input_ids'][:10].tolist()}")
