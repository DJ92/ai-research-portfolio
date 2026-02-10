"""Causal Language Modeling (CLM) objective.

Used in GPT-style models for next-token prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class CausalLanguageModeling:
    """Causal Language Modeling objective (next-token prediction).

    The model predicts each token given all previous tokens:
    P(x_t | x_1, ..., x_{t-1})

    Loss is cross-entropy between predicted and actual next tokens.

    Why it works:
    - Forces model to compress world knowledge to make accurate predictions
    - Every token is a training example (efficient)
    - Naturally aligns with generation tasks
    - No special masking or tokens needed
    """

    def __init__(self, model: nn.Module, ignore_index: int = -100):
        """Initialize CLM objective.

        Args:
            model: Language model (should output logits)
            ignore_index: Index to ignore in loss (for padding)
        """
        self.model = model
        self.ignore_index = ignore_index

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute CLM loss for a batch.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            attention_mask: Optional mask for padding

        Returns:
            loss: Scalar loss tensor
            metrics: Dictionary of metrics (loss, perplexity, accuracy)
        """
        batch_size, seq_len = input_ids.shape

        # Forward pass
        logits = self.model(input_ids, attention_mask=attention_mask)

        # Shift logits and labels for next-token prediction
        # logits[t] predicts token[t+1]
        shift_logits = logits[:, :-1, :].contiguous()  # (batch, seq_len-1, vocab)
        shift_labels = input_ids[:, 1:].contiguous()   # (batch, seq_len-1)

        # Create mask for valid positions (non-padding)
        if attention_mask is not None:
            shift_mask = attention_mask[:, 1:].contiguous()
            # Set padding positions to ignore_index
            shift_labels = shift_labels.masked_fill(shift_mask == 0, self.ignore_index)

        # Compute cross-entropy loss
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),  # (batch*(seq_len-1), vocab)
            shift_labels.view(-1),                          # (batch*(seq_len-1),)
            ignore_index=self.ignore_index,
            reduction='mean'
        )

        # Compute metrics
        with torch.no_grad():
            # Perplexity = exp(loss)
            perplexity = torch.exp(loss).item()

            # Accuracy (next-token prediction)
            predictions = shift_logits.argmax(dim=-1)
            if attention_mask is not None:
                # Only count non-padding tokens
                correct = (predictions == shift_labels) & (shift_mask == 1)
                accuracy = correct.sum().item() / shift_mask.sum().item()
            else:
                correct = predictions == shift_labels
                accuracy = correct.float().mean().item()

        metrics = {
            'loss': loss.item(),
            'perplexity': perplexity,
            'accuracy': accuracy
        }

        return loss, metrics

    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None
    ) -> torch.Tensor:
        """Generate text using the CLM model.

        Args:
            prompt_ids: Prompt token IDs (batch, prompt_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Returns:
            Generated token IDs (batch, prompt_len + max_new_tokens)
        """
        return self.model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )


def compute_perplexity(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None
) -> float:
    """Compute perplexity on a batch of sequences.

    Perplexity measures how well the model predicts the next token.
    Lower is better. Random guessing gives perplexity = vocab_size.

    Args:
        model: Language model
        input_ids: Input token IDs
        attention_mask: Optional padding mask

    Returns:
        Perplexity value
    """
    clm = CausalLanguageModeling(model)

    with torch.no_grad():
        loss, metrics = clm.compute_loss(input_ids, attention_mask)

    return metrics['perplexity']


if __name__ == "__main__":
    # Quick test
    from ...transformer_architecture.src.models import create_gpt_small

    # Create small model
    model = create_gpt_small(vocab_size=1000)

    # Create dummy batch
    batch_size, seq_len = 4, 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    # Compute loss
    clm = CausalLanguageModeling(model)
    loss, metrics = clm.compute_loss(input_ids)

    print(f"CLM Loss: {metrics['loss']:.4f}")
    print(f"Perplexity: {metrics['perplexity']:.2f}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")

    # Generate
    prompt = torch.randint(0, 1000, (1, 10))
    generated = clm.generate(prompt, max_new_tokens=20)
    print(f"Generated shape: {generated.shape}")
