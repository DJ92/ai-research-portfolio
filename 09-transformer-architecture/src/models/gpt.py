"""GPT-style decoder-only transformer model.

Implements a causal language model similar to GPT-2/GPT-3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .transformer_blocks import TransformerDecoderBlock
from ..positional import LearnedPositionalEncoding, SinusoidalPositionalEncoding


class GPTModel(nn.Module):
    """GPT-style decoder-only transformer.

    Architecture:
    Input → Token Embedding → Positional Encoding
      → [Masked Self-Attention → FFN] × N layers
      → Layer Norm → Output Projection → Logits

    Key characteristics:
    - Causal masking (can only attend to past tokens)
    - Autoregressive generation (predicts next token)
    - Pre-LN for stable training
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        d_ff: int = 3072,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        use_sinusoidal_pos: bool = False,
        tie_weights: bool = True
    ):
        """Initialize GPT model.

        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension (typically 4 * d_model)
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            use_sinusoidal_pos: Use sinusoidal (True) or learned (False) positions
            tie_weights: Tie input and output embeddings (reduces params)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        if use_sinusoidal_pos:
            self.pos_encoding = SinusoidalPositionalEncoding(
                d_model, max_seq_len, dropout
            )
        else:
            self.pos_encoding = LearnedPositionalEncoding(
                d_model, max_seq_len, dropout
            )

        # Transformer decoder blocks
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                pre_norm=True
            )
            for _ in range(num_layers)
        ])

        # Final layer norm (for Pre-LN architecture)
        self.ln_f = nn.LayerNorm(d_model)

        # Output projection (vocabulary logits)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights between input and output embeddings
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using GPT-2 style initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def _create_causal_mask(
        self,
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """Create causal mask for autoregressive generation.

        Mask prevents attending to future positions:
        [[False,  True,  True],
         [False, False,  True],
         [False, False, False]]

        Args:
            seq_len: Sequence length
            device: Device to create mask on

        Returns:
            Causal mask (seq_len, seq_len)
        """
        # Lower triangular matrix (1s below diagonal)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        # Convert to boolean (True = mask out)
        return mask.bool()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, list]:
        """Forward pass through GPT model.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            attention_mask: Optional mask for padding (batch, seq_len)
            return_hidden_states: Return intermediate layer outputs

        Returns:
            logits: Next token logits (batch, seq_len, vocab_size)
            hidden_states: Layer outputs (if return_hidden_states=True)
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        # Token embeddings
        x = self.token_embedding(input_ids)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Create causal mask
        causal_mask = self._create_causal_mask(seq_len, device)

        # Combine with padding mask if provided
        if attention_mask is not None:
            # Expand attention_mask: (batch, seq_len) -> (batch, 1, seq_len)
            # Then broadcast to (batch, seq_len, seq_len)
            padding_mask = attention_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
            # Combine: mask out if either causal OR padding
            combined_mask = causal_mask.unsqueeze(0) | ~padding_mask.bool()
        else:
            combined_mask = causal_mask

        # Pass through transformer blocks
        hidden_states = []
        for block in self.blocks:
            x = block(x, combined_mask)
            if return_hidden_states:
                hidden_states.append(x)

        # Final layer norm
        x = self.ln_f(x)

        # Project to vocabulary
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)

        if return_hidden_states:
            return logits, hidden_states
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True
    ) -> torch.Tensor:
        """Generate text autoregressively.

        Args:
            input_ids: Prompt tokens (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens (nucleus filtering)
            top_p: Keep top tokens with cumulative probability p
            do_sample: Sample from distribution (True) or greedy (False)

        Returns:
            Generated token IDs (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Truncate if sequence exceeds max length
            input_ids_cond = input_ids if input_ids.size(1) <= self.max_seq_len \
                else input_ids[:, -self.max_seq_len:]

            # Forward pass
            logits = self(input_ids_cond)  # (batch, seq_len, vocab_size)

            # Get logits for next token (last position)
            next_token_logits = logits[:, -1, :]  # (batch, vocab_size)

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('inf')

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least one token
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0

                # Scatter back to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = -float('inf')

            # Sample or greedy decode
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Get number of parameters.

        Args:
            non_embedding: Exclude embedding parameters

        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
            if hasattr(self.pos_encoding, 'position_embeddings'):
                n_params -= self.pos_encoding.position_embeddings.weight.numel()
        return n_params


def create_gpt_small(vocab_size: int = 50257) -> GPTModel:
    """Create small GPT model (~50M params, similar to GPT-2 small).

    Args:
        vocab_size: Vocabulary size

    Returns:
        GPT model instance
    """
    return GPTModel(
        vocab_size=vocab_size,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        max_seq_len=1024,
        dropout=0.1
    )


def create_gpt_medium(vocab_size: int = 50257) -> GPTModel:
    """Create medium GPT model (~100M params).

    Args:
        vocab_size: Vocabulary size

    Returns:
        GPT model instance
    """
    return GPTModel(
        vocab_size=vocab_size,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=3072,
        max_seq_len=1024,
        dropout=0.1
    )


def create_gpt_large(vocab_size: int = 50257) -> GPTModel:
    """Create large GPT model (~350M params, similar to GPT-2 large).

    Args:
        vocab_size: Vocabulary size

    Returns:
        GPT model instance
    """
    return GPTModel(
        vocab_size=vocab_size,
        d_model=1024,
        num_layers=24,
        num_heads=16,
        d_ff=4096,
        max_seq_len=1024,
        dropout=0.1
    )


if __name__ == "__main__":
    # Quick test
    model = create_gpt_small()
    print(f"GPT Small: {model.get_num_params() / 1e6:.1f}M parameters")

    # Test forward pass
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len))

    logits = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")

    # Test generation
    prompt = torch.randint(0, model.vocab_size, (1, 5))
    generated = model.generate(prompt, max_new_tokens=10, temperature=0.8)
    print(f"Generated shape: {generated.shape}")
