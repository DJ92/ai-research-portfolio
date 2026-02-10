"""Pre-training objectives for language models."""

from .clm import CausalLanguageModeling, compute_perplexity
from .mlm import MaskedLanguageModeling, verify_masking_strategy

__all__ = [
    "CausalLanguageModeling",
    "MaskedLanguageModeling",
    "compute_perplexity",
    "verify_masking_strategy",
]
