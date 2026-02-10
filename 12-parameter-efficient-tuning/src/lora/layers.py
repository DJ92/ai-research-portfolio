"""LoRA (Low-Rank Adaptation) layer implementations.

Implements efficient fine-tuning by adding low-rank updates to frozen weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class LoRALayer(nn.Module):
    """Base LoRA layer.

    Implements low-rank adaptation: h = W₀x + BAx

    Where:
    - W₀: Frozen pre-trained weights
    - B, A: Trainable low-rank matrices
    - Rank r << min(d, k) for efficiency
    """

    def __init__(
        self,
        rank: int,
        alpha: float = 1.0,
        dropout: float = 0.0
    ):
        """Initialize LoRA layer.

        Args:
            rank: Rank of low-rank decomposition
            alpha: Scaling factor (output scaled by alpha/rank)
            dropout: Dropout probability for LoRA path
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def reset_parameters(self, lora_A: nn.Module, lora_B: nn.Module):
        """Initialize LoRA weights.

        Uses Kaiming uniform for A, zeros for B (like original paper).
        This ensures LoRA contributes nothing at initialization.
        """
        nn.init.kaiming_uniform_(lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(lora_B.weight)


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation.

    Forward: h = W₀x + (α/r) * BAx

    Where:
    - W₀: Frozen pretrained weights (in_features, out_features)
    - B: Trainable matrix (out_features, rank)
    - A: Trainable matrix (rank, in_features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = True,
        pretrained_weight: Optional[torch.Tensor] = None
    ):
        """Initialize LoRA linear layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: LoRA rank
            alpha: LoRA alpha (scaling factor)
            dropout: Dropout probability
            bias: Whether to include bias
            pretrained_weight: Optional pretrained weight to freeze
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Frozen pretrained weights
        if pretrained_weight is not None:
            self.weight = nn.Parameter(pretrained_weight, requires_grad=False)
        else:
            self.weight = nn.Parameter(torch.zeros(out_features, in_features))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            self.weight.requires_grad = False

        # Bias (trainable if enabled)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # LoRA matrices (trainable)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA.

        Args:
            x: Input tensor (batch, ..., in_features)

        Returns:
            Output tensor (batch, ..., out_features)
        """
        # Pretrained path (frozen)
        result = F.linear(x, self.weight, self.bias)

        # LoRA path (trainable)
        if self.rank > 0:
            lora_out = self.lora_A(x)
            if self.dropout is not None:
                lora_out = self.dropout(lora_out)
            lora_out = self.lora_B(lora_out)
            result = result + self.scaling * lora_out

        return result

    def merge_weights(self) -> nn.Linear:
        """Merge LoRA weights into base weights for inference.

        Creates a standard Linear layer with merged weights.
        Useful for deployment (removes LoRA overhead).

        Returns:
            Merged linear layer
        """
        # Compute merged weight: W = W₀ + α/r * BA
        delta_weight = self.lora_B.weight @ self.lora_A.weight
        merged_weight = self.weight + self.scaling * delta_weight

        # Create standard linear layer
        merged_layer = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None
        )
        merged_layer.weight = nn.Parameter(merged_weight)
        if self.bias is not None:
            merged_layer.bias = nn.Parameter(self.bias)

        return merged_layer

    def get_lora_parameters(self):
        """Get only LoRA parameters (for optimizer).

        Returns:
            Iterator of LoRA parameters
        """
        for param in self.lora_A.parameters():
            yield param
        for param in self.lora_B.parameters():
            yield param
        if self.bias is not None:
            yield self.bias


def mark_only_lora_as_trainable(model: nn.Module):
    """Mark only LoRA parameters as trainable.

    Freezes all parameters except LoRA matrices and biases.

    Args:
        model: Model with LoRA layers
    """
    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze LoRA parameters
    for module in model.modules():
        if isinstance(module, LoRALinear):
            for param in module.get_lora_parameters():
                param.requires_grad = True


def get_lora_state_dict(model: nn.Module) -> dict:
    """Get state dict containing only LoRA parameters.

    Useful for saving only the adapted weights (tiny compared to full model).

    Args:
        model: Model with LoRA layers

    Returns:
        State dict with only LoRA parameters
    """
    lora_state = {}

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[f"{name}.lora_A.weight"] = module.lora_A.weight
            lora_state[f"{name}.lora_B.weight"] = module.lora_B.weight
            if module.bias is not None:
                lora_state[f"{name}.bias"] = module.bias

    return lora_state


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters.

    Args:
        model: Model to analyze

    Returns:
        (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # Quick test
    print("Testing LoRA layers...")

    # Create LoRA linear layer
    in_features, out_features = 512, 512
    lora_layer = LoRALinear(
        in_features,
        out_features,
        rank=8,
        alpha=16.0
    )

    # Forward pass
    x = torch.randn(2, 10, in_features)
    output = lora_layer(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Count parameters
    total, trainable = count_parameters(lora_layer)
    print(f"\nParameters:")
    print(f"  Total: {total:,}")
    print(f"  Trainable: {trainable:,}")
    print(f"  Trainable %: {trainable/total*100:.2f}%")

    # Test merging
    merged = lora_layer.merge_weights()
    print(f"\nMerged layer type: {type(merged)}")

    # Test that merged produces same output
    with torch.no_grad():
        output_lora = lora_layer(x)
        output_merged = merged(x)
        print(f"Outputs match: {torch.allclose(output_lora, output_merged, atol=1e-5)}")
