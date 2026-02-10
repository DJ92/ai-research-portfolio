"""Simple quantization utilities.

Implements basic quantization for reducing model memory footprint.
"""

import torch
import torch.nn as nn


def quantize_tensor(
    tensor: torch.Tensor,
    bits: int = 8,
    symmetric: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a tensor to lower bit precision.

    Args:
        tensor: Tensor to quantize
        bits: Number of bits (4 or 8)
        symmetric: Use symmetric quantization

    Returns:
        quantized: Quantized tensor (int8 or int4)
        scale: Quantization scale
        zero_point: Zero point for asymmetric quantization
    """
    if bits not in [4, 8]:
        raise ValueError(f"Only 4-bit and 8-bit quantization supported, got {bits}")

    # Compute quantization parameters
    min_val = tensor.min()
    max_val = tensor.max()

    if symmetric:
        # Symmetric: range is [-max_abs, max_abs]
        max_abs = max(abs(min_val), abs(max_val))
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1

        scale = max_abs / qmax
        zero_point = torch.tensor(0.0)

    else:
        # Asymmetric: full range [min_val, max_val]
        qmin = 0
        qmax = 2 ** bits - 1

        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - min_val / scale

    # Quantize
    quantized = torch.clamp(
        torch.round(tensor / scale + zero_point),
        qmin,
        qmax
    )

    if bits == 8:
        quantized = quantized.to(torch.int8)
    else:  # 4-bit stored as int8 (using only lower 4 bits)
        quantized = quantized.to(torch.int8)

    return quantized, scale, zero_point


def dequantize_tensor(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor
) -> torch.Tensor:
    """Dequantize a tensor back to float.

    Args:
        quantized: Quantized tensor
        scale: Quantization scale
        zero_point: Zero point

    Returns:
        Dequantized float tensor
    """
    return (quantized.float() - zero_point) * scale


class QuantizedLinear(nn.Module):
    """Linear layer with quantized weights.

    Stores weights in quantized format (int8/int4) to save memory.
    Dequantizes on-the-fly during forward pass.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 8,
        bias: bool = True
    ):
        """Initialize quantized linear layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            bits: Quantization bits (4 or 8)
            bias: Whether to include bias
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits

        # Store quantized weight
        self.register_buffer(
            'weight_quantized',
            torch.zeros(out_features, in_features, dtype=torch.int8)
        )
        self.register_buffer('weight_scale', torch.tensor(1.0))
        self.register_buffer('weight_zero_point', torch.tensor(0.0))

        # Bias (float)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def quantize_weight(self, weight: torch.Tensor):
        """Quantize and store weight.

        Args:
            weight: Weight tensor to quantize
        """
        quantized, scale, zero_point = quantize_tensor(weight, bits=self.bits)
        self.weight_quantized = quantized
        self.weight_scale = scale
        self.weight_zero_point = zero_point

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with on-the-fly dequantization.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        # Dequantize weight
        weight = dequantize_tensor(
            self.weight_quantized,
            self.weight_scale,
            self.weight_zero_point
        )

        # Standard linear
        return torch.nn.functional.linear(x, weight, self.bias)

    def memory_footprint(self) -> dict:
        """Compute memory footprint.

        Returns:
            Dictionary with memory statistics
        """
        weight_bytes = self.weight_quantized.numel() * 1  # int8 = 1 byte
        scale_bytes = 4  # float32
        zero_point_bytes = 4  # float32

        if self.bias is not None:
            bias_bytes = self.bias.numel() * 4  # float32
        else:
            bias_bytes = 0

        total_bytes = weight_bytes + scale_bytes + zero_point_bytes + bias_bytes

        # Compare to full precision
        full_precision_bytes = (
            self.in_features * self.out_features * 4 +  # weight
            (self.out_features * 4 if self.bias is not None else 0)  # bias
        )

        return {
            'quantized_bytes': total_bytes,
            'full_precision_bytes': full_precision_bytes,
            'compression_ratio': full_precision_bytes / total_bytes
        }


def convert_linear_to_quantized(
    module: nn.Module,
    bits: int = 8,
    skip_names: list[str] | None = None
) -> nn.Module:
    """Convert all Linear layers to QuantizedLinear.

    Args:
        module: Module to convert
        bits: Quantization bits
        skip_names: Layer names to skip

    Returns:
        Module with quantized layers
    """
    skip_names = skip_names or []

    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and name not in skip_names:
            # Create quantized version
            quantized_layer = QuantizedLinear(
                child.in_features,
                child.out_features,
                bits=bits,
                bias=child.bias is not None
            )

            # Quantize and store weights
            quantized_layer.quantize_weight(child.weight.data)

            if child.bias is not None:
                quantized_layer.bias.data = child.bias.data

            # Replace
            setattr(module, name, quantized_layer)
        else:
            # Recurse
            convert_linear_to_quantized(child, bits, skip_names)

    return module


if __name__ == "__main__":
    # Quick test
    print("Testing quantization...")

    # Test tensor quantization
    tensor = torch.randn(100, 100)
    quantized, scale, zero_point = quantize_tensor(tensor, bits=8)
    dequantized = dequantize_tensor(quantized, scale, zero_point)

    error = (tensor - dequantized).abs().mean()
    print(f"8-bit quantization error: {error:.6f}")

    # Test quantized linear
    linear = nn.Linear(512, 512)
    quantized_linear = QuantizedLinear(512, 512, bits=8)
    quantized_linear.quantize_weight(linear.weight.data)

    x = torch.randn(2, 512)
    output_original = linear(x)
    output_quantized = quantized_linear(x)

    error = (output_original - output_quantized).abs().mean()
    print(f"Linear layer error: {error:.4f}")

    # Memory footprint
    mem = quantized_linear.memory_footprint()
    print(f"\nMemory footprint:")
    print(f"  Quantized: {mem['quantized_bytes']/1024:.1f} KB")
    print(f"  Full precision: {mem['full_precision_bytes']/1024:.1f} KB")
    print(f"  Compression: {mem['compression_ratio']:.2f}×")
