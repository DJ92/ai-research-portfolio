"""Tests for parameter-efficient fine-tuning methods (simplified)."""

import pytest
import torch
import torch.nn as nn

import sys
sys.path.insert(0, '/Users/djoshi/Desktop/Codebase/ai-research-portfolio/12-parameter-efficient-tuning')

from src.lora import LoRALinear, mark_only_lora_as_trainable, get_lora_state_dict, count_parameters
from src.quantization import quantize_tensor, dequantize_tensor, QuantizedLinear


class TestLoRA:
    """Test LoRA layers."""

    def test_lora_linear_forward(self):
        """Test LoRA linear forward pass."""
        lora_layer = LoRALinear(128, 128, rank=8)
        x = torch.randn(2, 10, 128)
        output = lora_layer(x)
        assert output.shape == (2, 10, 128)

    def test_lora_reduces_parameters(self):
        """Test that LoRA reduces trainable parameters."""
        regular_layer = nn.Linear(512, 512)
        lora_layer = LoRALinear(512, 512, rank=8)
        mark_only_lora_as_trainable(lora_layer)
        
        regular_params = sum(p.numel() for p in regular_layer.parameters())
        trainable_params = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
        
        assert trainable_params < regular_params * 0.2  # Much less than 20%

    def test_lora_merge_weights(self):
        """Test merging LoRA weights."""
        lora_layer = LoRALinear(128, 128, rank=8)
        
        # Train briefly
        x = torch.randn(4, 128)
        optimizer = torch.optim.SGD(lora_layer.get_lora_parameters(), lr=0.01)
        for _ in range(5):
            loss = (lora_layer(x) - torch.randn(4, 128)).pow(2).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Merge
        merged_layer = lora_layer.merge_weights()
        
        # Outputs should match
        test_x = torch.randn(2, 128)
        with torch.no_grad():
            assert torch.allclose(lora_layer(test_x), merged_layer(test_x), atol=1e-5)

    def test_lora_state_dict(self):
        """Test saving only LoRA parameters."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = LoRALinear(128, 128, rank=8)
                
        model = SimpleModel()
        lora_state = get_lora_state_dict(model)
        
        assert 'layer1.lora_A.weight' in lora_state
        assert 'layer1.lora_B.weight' in lora_state


class TestQuantization:
    """Test quantization."""

    def test_quantize_8bit(self):
        """Test 8-bit quantization."""
        tensor = torch.randn(100, 100)
        quantized, scale, zero_point = quantize_tensor(tensor, bits=8)
        
        assert quantized.dtype == torch.int8
        
        dequantized = dequantize_tensor(quantized, scale, zero_point)
        error = (tensor - dequantized).abs().mean()
        assert error < 0.1

    def test_quantized_linear(self):
        """Test quantized linear layer."""
        quantized_layer = QuantizedLinear(128, 128, bits=8)
        weight = torch.randn(128, 128)
        quantized_layer.quantize_weight(weight)
        
        x = torch.randn(2, 128)
        output = quantized_layer(x)
        assert output.shape == (2, 128)

    def test_memory_footprint(self):
        """Test memory reduction."""
        quantized_layer = QuantizedLinear(512, 512, bits=8)
        weight = torch.randn(512, 512)
        quantized_layer.quantize_weight(weight)
        
        mem = quantized_layer.memory_footprint()
        assert mem['quantized_bytes'] < mem['full_precision_bytes']
        assert mem['compression_ratio'] > 3.0


class TestParameterEfficiency:
    """Test parameter efficiency."""

    def test_parameter_counts(self):
        """Test LoRA parameter reduction."""
        full_layer = nn.Linear(1024, 1024)
        lora_layer = LoRALinear(1024, 1024, rank=8)
        mark_only_lora_as_trainable(lora_layer)
        
        full_params = sum(p.numel() for p in full_layer.parameters())
        lora_params = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
        
        reduction = lora_params / full_params
        assert reduction < 0.05  # Less than 5%
