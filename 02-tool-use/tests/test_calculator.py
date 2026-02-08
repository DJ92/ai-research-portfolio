"""
Tests for Calculator tool.
"""

import pytest
from src.tools.calculator import Calculator


@pytest.fixture
def calculator():
    """Create Calculator instance for testing."""
    return Calculator()


class TestCalculatorBasics:
    """Test basic calculator functionality."""

    def test_addition(self, calculator):
        """Test addition operation."""
        result = calculator.execute(operation="add", a=5, b=3)
        assert result.success is True
        assert result.output == 8

    def test_subtraction(self, calculator):
        """Test subtraction operation."""
        result = calculator.execute(operation="subtract", a=10, b=3)
        assert result.success is True
        assert result.output == 7

    def test_multiplication(self, calculator):
        """Test multiplication operation."""
        result = calculator.execute(operation="multiply", a=4, b=5)
        assert result.success is True
        assert result.output == 20

    def test_division(self, calculator):
        """Test division operation."""
        result = calculator.execute(operation="divide", a=10, b=2)
        assert result.success is True
        assert result.output == 5.0


class TestCalculatorEdgeCases:
    """Test edge cases and error handling."""

    def test_divide_by_zero(self, calculator):
        """Test division by zero returns error."""
        result = calculator.execute(operation="divide", a=10, b=0)
        assert result.success is False
        assert "Division by zero" in result.error

    def test_invalid_operation(self, calculator):
        """Test invalid operation returns error."""
        result = calculator.execute(operation="power", a=2, b=3)
        assert result.success is False
        assert "Unknown operation" in result.error

    def test_negative_numbers(self, calculator):
        """Test operations with negative numbers."""
        result = calculator.execute(operation="add", a=-5, b=3)
        assert result.success is True
        assert result.output == -2

    def test_float_operations(self, calculator):
        """Test operations with floating point numbers."""
        result = calculator.execute(operation="multiply", a=2.5, b=4.0)
        assert result.success is True
        assert result.output == 10.0


class TestCalculatorSchema:
    """Test tool schema generation."""

    def test_anthropic_schema(self, calculator):
        """Test Anthropic schema format."""
        schema = calculator.to_anthropic_schema()

        assert schema["name"] == "calculator"
        assert "description" in schema
        assert "input_schema" in schema
        assert "properties" in schema["input_schema"]
        assert "operation" in schema["input_schema"]["properties"]
        assert "a" in schema["input_schema"]["properties"]
        assert "b" in schema["input_schema"]["properties"]

    def test_openai_schema(self, calculator):
        """Test OpenAI schema format."""
        schema = calculator.to_openai_schema()

        assert schema["type"] == "function"
        assert "function" in schema
        assert schema["function"]["name"] == "calculator"
        assert "parameters" in schema["function"]

    def test_parameter_validation(self, calculator):
        """Test parameter validation."""
        # Valid params
        is_valid, error = calculator.validate_params(operation="add", a=1, b=2)
        assert is_valid is True
        assert error is None

        # Missing required param
        is_valid, error = calculator.validate_params(operation="add", a=1)
        assert is_valid is False
        assert "Missing required parameter" in error

        # Invalid enum value
        is_valid, error = calculator.validate_params(operation="invalid", a=1, b=2)
        assert is_valid is False
        assert "Invalid value" in error


class TestCalculatorMetadata:
    """Test metadata in results."""

    def test_metadata_included(self, calculator):
        """Test that metadata is included in successful results."""
        result = calculator.execute(operation="add", a=5, b=3)

        assert "metadata" in result.model_fields_set or hasattr(result, "metadata")
        assert result.metadata["operation"] == "add"
        assert result.metadata["a"] == 5
        assert result.metadata["b"] == 3
