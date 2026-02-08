"""
Calculator tool for basic arithmetic operations.
"""

from typing import List
from .base_tool import Tool, ToolParameter, ToolResult


class Calculator(Tool):
    """Performs basic arithmetic operations."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Perform basic arithmetic operations (add, subtract, multiply, divide)"

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="operation",
                type="string",
                description="The arithmetic operation to perform",
                required=True,
                enum=["add", "subtract", "multiply", "divide"],
            ),
            ToolParameter(
                name="a",
                type="number",
                description="The first operand",
                required=True,
            ),
            ToolParameter(
                name="b",
                type="number",
                description="The second operand",
                required=True,
            ),
        ]

    def execute(self, operation: str, a: float, b: float) -> ToolResult:
        """
        Execute arithmetic operation.

        Args:
            operation: One of "add", "subtract", "multiply", "divide"
            a: First operand
            b: Second operand

        Returns:
            ToolResult with the computed value
        """
        try:
            if operation == "add":
                result = a + b
            elif operation == "subtract":
                result = a - b
            elif operation == "multiply":
                result = a * b
            elif operation == "divide":
                if b == 0:
                    return ToolResult(
                        success=False,
                        output=None,
                        error="Division by zero is undefined",
                    )
                result = a / b
            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Unknown operation: {operation}",
                )

            return ToolResult(
                success=True,
                output=result,
                metadata={"operation": operation, "a": a, "b": b},
            )

        except Exception as e:
            return ToolResult(
                success=False, output=None, error=f"Calculation error: {str(e)}"
            )


# Example usage
if __name__ == "__main__":
    calc = Calculator()

    # Test add
    result = calc.execute(operation="add", a=5, b=3)
    print(f"5 + 3 = {result.output}")

    # Test divide
    result = calc.execute(operation="divide", a=10, b=2)
    print(f"10 / 2 = {result.output}")

    # Test divide by zero
    result = calc.execute(operation="divide", a=10, b=0)
    print(f"10 / 0: {result.error}")

    # Print tool schema
    print("\nAnthropic Schema:")
    import json

    print(json.dumps(calc.to_anthropic_schema(), indent=2))
