"""
Base tool interface and registry for function calling.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from pydantic import BaseModel, Field
from enum import Enum


class ToolResult(BaseModel):
    """Result from tool execution."""

    success: bool
    output: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ToolParameter(BaseModel):
    """Tool parameter specification."""

    name: str
    type: str  # "string", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    enum: Optional[List[Any]] = None
    default: Optional[Any] = None


class Tool(ABC):
    """
    Abstract base class for tools that LLMs can call.

    Tools must define:
    - name: Unique identifier
    - description: What the tool does
    - parameters: List of ToolParameter objects
    - execute: Implementation of the tool's functionality
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name (unique identifier)."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> List[ToolParameter]:
        """List of parameters this tool accepts."""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given parameters.

        Args:
            **kwargs: Tool parameters

        Returns:
            ToolResult with success status and output
        """
        pass

    def to_anthropic_schema(self) -> Dict[str, Any]:
        """
        Convert tool definition to Anthropic's tool schema format.

        Returns:
            Dictionary in Anthropic tool schema format
        """
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                properties[param.name]["enum"] = param.enum
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def to_openai_schema(self) -> Dict[str, Any]:
        """
        Convert tool definition to OpenAI's function schema format.

        Returns:
            Dictionary in OpenAI function schema format
        """
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                properties[param.name]["enum"] = param.enum
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def validate_params(self, **kwargs) -> tuple[bool, Optional[str]]:
        """
        Validate parameters before execution.

        Args:
            **kwargs: Parameters to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required parameters
        for param in self.parameters:
            if param.required and param.name not in kwargs:
                return False, f"Missing required parameter: {param.name}"

        # Check enum constraints
        for param in self.parameters:
            if param.enum and param.name in kwargs:
                if kwargs[param.name] not in param.enum:
                    return (
                        False,
                        f"Invalid value for {param.name}. Must be one of {param.enum}",
                    )

        return True, None


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._approval_required: set[str] = set()

    def register(self, tool: Tool, require_approval: bool = False):
        """
        Register a tool.

        Args:
            tool: Tool instance to register
            require_approval: Whether this tool requires user approval before execution
        """
        self._tools[tool.name] = tool
        if require_approval:
            self._approval_required.add(tool.name)

    def unregister(self, tool_name: str):
        """Remove a tool from the registry."""
        if tool_name in self._tools:
            del self._tools[tool_name]
            self._approval_required.discard(tool_name)

    def get(self, tool_name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(tool_name)

    def get_all(self) -> List[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def requires_approval(self, tool_name: str) -> bool:
        """Check if a tool requires user approval."""
        return tool_name in self._approval_required

    def to_anthropic_tools(self) -> List[Dict[str, Any]]:
        """Convert all tools to Anthropic schema format."""
        return [tool.to_anthropic_schema() for tool in self._tools.values()]

    def to_openai_tools(self) -> List[Dict[str, Any]]:
        """Convert all tools to OpenAI schema format."""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool parameters

        Returns:
            ToolResult
        """
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(
                success=False, output=None, error=f"Tool not found: {tool_name}"
            )

        # Validate parameters
        is_valid, error_msg = tool.validate_params(**kwargs)
        if not is_valid:
            return ToolResult(success=False, output=None, error=error_msg)

        # Execute tool
        try:
            return tool.execute(**kwargs)
        except Exception as e:
            return ToolResult(
                success=False, output=None, error=f"Tool execution error: {str(e)}"
            )


# Example usage
if __name__ == "__main__":
    from src.tools.calculator import Calculator

    # Create registry
    registry = ToolRegistry()

    # Register tools
    registry.register(Calculator())

    # Get Anthropic schema
    tools_schema = registry.to_anthropic_tools()
    print("Anthropic Schema:")
    print(tools_schema[0])

    # Execute tool
    result = registry.execute("calculator", operation="add", a=5, b=3)
    print(f"\nResult: {result.output}")
