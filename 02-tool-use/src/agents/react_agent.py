"""
ReAct agent implementation: Reason + Act pattern.

Based on "ReAct: Synergizing Reasoning and Acting in Language Models"
https://arxiv.org/abs/2210.03629
"""

from typing import List, Dict, Any, Optional
import json
from dataclasses import dataclass

from anthropic import Anthropic
import os
from dotenv import load_dotenv

from ..tools.base_tool import ToolRegistry, ToolResult

load_dotenv()


@dataclass
class AgentStep:
    """Single step in agent execution."""

    step_num: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    is_final: bool = False


class ReActAgent:
    """
    ReAct agent that interleaves reasoning (Thought) and acting (Action).

    Flow:
    1. Thought: Reasoning about what to do next
    2. Action: Tool to call with parameters
    3. Observation: Result from tool execution
    4. Repeat until task is complete
    """

    REACT_PROMPT = """You are a helpful AI assistant that can use tools to answer questions.

Available tools:
{tools_description}

For each step, you should:
1. Think about what you need to do (Thought)
2. Decide which tool to use and with what parameters (Action)
3. Observe the result (Observation)
4. Repeat until you can provide a final answer

Respond in this exact format:

Thought: [Your reasoning about what to do next]
Action: [tool_name]
Action Input: {{"param1": "value1", "param2": "value2"}}

OR if you have enough information:

Thought: [Your reasoning]
Final Answer: [Your complete answer to the user's question]

IMPORTANT:
- Only call ONE tool per response
- Action Input must be valid JSON
- Use exact tool names from the list above
- If a tool call fails, think about why and try a different approach

User Question: {question}
"""

    def __init__(
        self,
        model: str = "claude-sonnet-4.5",
        tools: Optional[ToolRegistry] = None,
        max_iterations: int = 10,
        verbose: bool = True,
    ):
        """
        Initialize ReAct agent.

        Args:
            model: Claude model to use
            tools: ToolRegistry with available tools
            max_iterations: Maximum reasoning steps
            verbose: Whether to print agent's reasoning
        """
        self.model = model
        self.tools = tools or ToolRegistry()
        self.max_iterations = max_iterations
        self.verbose = verbose

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        self.client = Anthropic(api_key=api_key)

    def _build_tools_description(self) -> str:
        """Build human-readable description of available tools."""
        descriptions = []
        for tool in self.tools.get_all():
            params_desc = []
            for param in tool.parameters:
                required = "required" if param.required else "optional"
                param_str = f"  - {param.name} ({param.type}, {required}): {param.description}"
                if param.enum:
                    param_str += f" [options: {', '.join(map(str, param.enum))}]"
                params_desc.append(param_str)

            tool_desc = f"""
{tool.name}: {tool.description}
Parameters:
{chr(10).join(params_desc)}
"""
            descriptions.append(tool_desc)

        return "\n".join(descriptions)

    def _parse_action(self, text: str) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Parse action and action input from agent response.

        Returns:
            Tuple of (action_name, action_input_dict)
        """
        lines = text.strip().split("\n")

        action_name = None
        action_input = None

        for i, line in enumerate(lines):
            if line.startswith("Action:"):
                action_name = line.replace("Action:", "").strip()

            if line.startswith("Action Input:"):
                # Get JSON from this line and potentially next lines
                json_start = i
                json_text = "\n".join(lines[json_start:])
                json_text = json_text.replace("Action Input:", "").strip()

                try:
                    action_input = json.loads(json_text)
                except json.JSONDecodeError:
                    # Try to extract just the JSON object
                    import re

                    json_match = re.search(r"\{.*\}", json_text, re.DOTALL)
                    if json_match:
                        try:
                            action_input = json.loads(json_match.group(0))
                        except json.JSONDecodeError:
                            pass

        return action_name, action_input

    def _is_final_answer(self, text: str) -> bool:
        """Check if response contains a final answer."""
        return "Final Answer:" in text

    def _extract_final_answer(self, text: str) -> str:
        """Extract final answer from response."""
        if "Final Answer:" in text:
            return text.split("Final Answer:")[1].strip()
        return text.strip()

    def run(self, question: str) -> Dict[str, Any]:
        """
        Run the ReAct agent on a question.

        Args:
            question: User's question

        Returns:
            Dictionary with final answer and execution trace
        """
        steps: List[AgentStep] = []
        conversation_history = []

        # Build initial prompt
        tools_desc = self._build_tools_description()
        initial_prompt = self.REACT_PROMPT.format(
            tools_description=tools_desc, question=question
        )

        conversation_history.append({"role": "user", "content": initial_prompt})

        for iteration in range(self.max_iterations):
            # Get agent's response
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=conversation_history,
            )

            response_text = response.content[0].text

            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Iteration {iteration + 1}")
                print(f"{'='*60}")
                print(response_text)

            # Check if final answer
            if self._is_final_answer(response_text):
                final_answer = self._extract_final_answer(response_text)
                steps.append(
                    AgentStep(
                        step_num=iteration + 1,
                        thought=response_text.split("Final Answer:")[0].strip(),
                        is_final=True,
                    )
                )

                return {
                    "answer": final_answer,
                    "steps": steps,
                    "iterations": iteration + 1,
                    "success": True,
                }

            # Parse action
            action_name, action_input = self._parse_action(response_text)

            if not action_name or not action_input:
                # Agent didn't provide valid action, ask it to continue
                conversation_history.append({"role": "assistant", "content": response_text})
                conversation_history.append(
                    {
                        "role": "user",
                        "content": "Please provide a valid Action and Action Input in the correct format, or provide a Final Answer if you have enough information.",
                    }
                )
                continue

            # Execute tool
            result: ToolResult = self.tools.execute(action_name, **action_input)

            observation = result.output if result.success else f"Error: {result.error}"

            if self.verbose:
                print(f"\nObservation: {observation}")

            # Record step
            steps.append(
                AgentStep(
                    step_num=iteration + 1,
                    thought=response_text.split("Action:")[0].replace("Thought:", "").strip(),
                    action=action_name,
                    action_input=action_input,
                    observation=str(observation),
                )
            )

            # Add to conversation
            conversation_history.append({"role": "assistant", "content": response_text})
            conversation_history.append(
                {"role": "user", "content": f"Observation: {observation}"}
            )

        # Max iterations reached
        return {
            "answer": "Failed to complete task within maximum iterations",
            "steps": steps,
            "iterations": self.max_iterations,
            "success": False,
        }


# Example usage
if __name__ == "__main__":
    from ..tools.calculator import Calculator
    from ..tools.base_tool import ToolRegistry

    # Setup
    registry = ToolRegistry()
    registry.register(Calculator())

    agent = ReActAgent(tools=registry, verbose=True)

    # Test
    result = agent.run("What is 15% of 240?")

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"Answer: {result['answer']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Success: {result['success']}")
