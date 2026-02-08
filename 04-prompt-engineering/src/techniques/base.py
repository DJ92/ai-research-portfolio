"""
Base classes for prompting techniques.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

from anthropic import Anthropic
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class PromptResult:
    """Result from prompt execution."""

    output: str
    prompt: str
    model: str
    latency_ms: float
    tokens_input: int
    tokens_output: int
    cost_usd: float
    metadata: Dict[str, Any]


class PromptingTechnique(ABC):
    """
    Abstract base class for prompting techniques.

    All techniques must implement the prompt() method.
    """

    def __init__(self, model: str = "claude-sonnet-4.5", temperature: float = 0.0):
        """
        Initialize prompting technique.

        Args:
            model: LLM model to use
            temperature: Sampling temperature (0=deterministic, 1=creative)
        """
        self.model = model
        self.temperature = temperature

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        self.client = Anthropic(api_key=api_key)

    @abstractmethod
    def build_prompt(self, input_text: str, **kwargs) -> str:
        """
        Build the prompt for this technique.

        Args:
            input_text: The input/query text
            **kwargs: Additional arguments specific to the technique

        Returns:
            Formatted prompt string
        """
        pass

    def execute(
        self,
        input_text: str,
        max_tokens: int = 1024,
        **kwargs
    ) -> PromptResult:
        """
        Execute the prompting technique.

        Args:
            input_text: Input text/query
            max_tokens: Maximum tokens to generate
            **kwargs: Technique-specific arguments

        Returns:
            PromptResult with output and metadata
        """
        start_time = time.time()

        # Build prompt
        prompt = self.build_prompt(input_text, **kwargs)

        # Call LLM
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        latency_ms = (time.time() - start_time) * 1000

        # Extract response
        output = response.content[0].text
        tokens_input = response.usage.input_tokens
        tokens_output = response.usage.output_tokens

        # Calculate cost (example pricing for Sonnet 4.5)
        cost = (tokens_input * 0.003 + tokens_output * 0.015) / 1_000_000

        return PromptResult(
            output=output,
            prompt=prompt,
            model=self.model,
            latency_ms=latency_ms,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_usd=cost,
            metadata={
                "temperature": self.temperature,
                "stop_reason": response.stop_reason,
            },
        )


class ZeroShot(PromptingTechnique):
    """Zero-shot prompting: direct instruction without examples."""

    def __init__(
        self,
        instruction: str,
        model: str = "claude-sonnet-4.5",
        temperature: float = 0.0,
    ):
        """
        Initialize zero-shot prompting.

        Args:
            instruction: Task instruction
            model: LLM model
            temperature: Sampling temperature
        """
        super().__init__(model, temperature)
        self.instruction = instruction

    def build_prompt(self, input_text: str, **kwargs) -> str:
        """Build zero-shot prompt."""
        return f"""{self.instruction}

Input: {input_text}

Output:"""


class FewShot(PromptingTechnique):
    """Few-shot prompting: provide examples to guide the model."""

    def __init__(
        self,
        instruction: str,
        examples: List[Dict[str, str]],
        model: str = "claude-sonnet-4.5",
        temperature: float = 0.0,
    ):
        """
        Initialize few-shot prompting.

        Args:
            instruction: Task instruction
            examples: List of {"input": ..., "output": ...} examples
            model: LLM model
            temperature: Sampling temperature
        """
        super().__init__(model, temperature)
        self.instruction = instruction
        self.examples = examples

    def build_prompt(self, input_text: str, **kwargs) -> str:
        """Build few-shot prompt with examples."""
        prompt = f"{self.instruction}\n\n"

        # Add examples
        for i, example in enumerate(self.examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Input: {example['input']}\n"
            prompt += f"Output: {example['output']}\n\n"

        # Add current input
        prompt += f"Input: {input_text}\nOutput:"

        return prompt


class ChainOfThought(PromptingTechnique):
    """Chain-of-Thought: encourage step-by-step reasoning."""

    def __init__(
        self,
        instruction: str,
        model: str = "claude-sonnet-4.5",
        temperature: float = 0.0,
        show_reasoning: bool = True,
    ):
        """
        Initialize CoT prompting.

        Args:
            instruction: Task instruction
            model: LLM model
            temperature: Sampling temperature
            show_reasoning: Whether to include reasoning in output
        """
        super().__init__(model, temperature)
        self.instruction = instruction
        self.show_reasoning = show_reasoning

    def build_prompt(self, input_text: str, **kwargs) -> str:
        """Build CoT prompt."""
        prompt = f"""{self.instruction}

Input: {input_text}

Let's solve this step by step:
1."""
        return prompt

    def execute(self, input_text: str, max_tokens: int = 2048, **kwargs) -> PromptResult:
        """Execute with longer max_tokens for reasoning."""
        result = super().execute(input_text, max_tokens=max_tokens, **kwargs)

        if not self.show_reasoning:
            # Extract just the final answer if present
            lines = result.output.split("\n")
            for line in reversed(lines):
                if line.strip() and not line.strip().startswith(("1.", "2.", "3.")):
                    result.output = line.strip()
                    break

        return result


# Example usage
if __name__ == "__main__":
    # Test Zero-Shot
    print("=== Zero-Shot ===")
    zero_shot = ZeroShot(
        instruction="Classify the sentiment of this review as positive or negative."
    )
    result = zero_shot.execute("This product exceeded my expectations!")
    print(f"Output: {result.output}")
    print(f"Cost: ${result.cost_usd:.5f}")

    # Test Few-Shot
    print("\n=== Few-Shot ===")
    few_shot = FewShot(
        instruction="Classify sentiment as positive or negative.",
        examples=[
            {"input": "Great product!", "output": "positive"},
            {"input": "Terrible quality.", "output": "negative"},
            {"input": "Decent for the price.", "output": "positive"},
        ],
    )
    result = few_shot.execute("Not bad at all")
    print(f"Output: {result.output}")

    # Test Chain-of-Thought
    print("\n=== Chain-of-Thought ===")
    cot = ChainOfThought(
        instruction="Solve this math problem."
    )
    result = cot.execute("If a shirt costs $25 and is on sale for 20% off, what is the final price?")
    print(f"Output: {result.output}")
    print(f"Tokens: {result.tokens_input} in, {result.tokens_output} out")
