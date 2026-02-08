"""
Unified interface for LLM API clients (OpenAI, Anthropic).
Provides consistent interface for generation and cost tracking.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import time
import os

from anthropic import Anthropic
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMResponse:
    """Standardized response from any LLM."""

    text: str
    model: str
    latency_ms: float
    tokens_input: int
    tokens_output: int
    cost_usd: float
    metadata: Dict[str, Any]


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    # Pricing per 1M tokens (input, output)
    PRICING = {
        "gpt-4": (0.03, 0.06),
        "gpt-4-turbo": (0.01, 0.03),
        "gpt-3.5-turbo": (0.0005, 0.0015),
        "claude-opus-4.6": (0.015, 0.075),
        "claude-sonnet-4.5": (0.003, 0.015),
        "claude-haiku-4.5": (0.0008, 0.004),
    }

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass

    def calculate_cost(
        self,
        model: str,
        tokens_input: int,
        tokens_output: int
    ) -> float:
        """Calculate API cost in USD."""
        if model not in self.PRICING:
            return 0.0

        price_input, price_output = self.PRICING[model]
        cost = (tokens_input * price_input + tokens_output * price_output) / 1_000_000
        return cost


class AnthropicClient(LLMClient):
    """Client for Anthropic's Claude API."""

    def __init__(self, model: str = "claude-sonnet-4.5"):
        self.model = model
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        self.client = Anthropic(api_key=api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Claude."""
        start_time = time.time()

        messages = [{"role": "user", "content": prompt}]

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt or "",
            messages=messages,
            **kwargs
        )

        latency_ms = (time.time() - start_time) * 1000

        # Extract response data
        text = response.content[0].text
        tokens_input = response.usage.input_tokens
        tokens_output = response.usage.output_tokens

        cost = self.calculate_cost(self.model, tokens_input, tokens_output)

        return LLMResponse(
            text=text,
            model=self.model,
            latency_ms=latency_ms,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_usd=cost,
            metadata={"stop_reason": response.stop_reason}
        )


class OpenAIClient(LLMClient):
    """Client for OpenAI's GPT API."""

    def __init__(self, model: str = "gpt-4"):
        self.model = model
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        self.client = OpenAI(api_key=api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs
    ) -> LLMResponse:
        """Generate response using GPT."""
        start_time = time.time()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

        latency_ms = (time.time() - start_time) * 1000

        # Extract response data
        text = response.choices[0].message.content
        tokens_input = response.usage.prompt_tokens
        tokens_output = response.usage.completion_tokens

        cost = self.calculate_cost(self.model, tokens_input, tokens_output)

        return LLMResponse(
            text=text,
            model=self.model,
            latency_ms=latency_ms,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_usd=cost,
            metadata={"finish_reason": response.choices[0].finish_reason}
        )


def get_llm_client(model: str) -> LLMClient:
    """Factory function to get the appropriate LLM client."""
    if model.startswith("claude"):
        return AnthropicClient(model)
    elif model.startswith("gpt"):
        return OpenAIClient(model)
    else:
        raise ValueError(f"Unknown model: {model}")


# Example usage
if __name__ == "__main__":
    # Test with Claude
    claude = get_llm_client("claude-sonnet-4.5")
    response = claude.generate(
        prompt="What is the capital of France?",
        system_prompt="You are a helpful geography expert.",
        max_tokens=100
    )

    print(f"Response: {response.text}")
    print(f"Latency: {response.latency_ms:.1f}ms")
    print(f"Cost: ${response.cost_usd:.5f}")
    print(f"Tokens: {response.tokens_input} in, {response.tokens_output} out")
