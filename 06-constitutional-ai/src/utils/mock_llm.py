"""Mock LLM client for testing without API calls."""

from typing import Optional, Dict, Any


class MockLLMClient:
    """Mock LLM client that returns predefined responses.

    Used for testing constitutional AI without making actual API calls.
    """

    def __init__(self, response_mode: str = "helpful"):
        """Initialize mock client.

        Args:
            response_mode: Mode for generating responses
                - "helpful": Generate helpful but potentially unsafe responses
                - "safe": Generate safe responses
                - "harmful": Generate responses with violations (for testing)
        """
        self.response_mode = response_mode
        self.call_count = 0

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1000
    ) -> str:
        """Generate mock response.

        Args:
            prompt: Input prompt
            temperature: Temperature parameter (ignored)
            max_tokens: Max tokens (ignored)

        Returns:
            Mock response based on response_mode
        """
        self.call_count += 1

        # Detect if this is a critique request
        if "critique" in prompt.lower() or "identify" in prompt.lower():
            return self._generate_critique(prompt)

        # Detect if this is a revision request
        if "rewrite" in prompt.lower() or "revise" in prompt.lower():
            return self._generate_revision(prompt)

        # Otherwise generate initial response
        return self._generate_initial_response(prompt)

    def _generate_initial_response(self, prompt: str) -> str:
        """Generate initial response to user prompt."""
        if self.response_mode == "harmful":
            # Generate response with violations (for testing)
            return "You should definitely lie to get what you want. It's the most effective strategy and everyone does it anyway."

        elif self.response_mode == "helpful":
            # Helpful but might have minor issues
            return "Here's what you should do: First, consider all your options carefully. Then, make a decision based on what will benefit you most. If needed, be flexible with the truth to achieve your goals."

        else:  # safe
            # Already aligned response
            return "I'd recommend carefully considering your options and making an ethical decision that aligns with your values while respecting others."

    def _generate_critique(self, prompt: str) -> str:
        """Generate critique based on prompt content."""
        if "harmful" in prompt.lower() or "toxic" in prompt.lower():
            # Critiquing for harmfulness
            if self.response_mode == "harmful":
                return "The response explicitly encourages dishonesty and lying, which is harmful and unethical. It suggests that lying is acceptable and effective, which could lead to harmful outcomes."
            elif self.response_mode == "helpful":
                return "The response suggests being 'flexible with the truth,' which is a euphemism for lying or being dishonest. This could be harmful as it promotes unethical behavior."
            else:
                return "The response is appropriate and does not contain harmful content."

        elif "helpful" in prompt.lower() or "informative" in prompt.lower():
            # Critiquing for helpfulness
            if self.response_mode == "harmful":
                return "The response is not helpful as it provides unethical advice rather than genuinely useful guidance."
            else:
                return "The response is reasonably helpful and provides relevant guidance."

        elif "honest" in prompt.lower() or "misleading" in prompt.lower():
            # Critiquing for honesty
            if self.response_mode == "harmful" or self.response_mode == "helpful":
                return "The response contains suggestions to be dishonest or misleading, which violates honesty principles."
            else:
                return "The response is honest and does not contain misleading information."

        else:
            # Default critique
            return "No significant issues found."

    def _generate_revision(self, prompt: str) -> str:
        """Generate revised response."""
        # Extract context from revision prompt
        if "dishonest" in prompt.lower() or "harmful" in prompt.lower():
            return "I'd recommend carefully considering your options and making an ethical decision. It's important to be honest and transparent in your approach. Consider what aligns with your values and respects others."

        elif "helpful" in prompt.lower():
            return "Here's a comprehensive approach: First, clearly define your goals. Second, evaluate all options objectively. Third, choose the path that aligns with your ethical standards. Finally, implement your decision thoughtfully while remaining open to feedback."

        else:
            return "I'd recommend taking a balanced and ethical approach to this situation, considering both your needs and the impact on others."

    def reset_count(self) -> None:
        """Reset call counter."""
        self.call_count = 0
