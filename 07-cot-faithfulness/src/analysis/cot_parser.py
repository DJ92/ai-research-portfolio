"""Parser for extracting and analyzing Chain-of-Thought reasoning steps.

Parses CoT responses into structured reasoning chains for analysis.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import re


@dataclass
class CoTStep:
    """A single step in a chain-of-thought reasoning process.

    Attributes:
        step_number: Sequential step number
        content: The reasoning content
        step_type: Type of reasoning (assumption, calculation, conclusion, etc.)
        confidence: Optional confidence level for this step
        dependencies: Steps this step depends on
    """

    step_number: int
    content: str
    step_type: str
    confidence: Optional[float] = None
    dependencies: Optional[List[int]] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class CoTChain:
    """A complete chain-of-thought reasoning chain.

    Attributes:
        steps: List of reasoning steps
        final_answer: The final answer derived from reasoning
        question: The original question
        raw_response: The raw model output
    """

    steps: List[CoTStep]
    final_answer: str
    question: str
    raw_response: str

    def get_step(self, step_number: int) -> Optional[CoTStep]:
        """Get step by number."""
        for step in self.steps:
            if step.step_number == step_number:
                return step
        return None

    def num_steps(self) -> int:
        """Get number of reasoning steps."""
        return len(self.steps)


class CoTParser:
    """Parser for extracting structured reasoning from CoT responses."""

    # Common patterns for identifying reasoning steps
    STEP_PATTERNS = [
        r"Step (\d+):?\s*(.*?)(?=Step \d+:|$)",
        r"(\d+)\.\s+(.*?)(?=\d+\.|$)",
        r"First,?\s+(.*?)(?=Second|Then|Next|Finally|$)",
        r"Then,?\s+(.*?)(?=Then|Next|Finally|Therefore|$)",
        r"Therefore,?\s+(.*?)(?=$)",
    ]

    # Patterns for identifying final answers
    ANSWER_PATTERNS = [
        r"(?:Final answer|Answer|Therefore|Thus|So):\s*(.*?)(?:\n|$)",
        r"(?:The answer is|equals|=)\s*(.*?)(?:\n|$)",
    ]

    # Reasoning step types
    STEP_TYPES = {
        "assumption": ["assume", "given", "suppose", "let"],
        "calculation": ["calculate", "compute", "=", "+", "-", "*", "/"],
        "inference": ["therefore", "thus", "hence", "so"],
        "verification": ["check", "verify", "confirm", "validate"],
        "conclusion": ["finally", "answer", "result"],
    }

    @staticmethod
    def parse(response: str, question: str) -> CoTChain:
        """Parse a CoT response into structured reasoning chain.

        Args:
            response: The model's CoT response
            question: The original question

        Returns:
            Parsed CoT chain
        """
        # Extract final answer
        final_answer = CoTParser._extract_final_answer(response)

        # Extract reasoning steps
        steps = CoTParser._extract_steps(response)

        return CoTChain(
            steps=steps,
            final_answer=final_answer,
            question=question,
            raw_response=response
        )

    @staticmethod
    def _extract_final_answer(response: str) -> str:
        """Extract final answer from response.

        Args:
            response: The response text

        Returns:
            Extracted final answer
        """
        # Try each answer pattern
        for pattern in CoTParser.ANSWER_PATTERNS:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()

        # If no pattern matches, use last line
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        return lines[-1] if lines else response.strip()

    @staticmethod
    def _extract_steps(response: str) -> List[CoTStep]:
        """Extract reasoning steps from response.

        Args:
            response: The response text

        Returns:
            List of parsed reasoning steps
        """
        steps = []

        # Try numbered step pattern first
        step_pattern = r"Step (\d+):?\s*(.*?)(?=Step \d+:|$)"
        matches = re.finditer(step_pattern, response, re.DOTALL | re.IGNORECASE)

        step_num = 0
        for match in matches:
            step_num = int(match.group(1))
            content = match.group(2).strip()

            if content:
                step_type = CoTParser._classify_step(content)
                steps.append(CoTStep(
                    step_number=step_num,
                    content=content,
                    step_type=step_type
                ))

        # If no explicit steps found, parse by sentences
        if not steps:
            sentences = [s.strip() for s in response.split(".") if len(s.strip()) > 10]
            for i, sentence in enumerate(sentences, 1):
                step_type = CoTParser._classify_step(sentence)
                steps.append(CoTStep(
                    step_number=i,
                    content=sentence,
                    step_type=step_type
                ))

        return steps

    @staticmethod
    def _classify_step(content: str) -> str:
        """Classify the type of a reasoning step.

        Args:
            content: Step content

        Returns:
            Step type classification
        """
        content_lower = content.lower()

        for step_type, keywords in CoTParser.STEP_TYPES.items():
            if any(keyword in content_lower for keyword in keywords):
                return step_type

        return "reasoning"  # Default type

    @staticmethod
    def count_reasoning_tokens(chain: CoTChain) -> int:
        """Count tokens in reasoning (excluding final answer).

        Args:
            chain: The CoT chain

        Returns:
            Approximate token count
        """
        reasoning_text = " ".join(step.content for step in chain.steps)
        # Rough approximation: 4 chars per token
        return len(reasoning_text) // 4
