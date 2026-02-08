"""Constitutional AI critique-revision loop implementation.

Based on "Constitutional AI: Harmlessness from AI Feedback" (Anthropic, 2022).
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .principles import ConstitutionalPrinciple


@dataclass
class CritiqueResult:
    """Result of critiquing a response against a principle.

    Attributes:
        principle_name: Name of the principle used
        critique: The critique text
        has_violation: Whether a violation was detected
        confidence: Confidence score (0-1) if available
        latency_ms: Time taken for critique generation
    """

    principle_name: str
    critique: str
    has_violation: bool
    confidence: Optional[float] = None
    latency_ms: Optional[float] = None


@dataclass
class RevisionResult:
    """Result of revising a response based on critique.

    Attributes:
        revised_response: The revised response text
        principle_name: Name of the principle used for revision
        original_response: The original response before revision
        critique: The critique that prompted the revision
        latency_ms: Time taken for revision generation
    """

    revised_response: str
    principle_name: str
    original_response: str
    critique: str
    latency_ms: Optional[float] = None


class ConstitutionalLoop:
    """Implements the Constitutional AI critique-revision loop.

    The loop works as follows:
    1. Generate initial response to user question
    2. For each constitutional principle:
        a. Critique the response against the principle
        b. If critique identifies violations, revise the response
    3. Return final revised response

    This implements the "Critique → Revision" pattern from Constitutional AI.
    """

    def __init__(
        self,
        llm_client: Any,
        principles: List[ConstitutionalPrinciple],
        max_iterations: int = 3,
        temperature: float = 0.0,
    ):
        """Initialize Constitutional Loop.

        Args:
            llm_client: LLM client with generate() method
            principles: List of constitutional principles to apply
            max_iterations: Maximum critique-revision iterations per principle
            temperature: Temperature for LLM generation (0 = deterministic)
        """
        self.llm = llm_client
        self.principles = principles
        self.max_iterations = max_iterations
        self.temperature = temperature

    def critique(
        self,
        response: str,
        question: str,
        principle: ConstitutionalPrinciple
    ) -> CritiqueResult:
        """Critique a response against a constitutional principle.

        Args:
            response: The response to critique
            question: The original question
            principle: The principle to apply

        Returns:
            CritiqueResult with critique text and violation detection
        """
        start_time = time.time()

        critique_prompt = principle.format_critique(response, question)

        critique_text = self.llm.generate(
            prompt=critique_prompt,
            temperature=self.temperature,
            max_tokens=500
        )

        latency_ms = (time.time() - start_time) * 1000

        # Simple heuristic: if critique is short or says "no issues", no violation
        has_violation = self._detect_violation(critique_text)

        return CritiqueResult(
            principle_name=principle.name,
            critique=critique_text.strip(),
            has_violation=has_violation,
            latency_ms=latency_ms
        )

    def revise(
        self,
        response: str,
        question: str,
        critique: str,
        principle: ConstitutionalPrinciple
    ) -> RevisionResult:
        """Revise a response based on critique.

        Args:
            response: The response to revise
            question: The original question
            critique: The critique to address
            principle: The principle to apply

        Returns:
            RevisionResult with revised response
        """
        start_time = time.time()

        revision_prompt = principle.format_revision(response, question, critique)

        revised_response = self.llm.generate(
            prompt=revision_prompt,
            temperature=self.temperature,
            max_tokens=1000
        )

        latency_ms = (time.time() - start_time) * 1000

        return RevisionResult(
            revised_response=revised_response.strip(),
            principle_name=principle.name,
            original_response=response,
            critique=critique,
            latency_ms=latency_ms
        )

    def apply_principle(
        self,
        response: str,
        question: str,
        principle: ConstitutionalPrinciple
    ) -> Dict[str, Any]:
        """Apply a single principle through critique-revision loop.

        Args:
            response: Initial response
            question: Original question
            principle: Principle to apply

        Returns:
            Dict with final response and iteration history
        """
        current_response = response
        iterations = []

        for iteration in range(self.max_iterations):
            # Critique current response
            critique_result = self.critique(
                current_response,
                question,
                principle
            )

            iterations.append({
                "iteration": iteration,
                "critique": critique_result,
                "response_before": current_response
            })

            # If no violation, we're done
            if not critique_result.has_violation:
                break

            # Revise response
            revision_result = self.revise(
                current_response,
                question,
                critique_result.critique,
                principle
            )

            iterations[-1]["revision"] = revision_result
            current_response = revision_result.revised_response

        return {
            "final_response": current_response,
            "iterations": iterations,
            "total_iterations": len(iterations),
            "principle": principle.name
        }

    def run(
        self,
        question: str,
        initial_response: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run full constitutional loop on a question.

        Args:
            question: User question
            initial_response: Optional initial response (generated if not provided)

        Returns:
            Dict with final response and full history
        """
        start_time = time.time()

        # Generate initial response if not provided
        if initial_response is None:
            initial_response = self.llm.generate(
                prompt=question,
                temperature=self.temperature,
                max_tokens=1000
            )

        current_response = initial_response
        principle_results = []

        # Apply each principle sequentially
        for principle in self.principles:
            result = self.apply_principle(
                current_response,
                question,
                principle
            )
            principle_results.append(result)
            current_response = result["final_response"]

        total_latency_ms = (time.time() - start_time) * 1000

        return {
            "question": question,
            "initial_response": initial_response,
            "final_response": current_response,
            "principle_results": principle_results,
            "total_latency_ms": total_latency_ms,
            "num_principles": len(self.principles)
        }

    def _detect_violation(self, critique: str) -> bool:
        """Detect if critique indicates a violation.

        Args:
            critique: Critique text

        Returns:
            True if violation detected, False otherwise
        """
        critique_lower = critique.lower().strip()

        # Heuristics for "no violation"
        no_violation_phrases = [
            "no issues",
            "no violations",
            "appropriate",
            "does not contain",
            "not harmful",
            "not toxic",
            "not racist",
            "not sexist",
            "is helpful",
            "is accurate",
            "is honest",
            "is respectful"
        ]

        for phrase in no_violation_phrases:
            if phrase in critique_lower:
                return False

        # If critique is very short (< 20 chars), likely no violation
        if len(critique.strip()) < 20:
            return False

        # Otherwise, assume there's a violation to address
        return True
