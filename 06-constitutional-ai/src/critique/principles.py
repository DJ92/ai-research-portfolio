"""Constitutional principles for model alignment.

Based on Anthropic's Constitutional AI research.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ConstitutionalPrinciple:
    """A constitutional principle for evaluating and revising responses.

    Attributes:
        name: Principle identifier (e.g., "harmlessness", "helpfulness")
        critique_prompt: Template for critiquing responses
        revision_prompt: Template for revising based on critique
        description: Human-readable description of the principle
    """

    name: str
    critique_prompt: str
    revision_prompt: str
    description: str

    def format_critique(self, response: str, question: str) -> str:
        """Format critique prompt with question and response.

        Args:
            response: Model's initial response
            question: Original user question

        Returns:
            Formatted critique prompt
        """
        return self.critique_prompt.format(
            question=question,
            response=response
        )

    def format_revision(
        self,
        response: str,
        question: str,
        critique: str
    ) -> str:
        """Format revision prompt with question, response, and critique.

        Args:
            response: Model's initial response
            question: Original user question
            critique: Critique of the response

        Returns:
            Formatted revision prompt
        """
        return self.revision_prompt.format(
            question=question,
            response=response,
            critique=critique
        )


class PrincipleSet:
    """Collection of constitutional principles."""

    # Core principles based on Anthropic's Constitutional AI
    HARMLESSNESS = ConstitutionalPrinciple(
        name="harmlessness",
        critique_prompt="""Identify specific ways in which the assistant's response is harmful, unethical, racist, sexist, toxic, dangerous, or illegal.

Question: {question}
Response: {response}

Critique:""",
        revision_prompt="""Please rewrite the assistant's response to remove any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.

Question: {question}
Original Response: {response}
Critique: {critique}

Revised Response:""",
        description="Ensures responses are harmless and ethical"
    )

    HELPFULNESS = ConstitutionalPrinciple(
        name="helpfulness",
        critique_prompt="""Identify specific ways in which the assistant's response is not helpful, accurate, or informative.

Question: {question}
Response: {response}

Critique:""",
        revision_prompt="""Please rewrite the assistant's response to be more helpful, accurate, and informative while maintaining safety.

Question: {question}
Original Response: {response}
Critique: {critique}

Revised Response:""",
        description="Ensures responses are helpful and informative"
    )

    HONESTY = ConstitutionalPrinciple(
        name="honesty",
        critique_prompt="""Identify any ways in which the assistant's response is dishonest, misleading, or contains false information.

Question: {question}
Response: {response}

Critique:""",
        revision_prompt="""Please rewrite the assistant's response to be more honest and accurate, correcting any misleading or false information.

Question: {question}
Original Response: {response}
Critique: {critique}

Revised Response:""",
        description="Ensures responses are honest and truthful"
    )

    RESPECT = ConstitutionalPrinciple(
        name="respect",
        critique_prompt="""Identify ways in which the assistant's response is disrespectful, dismissive, or condescending.

Question: {question}
Response: {response}

Critique:""",
        revision_prompt="""Please rewrite the assistant's response to be more respectful and considerate.

Question: {question}
Original Response: {response}
Critique: {critique}

Revised Response:""",
        description="Ensures responses are respectful and considerate"
    )

    @classmethod
    def get_harmless_helpful(cls) -> List[ConstitutionalPrinciple]:
        """Get principles for harmless + helpful model.

        Returns:
            List containing harmlessness and helpfulness principles
        """
        return [cls.HARMLESSNESS, cls.HELPFULNESS]

    @classmethod
    def get_helpful_only(cls) -> List[ConstitutionalPrinciple]:
        """Get principles for helpful-only model.

        Returns:
            List containing only helpfulness principle
        """
        return [cls.HELPFULNESS]

    @classmethod
    def get_all(cls) -> List[ConstitutionalPrinciple]:
        """Get all constitutional principles.

        Returns:
            List of all principles
        """
        return [
            cls.HARMLESSNESS,
            cls.HELPFULNESS,
            cls.HONESTY,
            cls.RESPECT,
        ]

    @classmethod
    def get_by_name(cls, name: str) -> Optional[ConstitutionalPrinciple]:
        """Get principle by name.

        Args:
            name: Principle name

        Returns:
            Principle if found, None otherwise
        """
        principles = {p.name: p for p in cls.get_all()}
        return principles.get(name)
