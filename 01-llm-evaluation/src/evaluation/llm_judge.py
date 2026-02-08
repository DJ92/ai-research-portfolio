"""
LLM-as-a-Judge evaluation framework.
Uses Claude or GPT-4 to evaluate LLM outputs on various criteria.
"""

from typing import List, Dict, Optional, Literal
from dataclasses import dataclass
import json
import re

from ..models.llm_client import get_llm_client, LLMResponse


@dataclass
class JudgeResult:
    """Result from LLM judge evaluation."""

    criterion: str
    score: float  # 1-5 scale
    reasoning: str
    model: str
    cost_usd: float


class LLMJudge:
    """
    Evaluate LLM outputs using another LLM as a judge.

    Based on "Judging LLM-as-a-Judge" (Zheng et al., 2023).
    """

    CRITERIA_PROMPTS = {
        "correctness": """
Evaluate the CORRECTNESS of the response.
- Is the information factually accurate?
- Does it properly answer the question?
- Are there any errors or hallucinations?

Score 1-5 where:
1 = Completely incorrect or nonsensical
2 = Mostly incorrect with some accurate elements
3 = Partially correct but with notable errors
4 = Mostly correct with minor inaccuracies
5 = Completely correct and accurate
""",
        "completeness": """
Evaluate the COMPLETENESS of the response.
- Does it address all parts of the question?
- Is any important information missing?
- Is the depth of explanation appropriate?

Score 1-5 where:
1 = Missing most key information
2 = Addresses question superficially, major gaps
3 = Covers main points but missing some details
4 = Comprehensive with only minor omissions
5 = Fully complete and thorough
""",
        "coherence": """
Evaluate the COHERENCE of the response.
- Is the logic clear and easy to follow?
- Do ideas flow naturally?
- Are there contradictions or confusing parts?

Score 1-5 where:
1 = Incoherent or contradictory
2 = Difficult to follow with logical gaps
3 = Generally coherent but some unclear parts
4 = Clear and logical with minor issues
5 = Perfectly coherent and well-structured
""",
        "conciseness": """
Evaluate the CONCISENESS of the response.
- Is the response appropriately brief?
- Is there unnecessary repetition?
- Is it verbose or overly terse?

Score 1-5 where:
1 = Extremely verbose or way too brief
2 = Significant redundancy or insufficient detail
3 = Acceptable length but could be improved
4 = Well-balanced with minimal excess
5 = Perfectly concise and to the point
""",
        "helpfulness": """
Evaluate the HELPFULNESS of the response.
- Is it practical and actionable?
- Would it actually help the user?
- Is the tone appropriate?

Score 1-5 where:
1 = Not helpful at all
2 = Minimally helpful
3 = Somewhat helpful
4 = Very helpful
5 = Extremely helpful and valuable
""",
    }

    def __init__(
        self,
        model: str = "claude-sonnet-4.5",
        temperature: float = 0.0
    ):
        """
        Initialize LLM judge.

        Args:
            model: Model to use as judge (claude-sonnet-4.5, gpt-4, etc.)
            temperature: Temperature for judge model (0 for deterministic)
        """
        self.model = model
        self.temperature = temperature
        self.client = get_llm_client(model)

    def _build_evaluation_prompt(
        self,
        question: str,
        response: str,
        criterion: str,
        reference: Optional[str] = None
    ) -> str:
        """Build the evaluation prompt for the judge."""
        if criterion not in self.CRITERIA_PROMPTS:
            raise ValueError(f"Unknown criterion: {criterion}")

        prompt = f"""You are an expert evaluator assessing the quality of AI-generated responses.

QUESTION:
{question}

RESPONSE TO EVALUATE:
{response}
"""

        if reference:
            prompt += f"""
REFERENCE ANSWER (for comparison):
{reference}
"""

        prompt += f"""
{self.CRITERIA_PROMPTS[criterion]}

Provide your evaluation in this JSON format:
{{
  "score": <1-5>,
  "reasoning": "<brief explanation of your score>"
}}
"""

        return prompt

    def evaluate_single(
        self,
        question: str,
        response: str,
        criterion: str,
        reference: Optional[str] = None
    ) -> JudgeResult:
        """
        Evaluate a response on a single criterion.

        Args:
            question: The original question/prompt
            response: The LLM's response to evaluate
            criterion: Evaluation criterion (correctness, completeness, etc.)
            reference: Optional reference answer for comparison

        Returns:
            JudgeResult with score and reasoning
        """
        prompt = self._build_evaluation_prompt(
            question, response, criterion, reference
        )

        llm_response: LLMResponse = self.client.generate(
            prompt=prompt,
            system_prompt="You are a precise evaluator. Always respond with valid JSON.",
            max_tokens=500,
            temperature=self.temperature
        )

        # Parse JSON response
        try:
            # Extract JSON from response (handle markdown code blocks)
            text = llm_response.text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            result = json.loads(text)
            score = float(result["score"])
            reasoning = result["reasoning"]

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback: try to extract score using regex
            match = re.search(r'"score":\s*(\d+)', llm_response.text)
            if match:
                score = float(match.group(1))
                reasoning = f"Could not parse full response: {llm_response.text[:100]}"
            else:
                score = 0.0
                reasoning = f"Failed to parse: {str(e)}"

        return JudgeResult(
            criterion=criterion,
            score=score,
            reasoning=reasoning,
            model=self.model,
            cost_usd=llm_response.cost_usd
        )

    def evaluate(
        self,
        question: str,
        response: str,
        criteria: Optional[List[str]] = None,
        reference: Optional[str] = None
    ) -> Dict[str, JudgeResult]:
        """
        Evaluate a response on multiple criteria.

        Args:
            question: The original question/prompt
            response: The LLM's response to evaluate
            criteria: List of criteria to evaluate. If None, use all.
            reference: Optional reference answer

        Returns:
            Dictionary mapping criterion names to JudgeResults
        """
        if criteria is None:
            criteria = list(self.CRITERIA_PROMPTS.keys())

        results = {}
        for criterion in criteria:
            results[criterion] = self.evaluate_single(
                question, response, criterion, reference
            )

        return results

    def get_average_score(
        self,
        results: Dict[str, JudgeResult]
    ) -> float:
        """Compute average score across all criteria."""
        if not results:
            return 0.0
        return sum(r.score for r in results.values()) / len(results)

    def get_total_cost(
        self,
        results: Dict[str, JudgeResult]
    ) -> float:
        """Compute total evaluation cost."""
        return sum(r.cost_usd for r in results.values())


# Example usage
if __name__ == "__main__":
    judge = LLMJudge(model="claude-sonnet-4.5")

    question = "What is the capital of France?"
    response = "Paris is the capital of France. It's a beautiful city known for the Eiffel Tower."

    results = judge.evaluate(
        question=question,
        response=response,
        criteria=["correctness", "completeness", "conciseness"]
    )

    print("\nLLM-as-Judge Evaluation:")
    for criterion, result in results.items():
        print(f"\n{criterion.upper()}: {result.score}/5")
        print(f"Reasoning: {result.reasoning}")

    print(f"\nAverage Score: {judge.get_average_score(results):.2f}/5")
    print(f"Total Cost: ${judge.get_total_cost(results):.5f}")
