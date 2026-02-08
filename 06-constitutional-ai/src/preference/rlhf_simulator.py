"""RLHF simulation using constitutional AI feedback.

Simulates the RLHF process by generating preference data from
constitutional AI critiques instead of human feedback.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..critique.constitutional_loop import ConstitutionalLoop
from ..critique.principles import ConstitutionalPrinciple
from .preference_model import PreferenceDataset, PreferenceExample


@dataclass
class FeedbackResult:
    """Result of generating AI feedback for a response.

    Attributes:
        prompt: Original prompt
        initial_response: Response before constitutional revision
        revised_response: Response after constitutional revision
        improved: Whether revision improved the response
        principles_applied: Principles used for revision
        feedback_summary: Summary of changes made
    """

    prompt: str
    initial_response: str
    revised_response: str
    improved: bool
    principles_applied: List[str]
    feedback_summary: str


class RLHFSimulator:
    """Simulates RLHF using constitutional AI feedback.

    Instead of human feedback, this uses constitutional principles
    to generate preference pairs, simulating the RLHF data collection
    process with AI feedback (as in Constitutional AI).
    """

    def __init__(
        self,
        constitutional_loop: ConstitutionalLoop,
        min_improvement_threshold: float = 0.1
    ):
        """Initialize RLHF simulator.

        Args:
            constitutional_loop: Constitutional loop for generating feedback
            min_improvement_threshold: Minimum change threshold to count as improvement
        """
        self.constitutional_loop = constitutional_loop
        self.min_improvement_threshold = min_improvement_threshold

    def generate_feedback(
        self,
        prompts: List[str],
        initial_responses: Optional[List[str]] = None
    ) -> List[FeedbackResult]:
        """Generate AI feedback for a list of prompts.

        Args:
            prompts: List of prompts to generate feedback for
            initial_responses: Optional pre-generated initial responses

        Returns:
            List of feedback results
        """
        if initial_responses is None:
            initial_responses = [None] * len(prompts)

        results = []

        for prompt, initial_response in zip(prompts, initial_responses):
            # Run constitutional loop
            loop_result = self.constitutional_loop.run(
                question=prompt,
                initial_response=initial_response
            )

            # Determine if revision improved the response
            improved = self._check_improvement(loop_result)

            # Extract principles applied
            principles_applied = [
                pr["principle"]
                for pr in loop_result["principle_results"]
            ]

            # Generate feedback summary
            feedback_summary = self._generate_feedback_summary(loop_result)

            result = FeedbackResult(
                prompt=prompt,
                initial_response=loop_result["initial_response"],
                revised_response=loop_result["final_response"],
                improved=improved,
                principles_applied=principles_applied,
                feedback_summary=feedback_summary
            )

            results.append(result)

        return results

    def build_preference_dataset(
        self,
        feedback_results: List[FeedbackResult]
    ) -> PreferenceDataset:
        """Build preference dataset from feedback results.

        Args:
            feedback_results: List of feedback results

        Returns:
            PreferenceDataset with preference pairs
        """
        dataset = PreferenceDataset()

        for result in feedback_results:
            # Only add examples where revision improved the response
            if result.improved:
                example = PreferenceExample(
                    prompt=result.prompt,
                    chosen=result.revised_response,
                    rejected=result.initial_response,
                    reason=result.feedback_summary,
                    principle=", ".join(result.principles_applied)
                )
                dataset.add(example)

        return dataset

    def simulate_rlhf_iteration(
        self,
        train_prompts: List[str],
        val_prompts: List[str]
    ) -> Dict[str, Any]:
        """Simulate one iteration of RLHF.

        Args:
            train_prompts: Prompts for training data generation
            val_prompts: Prompts for validation

        Returns:
            Dict with training and validation datasets and metrics
        """
        # Generate training feedback
        train_feedback = self.generate_feedback(train_prompts)
        train_dataset = self.build_preference_dataset(train_feedback)

        # Generate validation feedback
        val_feedback = self.generate_feedback(val_prompts)
        val_dataset = self.build_preference_dataset(val_feedback)

        # Calculate statistics
        train_improvement_rate = sum(
            1 for f in train_feedback if f.improved
        ) / len(train_feedback) if train_feedback else 0.0

        val_improvement_rate = sum(
            1 for f in val_feedback if f.improved
        ) / len(val_feedback) if val_feedback else 0.0

        return {
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "train_improvement_rate": train_improvement_rate,
            "val_improvement_rate": val_improvement_rate,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset)
        }

    def _check_improvement(self, loop_result: Dict[str, Any]) -> bool:
        """Check if constitutional loop improved the response.

        Args:
            loop_result: Result from constitutional loop

        Returns:
            True if response was improved
        """
        initial = loop_result["initial_response"]
        final = loop_result["final_response"]

        # If responses are identical, no improvement
        if initial == final:
            return False

        # Check if any principles detected violations and made revisions
        for pr in loop_result["principle_results"]:
            if pr["total_iterations"] > 0:
                # At least one iteration means violation was found and addressed
                return True

        return False

    def _generate_feedback_summary(self, loop_result: Dict[str, Any]) -> str:
        """Generate summary of constitutional feedback.

        Args:
            loop_result: Result from constitutional loop

        Returns:
            Summary of changes
        """
        summaries = []

        for pr in loop_result["principle_results"]:
            principle = pr["principle"]
            iterations = pr["iterations"]

            if not iterations:
                continue

            # Get critiques from all iterations
            critiques = [
                it["critique"].critique
                for it in iterations
                if it.get("critique")
            ]

            if critiques:
                summary = f"{principle}: {critiques[0][:100]}..."
                summaries.append(summary)

        if not summaries:
            return "No significant issues found"

        return "; ".join(summaries)
