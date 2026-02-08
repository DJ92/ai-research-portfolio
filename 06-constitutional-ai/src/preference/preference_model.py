"""Preference model for learning from comparative feedback.

Implements a simple preference learning approach inspired by RLHF.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json


@dataclass
class PreferenceExample:
    """A preference example for training.

    Attributes:
        prompt: The input prompt
        chosen: The preferred response
        rejected: The less preferred response
        reason: Optional explanation for preference
        principle: Optional principle that guided preference
    """

    prompt: str
    chosen: str
    rejected: str
    reason: Optional[str] = None
    principle: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "reason": self.reason,
            "principle": self.principle
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreferenceExample":
        """Create from dictionary."""
        return cls(**data)


class PreferenceDataset:
    """Dataset of preference examples."""

    def __init__(self, examples: Optional[List[PreferenceExample]] = None):
        """Initialize dataset.

        Args:
            examples: Optional initial examples
        """
        self.examples = examples or []

    def add(self, example: PreferenceExample) -> None:
        """Add a preference example.

        Args:
            example: Preference example to add
        """
        self.examples.append(example)

    def add_from_constitutional_result(
        self,
        question: str,
        initial_response: str,
        final_response: str,
        principle: str,
        reason: str
    ) -> None:
        """Add preference from constitutional AI result.

        Args:
            question: Original question
            initial_response: Response before constitutional revision
            final_response: Response after constitutional revision
            principle: Principle that guided the revision
            reason: Explanation of why final is better
        """
        example = PreferenceExample(
            prompt=question,
            chosen=final_response,
            rejected=initial_response,
            reason=reason,
            principle=principle
        )
        self.add(example)

    def save(self, filepath: str) -> None:
        """Save dataset to JSON file.

        Args:
            filepath: Path to save dataset
        """
        data = [ex.to_dict() for ex in self.examples]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "PreferenceDataset":
        """Load dataset from JSON file.

        Args:
            filepath: Path to load dataset from

        Returns:
            Loaded dataset
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        examples = [PreferenceExample.from_dict(ex) for ex in data]
        return cls(examples=examples)

    def __len__(self) -> int:
        """Get number of examples."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> PreferenceExample:
        """Get example by index."""
        return self.examples[idx]

    def split(
        self,
        train_ratio: float = 0.8
    ) -> Tuple["PreferenceDataset", "PreferenceDataset"]:
        """Split dataset into train and validation sets.

        Args:
            train_ratio: Ratio of examples for training

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        n_train = int(len(self.examples) * train_ratio)
        train_examples = self.examples[:n_train]
        val_examples = self.examples[n_train:]

        return (
            PreferenceDataset(examples=train_examples),
            PreferenceDataset(examples=val_examples)
        )


class PreferenceModel:
    """Simple preference model for ranking responses.

    This is a lightweight implementation that uses an LLM to predict
    preferences based on examples. In production, this would be a
    fine-tuned reward model.
    """

    def __init__(self, llm_client: Any, temperature: float = 0.0):
        """Initialize preference model.

        Args:
            llm_client: LLM client with generate() method
            temperature: Temperature for generation
        """
        self.llm = llm_client
        self.temperature = temperature
        self.training_examples: List[PreferenceExample] = []

    def add_training_data(self, dataset: PreferenceDataset) -> None:
        """Add training data from a dataset.

        Args:
            dataset: Preference dataset
        """
        self.training_examples.extend(dataset.examples)

    def predict_preference(
        self,
        prompt: str,
        response_a: str,
        response_b: str
    ) -> Dict[str, Any]:
        """Predict which response is preferred.

        Args:
            prompt: The input prompt
            response_a: First response option
            response_b: Second response option

        Returns:
            Dict with preference (A or B) and reasoning
        """
        # Build few-shot prompt from training examples
        few_shot_examples = ""
        if self.training_examples:
            # Use up to 3 examples
            for i, ex in enumerate(self.training_examples[:3], 1):
                few_shot_examples += f"""
Example {i}:
Prompt: {ex.prompt}
Response A: {ex.chosen}
Response B: {ex.rejected}
Preference: A (chosen)
Reason: {ex.reason or 'More aligned with constitutional principles'}

"""

        prediction_prompt = f"""{few_shot_examples}
Now evaluate the following:

Prompt: {prompt}
Response A: {response_a}
Response B: {response_b}

Which response is better (A or B)? Provide your reasoning.

Preference:"""

        prediction = self.llm.generate(
            prompt=prediction_prompt,
            temperature=self.temperature,
            max_tokens=300
        )

        # Parse prediction
        preference = self._parse_preference(prediction)

        return {
            "preferred": preference,
            "reasoning": prediction.strip(),
            "prompt": prompt,
            "response_a": response_a,
            "response_b": response_b
        }

    def _parse_preference(self, prediction: str) -> str:
        """Parse preference from model output.

        Args:
            prediction: Raw model output

        Returns:
            'A' or 'B'
        """
        prediction_upper = prediction.upper().strip()

        # Look for explicit A or B at start
        if prediction_upper.startswith('A'):
            return 'A'
        elif prediction_upper.startswith('B'):
            return 'B'

        # Look for "Response A" or "Response B"
        if 'RESPONSE A' in prediction_upper:
            return 'A'
        elif 'RESPONSE B' in prediction_upper:
            return 'B'

        # Default to A if unclear
        return 'A'

    def evaluate_accuracy(self, test_dataset: PreferenceDataset) -> Dict[str, Any]:
        """Evaluate model accuracy on test dataset.

        Args:
            test_dataset: Dataset to evaluate on

        Returns:
            Dict with accuracy metrics
        """
        correct = 0
        total = len(test_dataset)

        predictions = []

        for example in test_dataset.examples:
            # Randomly assign chosen/rejected to A/B
            import random
            if random.random() < 0.5:
                response_a = example.chosen
                response_b = example.rejected
                correct_answer = 'A'
            else:
                response_a = example.rejected
                response_b = example.chosen
                correct_answer = 'B'

            result = self.predict_preference(
                example.prompt,
                response_a,
                response_b
            )

            is_correct = result["preferred"] == correct_answer
            if is_correct:
                correct += 1

            predictions.append({
                "example": example.to_dict(),
                "prediction": result,
                "correct": is_correct
            })

        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "predictions": predictions
        }
