"""
Tests for prompting techniques.
"""

import pytest
from src.techniques.base import ZeroShot, FewShot, ChainOfThought


class TestZeroShot:
    """Tests for zero-shot prompting."""

    def test_build_prompt(self):
        """Test zero-shot prompt construction."""
        technique = ZeroShot(
            instruction="Classify sentiment as positive or negative."
        )

        prompt = technique.build_prompt("Great product!")

        assert "Classify sentiment" in prompt
        assert "Great product!" in prompt
        assert "Input:" in prompt
        assert "Output:" in prompt

    def test_prompt_includes_instruction(self):
        """Test that instruction is included in prompt."""
        instruction = "Translate to French."
        technique = ZeroShot(instruction=instruction)

        prompt = technique.build_prompt("Hello world")

        assert instruction in prompt


class TestFewShot:
    """Tests for few-shot prompting."""

    @pytest.fixture
    def examples(self):
        """Sample few-shot examples."""
        return [
            {"input": "Great!", "output": "positive"},
            {"input": "Terrible.", "output": "negative"},
            {"input": "OK", "output": "neutral"},
        ]

    def test_build_prompt_includes_examples(self, examples):
        """Test that examples are included in prompt."""
        technique = FewShot(
            instruction="Classify sentiment.",
            examples=examples
        )

        prompt = technique.build_prompt("Good product")

        # Check all examples are present
        for example in examples:
            assert example["input"] in prompt
            assert example["output"] in prompt

    def test_prompt_structure(self, examples):
        """Test prompt has correct structure."""
        technique = FewShot(
            instruction="Classify sentiment.",
            examples=examples
        )

        prompt = technique.build_prompt("Test input")

        assert "Example 1:" in prompt
        assert "Example 2:" in prompt
        assert "Example 3:" in prompt
        assert "Test input" in prompt

    def test_empty_examples(self):
        """Test few-shot with no examples."""
        technique = FewShot(
            instruction="Classify sentiment.",
            examples=[]
        )

        prompt = technique.build_prompt("Test")

        assert "Test" in prompt
        assert "Classify sentiment" in prompt


class TestChainOfThought:
    """Tests for Chain-of-Thought prompting."""

    def test_build_prompt_includes_reasoning_prompt(self):
        """Test that CoT prompt encourages step-by-step reasoning."""
        technique = ChainOfThought(
            instruction="Solve this math problem."
        )

        prompt = technique.build_prompt("What is 2+2?")

        assert "step by step" in prompt.lower()
        assert "What is 2+2?" in prompt

    def test_prompt_starts_reasoning(self):
        """Test that prompt starts the reasoning chain."""
        technique = ChainOfThought(
            instruction="Solve the problem."
        )

        prompt = technique.build_prompt("Test problem")

        # Should start with "1." to encourage numbered steps
        assert "1." in prompt


class TestPromptingTechniqueBase:
    """Tests for base PromptingTechnique functionality."""

    def test_zero_shot_has_correct_attributes(self):
        """Test that technique has expected attributes."""
        technique = ZeroShot(
            instruction="Test",
            model="claude-sonnet-4.5",
            temperature=0.5
        )

        assert technique.model == "claude-sonnet-4.5"
        assert technique.temperature == 0.5
        assert technique.instruction == "Test"

    def test_few_shot_stores_examples(self):
        """Test that examples are stored correctly."""
        examples = [{"input": "a", "output": "b"}]
        technique = FewShot(
            instruction="Test",
            examples=examples
        )

        assert technique.examples == examples
        assert len(technique.examples) == 1
