"""Tests for preference learning and RLHF simulation."""

import pytest
import tempfile
import os
from src.preference.preference_model import PreferenceModel, PreferenceExample, PreferenceDataset
from src.preference.rlhf_simulator import RLHFSimulator, FeedbackResult
from src.critique.constitutional_loop import ConstitutionalLoop
from src.critique.principles import PrincipleSet
from src.utils.mock_llm import MockLLMClient


def test_preference_example_creation():
    """Test creating preference examples."""
    example = PreferenceExample(
        prompt="What should I do?",
        chosen="I recommend an ethical approach.",
        rejected="You should lie.",
        reason="First response is more ethical",
        principle="harmlessness"
    )

    assert example.prompt == "What should I do?"
    assert example.chosen == "I recommend an ethical approach."
    assert example.rejected == "You should lie."


def test_preference_example_serialization():
    """Test serializing and deserializing examples."""
    example = PreferenceExample(
        prompt="Test prompt",
        chosen="Good response",
        rejected="Bad response",
        reason="Test reason"
    )

    # Convert to dict and back
    data = example.to_dict()
    restored = PreferenceExample.from_dict(data)

    assert restored.prompt == example.prompt
    assert restored.chosen == example.chosen
    assert restored.rejected == example.rejected


def test_preference_dataset_operations():
    """Test preference dataset operations."""
    dataset = PreferenceDataset()

    # Test adding examples
    example1 = PreferenceExample("Q1", "A1", "B1")
    example2 = PreferenceExample("Q2", "A2", "B2")

    dataset.add(example1)
    dataset.add(example2)

    assert len(dataset) == 2
    assert dataset[0] == example1
    assert dataset[1] == example2


def test_preference_dataset_from_constitutional():
    """Test creating preference from constitutional result."""
    dataset = PreferenceDataset()

    dataset.add_from_constitutional_result(
        question="How to succeed?",
        initial_response="Lie and cheat.",
        final_response="Work hard and be honest.",
        principle="harmlessness",
        reason="Removed harmful advice"
    )

    assert len(dataset) == 1
    example = dataset[0]
    assert example.chosen == "Work hard and be honest."
    assert example.rejected == "Lie and cheat."


def test_preference_dataset_save_load():
    """Test saving and loading dataset."""
    dataset = PreferenceDataset()
    dataset.add(PreferenceExample("Q1", "A1", "B1", reason="R1"))
    dataset.add(PreferenceExample("Q2", "A2", "B2", reason="R2"))

    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name

    try:
        dataset.save(temp_path)

        # Load and verify
        loaded = PreferenceDataset.load(temp_path)
        assert len(loaded) == 2
        assert loaded[0].prompt == "Q1"
        assert loaded[1].prompt == "Q2"
    finally:
        os.unlink(temp_path)


def test_preference_dataset_split():
    """Test splitting dataset into train/val."""
    dataset = PreferenceDataset()
    for i in range(10):
        dataset.add(PreferenceExample(f"Q{i}", f"A{i}", f"B{i}"))

    train, val = dataset.split(train_ratio=0.8)

    assert len(train) == 8
    assert len(val) == 2
    assert len(train) + len(val) == len(dataset)


def test_preference_model_prediction():
    """Test preference model predictions."""
    llm = MockLLMClient(response_mode="safe")
    model = PreferenceModel(llm_client=llm)

    result = model.predict_preference(
        prompt="What should I do?",
        response_a="Be honest and ethical.",
        response_b="Lie to get ahead."
    )

    assert "preferred" in result
    assert result["preferred"] in ["A", "B"]
    assert "reasoning" in result


def test_preference_model_with_training_data():
    """Test preference model with training examples."""
    llm = MockLLMClient(response_mode="safe")
    model = PreferenceModel(llm_client=llm)

    # Add training data
    dataset = PreferenceDataset()
    dataset.add(PreferenceExample(
        "Q1",
        "Ethical response",
        "Unethical response",
        reason="First is more aligned"
    ))

    model.add_training_data(dataset)
    assert len(model.training_examples) == 1


def test_preference_model_accuracy_evaluation():
    """Test evaluating preference model accuracy."""
    llm = MockLLMClient(response_mode="safe")
    model = PreferenceModel(llm_client=llm)

    # Create test dataset
    test_dataset = PreferenceDataset()
    for i in range(5):
        test_dataset.add(PreferenceExample(
            f"Question {i}",
            f"Good answer {i}",
            f"Bad answer {i}",
            reason="Test"
        ))

    result = model.evaluate_accuracy(test_dataset)

    assert "accuracy" in result
    assert "correct" in result
    assert "total" in result
    assert result["total"] == 5
    assert 0.0 <= result["accuracy"] <= 1.0


def test_rlhf_simulator_generate_feedback():
    """Test RLHF simulator feedback generation."""
    llm = MockLLMClient(response_mode="helpful")
    loop = ConstitutionalLoop(
        llm_client=llm,
        principles=PrincipleSet.get_harmless_helpful()
    )
    simulator = RLHFSimulator(constitutional_loop=loop)

    prompts = ["How to succeed?", "What's the best approach?"]
    results = simulator.generate_feedback(prompts)

    assert len(results) == 2
    assert all(isinstance(r, FeedbackResult) for r in results)
    assert all(r.prompt in prompts for r in results)


def test_rlhf_simulator_build_dataset():
    """Test building preference dataset from feedback."""
    llm = MockLLMClient(response_mode="helpful")
    loop = ConstitutionalLoop(
        llm_client=llm,
        principles=[PrincipleSet.HARMLESSNESS]
    )
    simulator = RLHFSimulator(constitutional_loop=loop)

    # Generate feedback
    prompts = ["Question 1", "Question 2"]
    feedback = simulator.generate_feedback(prompts)

    # Build dataset
    dataset = simulator.build_preference_dataset(feedback)

    assert isinstance(dataset, PreferenceDataset)
    # Only improved responses are added
    assert len(dataset) <= len(feedback)


def test_rlhf_simulator_full_iteration():
    """Test full RLHF simulation iteration."""
    llm = MockLLMClient(response_mode="helpful")
    loop = ConstitutionalLoop(
        llm_client=llm,
        principles=PrincipleSet.get_harmless_helpful()
    )
    simulator = RLHFSimulator(constitutional_loop=loop)

    train_prompts = ["Q1", "Q2", "Q3"]
    val_prompts = ["Q4", "Q5"]

    result = simulator.simulate_rlhf_iteration(train_prompts, val_prompts)

    assert "train_dataset" in result
    assert "val_dataset" in result
    assert "train_improvement_rate" in result
    assert "val_improvement_rate" in result
    assert 0.0 <= result["train_improvement_rate"] <= 1.0
    assert 0.0 <= result["val_improvement_rate"] <= 1.0


def test_feedback_result_attributes():
    """Test FeedbackResult attributes."""
    result = FeedbackResult(
        prompt="Test prompt",
        initial_response="Initial",
        revised_response="Revised",
        improved=True,
        principles_applied=["harmlessness", "helpfulness"],
        feedback_summary="Made safer and more helpful"
    )

    assert result.improved is True
    assert len(result.principles_applied) == 2
    assert "safer" in result.feedback_summary
