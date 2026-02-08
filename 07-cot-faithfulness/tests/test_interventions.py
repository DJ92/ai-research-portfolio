"""Tests for counterfactual interventions."""

import pytest
from src.analysis.cot_parser import CoTParser, CoTStep, CoTChain
from src.interventions.counterfactual import CounterfactualInterventions, InterventionResult
from src.utils.mock_llm import MockCoTLLM


def test_faithful_model_interventions():
    """Test that faithful model changes answer when reasoning is modified."""
    llm = MockCoTLLM(mode="faithful")
    interventions = CounterfactualInterventions(llm_client=llm)

    response = llm.generate("What is 2+2?")
    chain = CoTParser.parse(response, "What is 2+2?")

    result = interventions.run_intervention(chain, "modify_step")

    assert isinstance(result, InterventionResult)
    # Faithful model should change answer when step is modified
    assert result.answer_changed is True


def test_posthoc_model_resistant_to_interventions():
    """Test that post-hoc model doesn't change answer when reasoning is modified."""
    llm = MockCoTLLM(mode="post_hoc")
    interventions = CounterfactualInterventions(llm_client=llm)

    response = llm.generate("What is 2+2?")
    chain = CoTParser.parse(response, "What is 2+2?")

    result = interventions.run_intervention(chain, "modify_step")

    # Post-hoc model should NOT change answer (reasoning not used)
    assert result.answer_changed is False


def test_remove_step_intervention():
    """Test removing a step from reasoning."""
    llm = MockCoTLLM(mode="faithful")
    interventions = CounterfactualInterventions(llm_client=llm)

    steps = [
        CoTStep(1, "Step 1", "assumption"),
        CoTStep(2, "Step 2", "calculation"),
        CoTStep(3, "Step 3", "conclusion"),
    ]

    chain = CoTChain(
        steps=steps,
        final_answer="4",
        question="What is 2+2?",
        raw_response=""
    )

    result = interventions.run_intervention(chain, "remove_step", step_number=2)

    assert result.intervention_type == "remove_step"
    assert result.step_modified == 2


def test_reorder_steps_intervention():
    """Test reordering reasoning steps."""
    llm = MockCoTLLM(mode="faithful")
    interventions = CounterfactualInterventions(llm_client=llm)

    steps = [
        CoTStep(1, "Step 1", "assumption"),
        CoTStep(2, "Step 2", "calculation"),
        CoTStep(3, "Step 3", "inference"),
        CoTStep(4, "Step 4", "conclusion"),
    ]

    chain = CoTChain(
        steps=steps,
        final_answer="4",
        question="Test",
        raw_response=""
    )

    result = interventions.run_intervention(chain, "reorder_steps")

    assert result.intervention_type == "reorder_steps"


def test_corrupt_early_step():
    """Test corrupting an early reasoning step."""
    llm = MockCoTLLM(mode="faithful")
    interventions = CounterfactualInterventions(llm_client=llm)

    response = llm.generate("What is 2+2?")
    chain = CoTParser.parse(response, "What is 2+2?")

    result = interventions.run_intervention(chain, "corrupt_early")

    assert result.intervention_type == "corrupt_early"
    # Faithful model should change answer when early step is corrupted
    assert result.answer_changed is True


def test_run_all_interventions():
    """Test running all intervention types."""
    llm = MockCoTLLM(mode="faithful")
    interventions = CounterfactualInterventions(llm_client=llm)

    steps = [
        CoTStep(1, "Step 1", "assumption"),
        CoTStep(2, "Step 2", "calculation"),
        CoTStep(3, "Step 3", "conclusion"),
    ]

    chain = CoTChain(
        steps=steps,
        final_answer="4",
        question="What is 2+2?",
        raw_response=""
    )

    results = interventions.run_all_interventions(chain)

    assert len(results) > 0
    assert all(isinstance(r, InterventionResult) for r in results)


def test_faithfulness_score_calculation():
    """Test calculating faithfulness score from interventions."""
    interventions = CounterfactualInterventions(llm_client=MockCoTLLM())

    # Faithful reasoning (all interventions change answer)
    faithful_results = [
        InterventionResult("4", "5", True, "modify", "test"),
        InterventionResult("4", "6", True, "remove", "test"),
        InterventionResult("4", "3", True, "corrupt", "test"),
    ]

    faithful_score = interventions.calculate_faithfulness_score(faithful_results)
    assert faithful_score == 1.0

    # Unfaithful reasoning (interventions don't change answer)
    unfaithful_results = [
        InterventionResult("4", "4", False, "modify", "test"),
        InterventionResult("4", "4", False, "remove", "test"),
    ]

    unfaithful_score = interventions.calculate_faithfulness_score(unfaithful_results)
    assert unfaithful_score == 0.0


def test_answers_differ_detection():
    """Test detecting when answers differ."""
    interventions = CounterfactualInterventions(llm_client=MockCoTLLM())

    # Same answers
    assert interventions._answers_differ("42", "42") is False
    assert interventions._answers_differ("The answer is 5", "The answer is 5") is False

    # Different answers
    assert interventions._answers_differ("42", "43") is True
    assert interventions._answers_differ("Yes", "No") is True

    # Different numeric values
    assert interventions._answers_differ("The answer is 5", "The answer is 7") is True


def test_intervention_with_minimal_steps():
    """Test interventions on chain with minimal steps."""
    llm = MockCoTLLM()
    interventions = CounterfactualInterventions(llm_client=llm)

    # Single step chain
    chain = CoTChain(
        steps=[CoTStep(1, "The answer is 4", "conclusion")],
        final_answer="4",
        question="What is 2+2?",
        raw_response=""
    )

    result = interventions.run_intervention(chain, "modify_step")
    assert isinstance(result, InterventionResult)

    # Empty chain
    empty_chain = CoTChain(
        steps=[],
        final_answer="4",
        question="Test",
        raw_response=""
    )

    result = interventions.run_intervention(empty_chain, "modify_step")
    assert result.answer_changed is False
