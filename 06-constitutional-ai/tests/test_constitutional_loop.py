"""Tests for constitutional critique-revision loop."""

import pytest
from src.critique.constitutional_loop import ConstitutionalLoop, CritiqueResult, RevisionResult
from src.critique.principles import PrincipleSet
from src.utils.mock_llm import MockLLMClient


def test_critique_detects_violations():
    """Test that critique detects violations in harmful responses."""
    llm = MockLLMClient(response_mode="harmful")
    loop = ConstitutionalLoop(
        llm_client=llm,
        principles=PrincipleSet.get_harmless_helpful()
    )

    question = "How should I handle a difficult situation?"
    response = llm.generate(question)

    critique = loop.critique(
        response=response,
        question=question,
        principle=PrincipleSet.HARMLESSNESS
    )

    assert isinstance(critique, CritiqueResult)
    assert critique.principle_name == "harmlessness"
    assert critique.has_violation is True
    assert len(critique.critique) > 0


def test_critique_no_violations_for_safe_response():
    """Test that critique doesn't flag safe responses."""
    llm = MockLLMClient(response_mode="safe")
    loop = ConstitutionalLoop(
        llm_client=llm,
        principles=[PrincipleSet.HARMLESSNESS]
    )

    question = "How should I handle a difficult situation?"
    response = llm.generate(question)

    critique = loop.critique(
        response=response,
        question=question,
        principle=PrincipleSet.HARMLESSNESS
    )

    assert critique.has_violation is False


def test_revision_improves_response():
    """Test that revision produces different output."""
    llm = MockLLMClient(response_mode="harmful")
    loop = ConstitutionalLoop(
        llm_client=llm,
        principles=[PrincipleSet.HARMLESSNESS]
    )

    question = "How should I handle a difficult situation?"
    response = "You should lie to get what you want."
    critique = "This encourages dishonesty, which is harmful."

    revision = loop.revise(
        response=response,
        question=question,
        critique=critique,
        principle=PrincipleSet.HARMLESSNESS
    )

    assert isinstance(revision, RevisionResult)
    assert revision.revised_response != response
    assert len(revision.revised_response) > 0
    assert revision.principle_name == "harmlessness"


def test_apply_principle_single_iteration():
    """Test applying a single principle through critique-revision."""
    llm = MockLLMClient(response_mode="harmful")
    loop = ConstitutionalLoop(
        llm_client=llm,
        principles=[PrincipleSet.HARMLESSNESS],
        max_iterations=3
    )

    question = "What should I do?"
    response = llm.generate(question)

    result = loop.apply_principle(
        response=response,
        question=question,
        principle=PrincipleSet.HARMLESSNESS
    )

    assert "final_response" in result
    assert "iterations" in result
    assert result["total_iterations"] > 0
    assert result["principle"] == "harmlessness"


def test_full_constitutional_loop():
    """Test full constitutional loop with multiple principles."""
    llm = MockLLMClient(response_mode="helpful")
    loop = ConstitutionalLoop(
        llm_client=llm,
        principles=PrincipleSet.get_harmless_helpful(),
        max_iterations=2
    )

    question = "How can I be more successful?"

    result = loop.run(question=question)

    assert "initial_response" in result
    assert "final_response" in result
    assert "principle_results" in result
    assert result["num_principles"] == 2
    assert len(result["principle_results"]) == 2


def test_loop_with_safe_initial_response():
    """Test that safe responses don't trigger many revisions."""
    llm = MockLLMClient(response_mode="safe")
    loop = ConstitutionalLoop(
        llm_client=llm,
        principles=PrincipleSet.get_harmless_helpful()
    )

    question = "How should I approach learning?"
    result = loop.run(question=question)

    # Safe response shouldn't need many iterations
    total_iterations = sum(
        pr["total_iterations"]
        for pr in result["principle_results"]
    )
    assert total_iterations <= 2


def test_max_iterations_limit():
    """Test that max_iterations limit is respected."""
    llm = MockLLMClient(response_mode="harmful")
    loop = ConstitutionalLoop(
        llm_client=llm,
        principles=[PrincipleSet.HARMLESSNESS],
        max_iterations=2
    )

    question = "What should I do?"
    response = llm.generate(question)

    result = loop.apply_principle(
        response=response,
        question=question,
        principle=PrincipleSet.HARMLESSNESS
    )

    assert result["total_iterations"] <= 2


def test_violation_detection_heuristics():
    """Test violation detection logic."""
    llm = MockLLMClient()
    loop = ConstitutionalLoop(llm_client=llm, principles=[])

    # Test phrases that indicate no violation
    assert loop._detect_violation("No issues found.") is False
    assert loop._detect_violation("The response is appropriate.") is False
    assert loop._detect_violation("Not harmful.") is False

    # Test phrases that indicate violation
    assert loop._detect_violation("This response contains harmful content that promotes violence.") is True
    assert loop._detect_violation("The response is problematic because it encourages unethical behavior.") is True

    # Test short critiques (likely no violation)
    assert loop._detect_violation("None.") is False
