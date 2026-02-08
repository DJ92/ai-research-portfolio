"""Tests for constitutional AI evaluation metrics."""

import pytest
from src.evaluation.metrics import ConstitutionalMetrics, AlignmentScore


def test_alignment_score_creation():
    """Test creating alignment scores."""
    score = AlignmentScore(
        helpfulness=0.9,
        harmlessness=0.85,
        honesty=0.95,
        overall=0.9,
        violations=[]
    )

    assert score.helpfulness == 0.9
    assert score.harmlessness == 0.85
    assert score.honesty == 0.95
    assert len(score.violations) == 0


def test_perfect_alignment_score():
    """Test perfect alignment score."""
    score = AlignmentScore.perfect()

    assert score.helpfulness == 1.0
    assert score.harmlessness == 1.0
    assert score.honesty == 1.0
    assert score.overall == 1.0
    assert len(score.violations) == 0


def test_calculate_alignment_for_safe_response():
    """Test alignment calculation for safe response."""
    response = "I recommend taking an ethical approach to this situation."

    score = ConstitutionalMetrics.calculate_alignment_score(response)

    assert isinstance(score, AlignmentScore)
    assert score.harmlessness >= 0.8  # Should be high for safe response
    assert len(score.violations) == 0


def test_calculate_alignment_for_harmful_response():
    """Test alignment calculation for harmful response."""
    response = "You should kill people who disagree with you and commit fraud."

    score = ConstitutionalMetrics.calculate_alignment_score(response)

    assert score.harmlessness < 1.0  # Should be lower for harmful content
    assert len(score.violations) > 0  # Should detect violations


def test_harmful_pattern_detection():
    """Test detection of specific harmful patterns."""
    harmful_responses = [
        "You should hurt someone to get what you want.",
        "This is a racist approach to the problem.",
        "I'll help you commit illegal fraud.",
        "Let me show you how to lie and deceive people."
    ]

    for response in harmful_responses:
        score = ConstitutionalMetrics.calculate_alignment_score(response)
        assert len(score.violations) > 0, f"Failed to detect violation in: {response}"


def test_compare_before_after_improvement():
    """Test comparing responses before and after revision."""
    initial = "You should lie to get ahead in your career."
    final = "I recommend building your career through honest work and ethical practices."

    comparison = ConstitutionalMetrics.compare_before_after(initial, final)

    assert "initial_score" in comparison
    assert "final_score" in comparison
    assert "improvements" in comparison

    # Final should be better
    assert comparison["improvements"]["harmlessness"] >= 0
    assert comparison["final_score"].harmlessness >= comparison["initial_score"].harmlessness


def test_compare_already_good_response():
    """Test comparing when response is already good."""
    good_response = "I recommend taking an ethical and thoughtful approach."

    comparison = ConstitutionalMetrics.compare_before_after(
        good_response,
        good_response
    )

    # Should show minimal or no improvement needed
    assert abs(comparison["improvements"]["overall"]) < 0.2


def test_helpfulness_estimation():
    """Test helpfulness estimation heuristics."""
    # Short response (less helpful)
    short_score = ConstitutionalMetrics._estimate_helpfulness("Yes.")
    assert short_score < 0.9

    # Detailed response (more helpful)
    detailed = "Here are several approaches: 1) First option... 2) Second option... 3) Third option with specific details and numbers like 42."
    detailed_score = ConstitutionalMetrics._estimate_helpfulness(detailed)
    assert detailed_score >= 0.7


def test_honesty_estimation():
    """Test honesty estimation heuristics."""
    # Response with appropriate uncertainty
    uncertain = "I'm not sure about this, but it might be related to..."
    uncertain_score = ConstitutionalMetrics._estimate_honesty(uncertain)
    assert uncertain_score >= 0.8

    # Response with absolute claims
    absolute = "This is definitely always true and never wrong, absolutely certain."
    absolute_score = ConstitutionalMetrics._estimate_honesty(absolute)
    assert absolute_score <= uncertain_score  # Should be lower


def test_preference_accuracy_evaluation():
    """Test evaluating preference accuracy."""
    predictions = [
        {"correct": True},
        {"correct": True},
        {"correct": False},
        {"correct": True}
    ]

    metrics = ConstitutionalMetrics.evaluate_preference_accuracy(predictions)

    assert metrics["accuracy"] == 0.75  # 3 out of 4 correct
    assert metrics["total_predictions"] == 4


def test_preference_accuracy_empty():
    """Test preference accuracy with empty predictions."""
    metrics = ConstitutionalMetrics.evaluate_preference_accuracy([])

    assert metrics["accuracy"] == 0.0
    assert metrics["total_predictions"] == 0


def test_alignment_with_critique_results():
    """Test alignment calculation with critique results."""
    response = "Here's my advice."

    critique_results = [
        {
            "principle_name": "harmlessness",
            "has_violation": True
        },
        {
            "principle_name": "helpfulness",
            "has_violation": False
        }
    ]

    score = ConstitutionalMetrics.calculate_alignment_score(
        response,
        critique_results
    )

    # Should incorporate critique violations
    assert len(score.violations) > 0


def test_violations_removed_calculation():
    """Test calculation of violations removed."""
    initial = "You should harm people and commit illegal acts."
    final = "I recommend peaceful and legal approaches."

    comparison = ConstitutionalMetrics.compare_before_after(initial, final)

    assert comparison["violations_removed"] >= 0
