"""Tests for faithfulness analysis."""

import pytest
from src.analysis.cot_parser import CoTParser, CoTChain, CoTStep
from src.analysis.faithfulness_analyzer import FaithfulnessAnalyzer, FaithfulnessResult
from src.utils.mock_llm import MockCoTLLM


def test_analyze_faithful_reasoning():
    """Test analyzing faithful reasoning."""
    llm = MockCoTLLM(mode="faithful")
    analyzer = FaithfulnessAnalyzer(llm_client=llm)

    response = llm.generate("What is 2+2?")
    chain = CoTParser.parse(response, "What is 2+2?")

    result = analyzer.analyze(chain)

    assert isinstance(result, FaithfulnessResult)
    assert result.reasoning_quality > 0.5
    assert "post_hoc_rationalization" not in result.failure_modes


def test_analyze_posthoc_reasoning():
    """Test detecting post-hoc rationalization."""
    llm = MockCoTLLM(mode="post_hoc")
    analyzer = FaithfulnessAnalyzer(llm_client=llm)

    response = llm.generate("What is 2+2?")
    chain = CoTParser.parse(response, "What is 2+2?")

    result = analyzer.analyze(chain)

    # Post-hoc reasoning should have higher post-hoc score
    assert result.metrics.get("posthoc_score", 0) > 0.3


def test_reasoning_quality_assessment():
    """Test reasoning quality scoring."""
    llm = MockCoTLLM()
    analyzer = FaithfulnessAnalyzer(llm_client=llm)

    # Good reasoning with multiple steps
    good_steps = [
        CoTStep(1, "Identify the problem: 2+2", "assumption"),
        CoTStep(2, "Perform calculation: 2+2=4", "calculation"),
        CoTStep(3, "Verify: 4 is correct", "verification"),
        CoTStep(4, "Therefore, answer is 4", "conclusion"),
    ]

    good_chain = CoTChain(
        steps=good_steps,
        final_answer="4",
        question="What is 2+2?",
        raw_response="Full response with steps"
    )

    quality = analyzer._assess_reasoning_quality(good_chain)
    assert quality > 0.6

    # Poor reasoning with minimal steps
    poor_chain = CoTChain(
        steps=[CoTStep(1, "It's 4", "conclusion")],
        final_answer="4",
        question="What is 2+2?",
        raw_response="It's 4"
    )

    poor_quality = analyzer._assess_reasoning_quality(poor_chain)
    assert poor_quality < quality


def test_consistency_checking():
    """Test logical consistency checking."""
    llm = MockCoTLLM()
    analyzer = FaithfulnessAnalyzer(llm_client=llm)

    # Consistent steps
    consistent_steps = [
        CoTStep(1, "Let x = 2", "assumption"),
        CoTStep(2, "Then x + x = 4", "calculation"),
        CoTStep(3, "So 2 + 2 = 4", "conclusion"),
    ]

    consistent_chain = CoTChain(
        steps=consistent_steps,
        final_answer="4",
        question="What is 2+2?",
        raw_response=""
    )

    score, issues = analyzer._check_consistency(consistent_chain)
    assert score > 0.5
    assert len(issues) == 0 or "answer_not_in_reasoning" in issues


def test_necessity_checking():
    """Test step necessity checking."""
    llm = MockCoTLLM()
    analyzer = FaithfulnessAnalyzer(llm_client=llm)

    # Necessary steps
    steps = [
        CoTStep(1, "Identify numbers", "assumption"),
        CoTStep(2, "Add them", "calculation"),
        CoTStep(3, "Get result", "conclusion"),
    ]

    chain = CoTChain(
        steps=steps,
        final_answer="4",
        question="What is 2+2?",
        raw_response=""
    )

    score, issues = analyzer._check_necessity(chain)
    assert score > 0.0


def test_posthoc_pattern_detection():
    """Test detection of post-hoc patterns."""
    llm = MockCoTLLM()
    analyzer = FaithfulnessAnalyzer(llm_client=llm)

    # Response with answer mentioned early (post-hoc indicator)
    posthoc_steps = [
        CoTStep(1, "The answer is clearly 4", "assumption"),
        CoTStep(2, "Because 2+2 equals 4", "calculation"),
        CoTStep(3, "So it's 4", "conclusion"),
    ]

    posthoc_chain = CoTChain(
        steps=posthoc_steps,
        final_answer="4",
        question="What is 2+2?",
        raw_response="The answer is clearly 4. Because 2+2 equals 4. So it's 4."
    )

    posthoc_score = analyzer._detect_posthoc_patterns(posthoc_chain)
    assert posthoc_score > 0.2


def test_answer_quality_assessment():
    """Test answer quality scoring."""
    llm = MockCoTLLM()
    analyzer = FaithfulnessAnalyzer(llm_client=llm)

    # Good answer
    good_chain = CoTChain(
        steps=[],
        final_answer="The result is 42, calculated through systematic analysis.",
        question="What is the answer?",
        raw_response=""
    )

    good_quality = analyzer._assess_answer_quality(good_chain)
    assert good_quality > 0.3

    # Poor answer (empty)
    poor_chain = CoTChain(
        steps=[],
        final_answer="",
        question="What is the answer?",
        raw_response=""
    )

    poor_quality = analyzer._assess_answer_quality(poor_chain)
    assert poor_quality < good_quality


def test_faithfulness_calculation():
    """Test overall faithfulness calculation."""
    llm = MockCoTLLM()
    analyzer = FaithfulnessAnalyzer(llm_client=llm)

    # High quality metrics
    good_metrics = {
        "reasoning_quality": 0.9,
        "consistency_score": 0.9,
        "necessity_score": 0.8,
        "posthoc_score": 0.1
    }

    is_faithful, confidence = analyzer._calculate_faithfulness(good_metrics)
    assert is_faithful is True
    assert confidence > 0.3

    # Low quality metrics
    poor_metrics = {
        "reasoning_quality": 0.3,
        "consistency_score": 0.4,
        "necessity_score": 0.3,
        "posthoc_score": 0.8
    }

    is_faithful, confidence = analyzer._calculate_faithfulness(poor_metrics)
    assert is_faithful is False
