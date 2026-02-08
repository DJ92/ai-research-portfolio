"""Tests for CoT parser."""

import pytest
from src.analysis.cot_parser import CoTParser, CoTChain, CoTStep


def test_parse_explicit_steps():
    """Test parsing response with explicit steps."""
    response = """Step 1: First, let's identify the numbers.
Step 2: We need to add 2 and 2.
Step 3: 2 + 2 = 4.
Therefore, the answer is 4."""

    chain = CoTParser.parse(response, "What is 2+2?")

    assert isinstance(chain, CoTChain)
    assert chain.num_steps() == 3
    assert chain.final_answer == "4"
    assert chain.question == "What is 2+2?"


def test_parse_implicit_steps():
    """Test parsing response without explicit step markers."""
    response = """First, we identify the problem. Then, we solve it. Finally, we get 4."""

    chain = CoTParser.parse(response, "What is 2+2?")

    assert chain.num_steps() > 0
    assert len(chain.steps) >= 1


def test_extract_final_answer():
    """Test extracting final answer from various formats."""
    test_cases = [
        ("Therefore, the answer is 42.", "42"),
        ("The answer is: 100", "100"),
        ("So it equals 7", "7"),
        ("Final answer: Yes", "Yes"),
    ]

    for response, expected in test_cases:
        answer = CoTParser._extract_final_answer(response)
        assert expected.lower() in answer.lower()


def test_classify_step_types():
    """Test step type classification."""
    test_cases = [
        ("Let's assume x = 5", "assumption"),
        ("Calculate 2 + 2 = 4", "calculation"),
        ("Therefore, the result is 10", "inference"),
        ("Finally, we conclude that...", "conclusion"),
    ]

    for content, expected_type in test_cases:
        step_type = CoTParser._classify_step(content)
        assert step_type == expected_type


def test_cot_step_creation():
    """Test creating CoT steps."""
    step = CoTStep(
        step_number=1,
        content="First step",
        step_type="assumption",
        dependencies=[0]
    )

    assert step.step_number == 1
    assert step.content == "First step"
    assert step.step_type == "assumption"
    assert 0 in step.dependencies


def test_cot_chain_get_step():
    """Test retrieving steps from chain."""
    steps = [
        CoTStep(1, "Step 1", "assumption"),
        CoTStep(2, "Step 2", "calculation"),
        CoTStep(3, "Step 3", "conclusion"),
    ]

    chain = CoTChain(
        steps=steps,
        final_answer="42",
        question="Test",
        raw_response="Test response"
    )

    assert chain.get_step(2).content == "Step 2"
    assert chain.get_step(99) is None


def test_count_reasoning_tokens():
    """Test token counting."""
    steps = [
        CoTStep(1, "A" * 40, "assumption"),
        CoTStep(2, "B" * 40, "calculation"),
    ]

    chain = CoTChain(
        steps=steps,
        final_answer="42",
        question="Test",
        raw_response=""
    )

    token_count = CoTParser.count_reasoning_tokens(chain)
    assert token_count > 0
    assert token_count == (80 + 1) // 4  # Rough approximation


def test_parse_numbered_list():
    """Test parsing numbered list format."""
    response = """1. First, identify the problem.
2. Second, find the solution.
3. Third, verify the answer.
The answer is 42."""

    chain = CoTParser.parse(response, "Test question")

    assert chain.num_steps() >= 3
    assert "42" in chain.final_answer


def test_empty_response():
    """Test parsing empty or minimal response."""
    response = ""
    chain = CoTParser.parse(response, "Test")

    assert chain.num_steps() >= 0
    assert chain.final_answer == ""
