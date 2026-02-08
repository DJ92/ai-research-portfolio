"""
Unit tests for automated evaluation metrics.
"""

import pytest
from src.evaluation.automated_metrics import AutomatedMetrics


@pytest.fixture
def metrics():
    """Create AutomatedMetrics instance for testing."""
    return AutomatedMetrics()


class TestBLEUScore:
    """Tests for BLEU score computation."""

    def test_perfect_match(self, metrics):
        """Test BLEU score for identical texts."""
        text = "The cat sat on the mat."
        score = metrics.bleu_score(text, text)
        assert score == 1.0

    def test_completely_different(self, metrics):
        """Test BLEU score for completely different texts."""
        pred = "Hello world"
        ref = "Goodbye universe"
        score = metrics.bleu_score(pred, ref)
        assert score < 0.1

    def test_partial_overlap(self, metrics):
        """Test BLEU score for partial overlap."""
        pred = "The cat sat on the mat."
        ref = "The dog sat on the mat."
        score = metrics.bleu_score(pred, ref)
        assert 0.3 < score < 0.9


class TestROUGEScores:
    """Tests for ROUGE scores computation."""

    def test_perfect_match(self, metrics):
        """Test ROUGE scores for identical texts."""
        text = "The cat sat on the mat."
        scores = metrics.rouge_scores(text, text)
        assert scores["rouge1"] == 1.0
        assert scores["rouge2"] == 1.0
        assert scores["rougeL"] == 1.0

    def test_no_overlap(self, metrics):
        """Test ROUGE scores for no overlap."""
        pred = "Hello world"
        ref = "Goodbye universe"
        scores = metrics.rouge_scores(pred, ref)
        assert scores["rouge1"] == 0.0


class TestSemanticSimilarity:
    """Tests for semantic similarity."""

    def test_identical_text(self, metrics):
        """Test semantic similarity for identical texts."""
        text = "Paris is the capital of France."
        score = metrics.semantic_similarity(text, text)
        assert score > 0.99

    def test_similar_meaning(self, metrics):
        """Test semantic similarity for similar meanings."""
        pred = "Paris is the capital of France."
        ref = "The capital of France is Paris."
        score = metrics.semantic_similarity(pred, ref)
        assert score > 0.8

    def test_different_meaning(self, metrics):
        """Test semantic similarity for different meanings."""
        pred = "Paris is the capital of France."
        ref = "Python is a programming language."
        score = metrics.semantic_similarity(pred, ref)
        assert score < 0.5


class TestExactMatch:
    """Tests for exact match metric."""

    def test_exact_match_true(self, metrics):
        """Test exact match when texts are identical."""
        text = "Hello world"
        assert metrics.exact_match(text, text) == 1.0

    def test_exact_match_false(self, metrics):
        """Test exact match when texts differ."""
        assert metrics.exact_match("Hello", "World") == 0.0

    def test_exact_match_normalized(self, metrics):
        """Test exact match with normalization."""
        pred = "  HELLO WORLD  "
        ref = "hello world"
        assert metrics.exact_match(pred, ref, normalize=True) == 1.0


class TestTokenF1:
    """Tests for token F1 score."""

    def test_perfect_match(self, metrics):
        """Test token F1 for identical texts."""
        text = "The cat sat on the mat"
        assert metrics.token_f1(text, text) == 1.0

    def test_partial_overlap(self, metrics):
        """Test token F1 for partial overlap."""
        pred = "The cat sat"
        ref = "The dog sat on the mat"
        f1 = metrics.token_f1(pred, ref)
        # Precision: 2/3 (the, sat), Recall: 2/6
        # F1 = 2 * (2/3 * 2/6) / (2/3 + 2/6) = 0.4
        assert 0.35 < f1 < 0.45

    def test_no_overlap(self, metrics):
        """Test token F1 for no overlap."""
        pred = "Hello world"
        ref = "Goodbye universe"
        assert metrics.token_f1(pred, ref) == 0.0


class TestDistinctNgrams:
    """Tests for distinct n-grams metric."""

    def test_all_distinct(self, metrics):
        """Test distinct ratio when all n-grams are unique."""
        text = "the cat sat on the mat with the hat"
        score = metrics.distinct_ngrams(text, n=2)
        # 8 bigrams total, some repeated "the"
        assert score < 1.0

    def test_all_repeated(self, metrics):
        """Test distinct ratio for repeated text."""
        text = "the the the the"
        score = metrics.distinct_ngrams(text, n=2)
        # All bigrams are "the the"
        assert score == 1.0

    def test_empty_text(self, metrics):
        """Test distinct ratio for empty or short text."""
        assert metrics.distinct_ngrams("", n=2) == 0.0
        assert metrics.distinct_ngrams("hi", n=2) == 0.0


class TestEvaluate:
    """Tests for the main evaluate method."""

    def test_evaluate_all_metrics(self, metrics):
        """Test evaluation with all metrics."""
        pred = "Paris is the capital of France."
        ref = "The capital of France is Paris."

        results = metrics.evaluate(pred, ref)

        assert "bleu" in results
        assert "rouge1" in results
        assert "semantic_similarity" in results
        assert "exact_match" in results
        assert "token_f1" in results
        assert "distinct_2" in results

        # All scores should be between 0 and 1
        for metric, score in results.items():
            assert 0.0 <= score <= 1.0

    def test_evaluate_specific_metrics(self, metrics):
        """Test evaluation with specific metrics only."""
        pred = "Hello world"
        ref = "Hello world"

        results = metrics.evaluate(
            pred,
            ref,
            metrics=["bleu", "exact_match"]
        )

        assert "bleu" in results
        assert "exact_match" in results
        assert "rouge1" not in results

    def test_evaluate_without_reference(self, metrics):
        """Test evaluation metrics that don't require reference."""
        pred = "The cat sat on the mat with the hat"

        results = metrics.evaluate(
            pred,
            reference=None,
            metrics=["distinct_2", "distinct_3"]
        )

        assert "distinct_2" in results
        assert "distinct_3" in results
        # Bleu/rouge should not be present without reference
        assert "bleu" not in results
