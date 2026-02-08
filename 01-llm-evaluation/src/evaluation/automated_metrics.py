"""
Automated evaluation metrics for LLM outputs.
Includes n-gram metrics, semantic similarity, and diversity measures.
"""

from typing import List, Dict, Optional
import numpy as np
from collections import Counter

# NLP metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer, util


class AutomatedMetrics:
    """Collection of automated evaluation metrics for LLM outputs."""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize automated metrics.

        Args:
            embedding_model: Model name for semantic similarity computation
        """
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True
        )
        self.embedding_model = SentenceTransformer(embedding_model)
        self.smoothing = SmoothingFunction()

    def bleu_score(
        self,
        prediction: str,
        reference: str,
        n: int = 4
    ) -> float:
        """
        Compute BLEU score between prediction and reference.

        Args:
            prediction: Generated text
            reference: Reference text
            n: Maximum n-gram order (default: 4 for BLEU-4)

        Returns:
            BLEU score between 0 and 1
        """
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()

        weights = [1.0 / n] * n
        score = sentence_bleu(
            [ref_tokens],
            pred_tokens,
            weights=weights,
            smoothing_function=self.smoothing.method1
        )
        return score

    def rouge_scores(
        self,
        prediction: str,
        reference: str
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores (R1, R2, RL).

        Args:
            prediction: Generated text
            reference: Reference text

        Returns:
            Dictionary with rouge1, rouge2, rougeL F1 scores
        """
        scores = self.rouge_scorer.score(reference, prediction)
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }

    def semantic_similarity(
        self,
        prediction: str,
        reference: str
    ) -> float:
        """
        Compute semantic similarity using sentence embeddings.

        Args:
            prediction: Generated text
            reference: Reference text

        Returns:
            Cosine similarity between 0 and 1
        """
        embeddings = self.embedding_model.encode(
            [prediction, reference],
            convert_to_tensor=True
        )
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        return max(0.0, min(1.0, similarity))

    def exact_match(
        self,
        prediction: str,
        reference: str,
        normalize: bool = True
    ) -> float:
        """
        Compute exact match score.

        Args:
            prediction: Generated text
            reference: Reference text
            normalize: Whether to lowercase and strip whitespace

        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        if normalize:
            prediction = prediction.lower().strip()
            reference = reference.lower().strip()

        return 1.0 if prediction == reference else 0.0

    def token_f1(
        self,
        prediction: str,
        reference: str
    ) -> float:
        """
        Compute token-level F1 score.

        Args:
            prediction: Generated text
            reference: Reference text

        Returns:
            Token F1 score between 0 and 1
        """
        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(reference.lower().split())

        if len(pred_tokens) == 0 and len(ref_tokens) == 0:
            return 1.0
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0

        common = pred_tokens & ref_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def distinct_ngrams(
        self,
        text: str,
        n: int = 2
    ) -> float:
        """
        Compute distinct n-gram ratio (diversity metric).

        Args:
            text: Input text
            n: N-gram size

        Returns:
            Ratio of distinct n-grams to total n-grams
        """
        tokens = text.lower().split()
        if len(tokens) < n:
            return 0.0

        ngrams = [
            tuple(tokens[i:i + n])
            for i in range(len(tokens) - n + 1)
        ]

        if len(ngrams) == 0:
            return 0.0

        distinct_ratio = len(set(ngrams)) / len(ngrams)
        return distinct_ratio

    def self_bleu(
        self,
        texts: List[str],
        n: int = 4
    ) -> float:
        """
        Compute self-BLEU score (diversity metric).
        Lower scores indicate higher diversity.

        Args:
            texts: List of generated texts
            n: Maximum n-gram order

        Returns:
            Average self-BLEU score
        """
        if len(texts) < 2:
            return 0.0

        scores = []
        for i, text in enumerate(texts):
            references = [t for j, t in enumerate(texts) if i != j]
            for ref in references:
                score = self.bleu_score(text, ref, n=n)
                scores.append(score)

        return np.mean(scores)

    def evaluate(
        self,
        prediction: str,
        reference: Optional[str] = None,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate prediction against reference using specified metrics.

        Args:
            prediction: Generated text
            reference: Reference text (required for most metrics)
            metrics: List of metric names to compute. If None, compute all.

        Returns:
            Dictionary of metric names to scores
        """
        if metrics is None:
            metrics = [
                "bleu",
                "rouge",
                "semantic_similarity",
                "exact_match",
                "token_f1",
                "distinct_2",
            ]

        results = {}

        # Metrics requiring reference
        if reference:
            if "bleu" in metrics:
                results["bleu"] = self.bleu_score(prediction, reference)

            if "rouge" in metrics:
                rouge = self.rouge_scores(prediction, reference)
                results.update(rouge)

            if "semantic_similarity" in metrics:
                results["semantic_similarity"] = self.semantic_similarity(
                    prediction, reference
                )

            if "exact_match" in metrics:
                results["exact_match"] = self.exact_match(prediction, reference)

            if "token_f1" in metrics:
                results["token_f1"] = self.token_f1(prediction, reference)

        # Metrics not requiring reference
        if "distinct_2" in metrics:
            results["distinct_2"] = self.distinct_ngrams(prediction, n=2)

        if "distinct_3" in metrics:
            results["distinct_3"] = self.distinct_ngrams(prediction, n=3)

        return results


# Example usage
if __name__ == "__main__":
    metrics = AutomatedMetrics()

    prediction = "Paris is the capital of France."
    reference = "The capital of France is Paris."

    scores = metrics.evaluate(prediction, reference)

    print("Automated Metrics:")
    for metric, score in scores.items():
        print(f"  {metric}: {score:.3f}")
