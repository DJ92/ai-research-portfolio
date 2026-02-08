"""
Unified embedding interface for multiple providers.
"""

from typing import List, Union
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer
import openai
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class EmbeddingResult:
    """Result from embedding operation."""

    embeddings: np.ndarray  # Shape: (n_texts, embedding_dim)
    model: str
    dimensions: int
    latency_ms: float


class Embedder(ABC):
    """Abstract base class for embedding models."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Embedding dimensionality."""
        pass

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            Matrix of embeddings, shape (len(texts), dimensions)
        """
        pass


class SentenceTransformerEmbedder(Embedder):
    """
    Embedder using Sentence-Transformers models.

    Free, runs locally, good quality for many use cases.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize Sentence-Transformer embedder.

        Args:
            model_name: HuggingFace model name
                Popular choices:
                - all-MiniLM-L6-v2 (384d, fast, good quality)
                - all-mpnet-base-v2 (768d, slower, better quality)
                - all-MiniLM-L12-v2 (384d, balanced)
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        return self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        """Embed single text."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed multiple texts efficiently."""
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=False
        )
        return embeddings


class OpenAIEmbedder(Embedder):
    """
    Embedder using OpenAI's embedding models.

    Paid API, high quality, requires API key.
    """

    DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model: str = "text-embedding-3-small"):
        """
        Initialize OpenAI embedder.

        Args:
            model: OpenAI model name
                - text-embedding-3-small (1536d, $0.02/1M tokens)
                - text-embedding-3-large (3072d, $0.13/1M tokens)
                - text-embedding-ada-002 (1536d, legacy)
        """
        self.model = model
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        self.client = openai.OpenAI(api_key=api_key)

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        return self.DIMENSIONS.get(self.model, 1536)

    def embed(self, text: str) -> np.ndarray:
        """Embed single text."""
        response = self.client.embeddings.create(input=text, model=self.model)
        embedding = np.array(response.data[0].embedding)
        return embedding

    def embed_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Embed multiple texts."""
        # OpenAI supports batch embedding
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(input=batch, model=self.model)

            batch_embeddings = [np.array(item.embedding) for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings)


def get_embedder(provider: str, model: str = None) -> Embedder:
    """
    Factory function to get embedder by provider.

    Args:
        provider: One of "sentence-transformers", "openai"
        model: Model name (uses default if None)

    Returns:
        Embedder instance
    """
    if provider == "sentence-transformers":
        model = model or "all-MiniLM-L6-v2"
        return SentenceTransformerEmbedder(model)
    elif provider == "openai":
        model = model or "text-embedding-3-small"
        return OpenAIEmbedder(model)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


# Example usage
if __name__ == "__main__":
    # Test Sentence-Transformers
    print("=== Sentence-Transformers Embedder ===")
    st_embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    print(f"Dimensions: {st_embedder.dimensions}")

    texts = [
        "RAG systems retrieve relevant context",
        "Embedding models convert text to vectors",
        "Pizza is my favorite food",
    ]

    embeddings = st_embedder.embed_batch(texts)
    print(f"Embedded {len(texts)} texts, shape: {embeddings.shape}")

    # Compute similarity
    from sklearn.metrics.pairwise import cosine_similarity

    similarities = cosine_similarity(embeddings)
    print("\nSimilarity matrix:")
    print(similarities)

    print("\nSimilarity between texts:")
    print(f"Text 0 <-> Text 1: {similarities[0, 1]:.3f}")
    print(f"Text 0 <-> Text 2: {similarities[0, 2]:.3f}")
    print(f"Text 1 <-> Text 2: {similarities[1, 2]:.3f}")
