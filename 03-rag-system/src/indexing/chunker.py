"""
Document chunking strategies for RAG systems.
"""

from typing import List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import re


@dataclass
class Chunk:
    """Represents a chunk of text with metadata."""

    text: str
    doc_id: str
    chunk_id: str
    start_char: int
    end_char: int
    metadata: dict


class Chunker(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, doc_id: str, metadata: Optional[dict] = None) -> List[Chunk]:
        """
        Split text into chunks.

        Args:
            text: Document text to chunk
            doc_id: Unique document identifier
            metadata: Optional metadata to attach to chunks

        Returns:
            List of Chunk objects
        """
        pass


class FixedSizeChunker(Chunker):
    """
    Fixed-size chunking with optional overlap.

    Splits text into chunks of approximately equal size.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize fixed-size chunker.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, doc_id: str, metadata: Optional[dict] = None) -> List[Chunk]:
        """Chunk text into fixed-size pieces."""
        if metadata is None:
            metadata = {}

        chunks = []
        start = 0
        chunk_num = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Try to break at word boundary
            if end < len(text):
                # Look back for space
                space_pos = text.rfind(" ", start, end)
                if space_pos > start:
                    end = space_pos

            chunk_text = text[start:end].strip()

            if chunk_text:  # Only add non-empty chunks
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        doc_id=doc_id,
                        chunk_id=f"{doc_id}_chunk_{chunk_num}",
                        start_char=start,
                        end_char=end,
                        metadata=metadata.copy(),
                    )
                )
                chunk_num += 1

            # Move to next chunk with overlap
            start = end - self.chunk_overlap if end < len(text) else end

        return chunks


class SentenceChunker(Chunker):
    """
    Sentence-based chunking.

    Splits on sentence boundaries, combining sentences to reach target size.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap_sentences: int = 1):
        """
        Initialize sentence-based chunker.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap_sentences: Number of overlapping sentences
        """
        self.chunk_size = chunk_size
        self.chunk_overlap_sentences = chunk_overlap_sentences

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple regex."""
        # Simple sentence splitting (can be improved with spaCy/NLTK)
        pattern = r'(?<=[.!?])\s+'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, text: str, doc_id: str, metadata: Optional[dict] = None) -> List[Chunk]:
        """Chunk text by sentences."""
        if metadata is None:
            metadata = {}

        sentences = self._split_sentences(text)
        chunks = []
        chunk_num = 0

        current_chunk = []
        current_size = 0
        start_char = 0

        for i, sentence in enumerate(sentences):
            sentence_len = len(sentence)

            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_size += sentence_len

            # Check if we should create a chunk
            if current_size >= self.chunk_size or i == len(sentences) - 1:
                chunk_text = " ".join(current_chunk)
                end_char = start_char + len(chunk_text)

                chunks.append(
                    Chunk(
                        text=chunk_text,
                        doc_id=doc_id,
                        chunk_id=f"{doc_id}_chunk_{chunk_num}",
                        start_char=start_char,
                        end_char=end_char,
                        metadata=metadata.copy(),
                    )
                )

                chunk_num += 1

                # Prepare next chunk with overlap
                if i < len(sentences) - 1:
                    overlap_start = max(0, len(current_chunk) - self.chunk_overlap_sentences)
                    current_chunk = current_chunk[overlap_start:]
                    current_size = sum(len(s) for s in current_chunk)
                    start_char = end_char - sum(len(s) for s in current_chunk)
                else:
                    current_chunk = []
                    current_size = 0

        return chunks


class SemanticChunker(Chunker):
    """
    Semantic chunking based on topic coherence.

    Uses sentence embeddings to detect topic shifts.
    Falls back to sentence chunking if embedding model unavailable.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        similarity_threshold: float = 0.7,
    ):
        """
        Initialize semantic chunker.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            similarity_threshold: Threshold for detecting topic changes
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold

        # For now, fall back to sentence chunking
        # In production, would use sentence-transformers for semantic analysis
        self.fallback_chunker = SentenceChunker(chunk_size=chunk_size)

    def chunk(self, text: str, doc_id: str, metadata: Optional[dict] = None) -> List[Chunk]:
        """
        Chunk text based on semantic coherence.

        Currently uses sentence-based chunking as fallback.
        Can be extended with embedding-based topic detection.
        """
        # TODO: Implement actual semantic chunking with embeddings
        # For now, use sentence-based chunking
        return self.fallback_chunker.chunk(text, doc_id, metadata)


def get_chunker(strategy: str, **kwargs) -> Chunker:
    """
    Factory function to get chunker by strategy name.

    Args:
        strategy: One of "fixed_size", "sentence", "semantic"
        **kwargs: Arguments to pass to chunker constructor

    Returns:
        Chunker instance
    """
    chunkers = {
        "fixed_size": FixedSizeChunker,
        "sentence": SentenceChunker,
        "semantic": SemanticChunker,
    }

    if strategy not in chunkers:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    return chunkers[strategy](**kwargs)


# Example usage
if __name__ == "__main__":
    sample_text = """
    Retrieval-Augmented Generation (RAG) is a technique that enhances large language models
    by retrieving relevant information from a knowledge base. This approach combines the
    benefits of parametric knowledge (stored in model weights) with non-parametric knowledge
    (stored in external documents).

    RAG systems typically consist of two main components: a retriever and a generator.
    The retriever finds relevant documents from a corpus, while the generator uses these
    documents to produce accurate and contextual responses. This architecture enables
    models to access up-to-date information without retraining.

    There are several chunking strategies for RAG systems. Fixed-size chunking splits
    documents into equal-sized pieces. Sentence-based chunking respects sentence boundaries.
    Semantic chunking tries to preserve topical coherence by detecting topic shifts.
    """

    # Test different chunkers
    print("=== Fixed Size Chunker ===")
    fixed_chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=20)
    fixed_chunks = fixed_chunker.chunk(sample_text.strip(), "doc_1")
    for i, chunk in enumerate(fixed_chunks):
        print(f"\nChunk {i + 1} ({len(chunk.text)} chars):")
        print(chunk.text[:100] + "...")

    print("\n\n=== Sentence Chunker ===")
    sentence_chunker = SentenceChunker(chunk_size=300, chunk_overlap_sentences=1)
    sentence_chunks = sentence_chunker.chunk(sample_text.strip(), "doc_1")
    for i, chunk in enumerate(sentence_chunks):
        print(f"\nChunk {i + 1} ({len(chunk.text)} chars):")
        print(chunk.text[:100] + "...")
