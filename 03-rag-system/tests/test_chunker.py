"""
Tests for document chunking strategies.
"""

import pytest
from src.indexing.chunker import (
    FixedSizeChunker,
    SentenceChunker,
    SemanticChunker,
    get_chunker,
)


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
    Retrieval-Augmented Generation (RAG) is a technique that enhances large language models.
    It combines parametric and non-parametric knowledge. RAG systems have two components:
    a retriever and a generator. The retriever finds relevant documents. The generator
    produces responses using those documents.
    """.strip()


class TestFixedSizeChunker:
    """Tests for fixed-size chunking."""

    def test_basic_chunking(self, sample_text):
        """Test basic fixed-size chunking."""
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(sample_text, "doc_1")

        assert len(chunks) > 1
        assert all(c.doc_id == "doc_1" for c in chunks)
        assert all(c.text for c in chunks)  # No empty chunks

    def test_chunk_size_approximately_correct(self, sample_text):
        """Test that chunks are approximately the target size."""
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(sample_text, "doc_1")

        for chunk in chunks[:-1]:  # Except last chunk
            assert 80 <= len(chunk.text) <= 120  # Allow some variance

    def test_chunk_overlap(self, sample_text):
        """Test that chunks have overlap."""
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(sample_text, "doc_1")

        if len(chunks) >= 2:
            # Check that consecutive chunks share some content
            chunk1_end = chunks[0].text[-20:]
            chunk2_start = chunks[1].text[:20]
            # There should be some overlap (not exact due to word boundaries)
            assert len(set(chunk1_end.split()) & set(chunk2_start.split())) > 0

    def test_metadata_preserved(self, sample_text):
        """Test that metadata is preserved in chunks."""
        chunker = FixedSizeChunker(chunk_size=100)
        metadata = {"source": "test", "page": 1}
        chunks = chunker.chunk(sample_text, "doc_1", metadata)

        assert all(c.metadata == metadata for c in chunks)

    def test_chunk_ids_unique(self, sample_text):
        """Test that chunk IDs are unique."""
        chunker = FixedSizeChunker(chunk_size=100)
        chunks = chunker.chunk(sample_text, "doc_1")

        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))  # All unique


class TestSentenceChunker:
    """Tests for sentence-based chunking."""

    def test_sentence_boundaries_respected(self, sample_text):
        """Test that chunks don't break sentences."""
        chunker = SentenceChunker(chunk_size=150)
        chunks = chunker.chunk(sample_text, "doc_1")

        # Each chunk should contain complete sentences
        for chunk in chunks:
            # Should end with sentence-ending punctuation or be last chunk
            assert chunk.text[-1] in ".!?" or chunk == chunks[-1]

    def test_approximate_size(self, sample_text):
        """Test that chunks are approximately target size."""
        chunker = SentenceChunker(chunk_size=150)
        chunks = chunker.chunk(sample_text, "doc_1")

        # Most chunks should be reasonably close to target
        # (allowing variance due to sentence boundaries)
        for chunk in chunks:
            assert 50 <= len(chunk.text) <= 300


class TestSemanticChunker:
    """Tests for semantic chunking."""

    def test_fallback_to_sentence_chunking(self, sample_text):
        """Test that semantic chunker falls back to sentence chunking."""
        chunker = SemanticChunker(chunk_size=150)
        chunks = chunker.chunk(sample_text, "doc_1")

        # Should produce chunks
        assert len(chunks) > 0
        assert all(c.text for c in chunks)


class TestChunkerFactory:
    """Tests for chunker factory function."""

    def test_get_fixed_size_chunker(self):
        """Test getting fixed-size chunker."""
        chunker = get_chunker("fixed_size", chunk_size=256)
        assert isinstance(chunker, FixedSizeChunker)
        assert chunker.chunk_size == 256

    def test_get_sentence_chunker(self):
        """Test getting sentence chunker."""
        chunker = get_chunker("sentence", chunk_size=512)
        assert isinstance(chunker, SentenceChunker)
        assert chunker.chunk_size == 512

    def test_get_semantic_chunker(self):
        """Test getting semantic chunker."""
        chunker = get_chunker("semantic", chunk_size=512)
        assert isinstance(chunker, SemanticChunker)

    def test_unknown_strategy_raises(self):
        """Test that unknown strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            get_chunker("unknown_strategy")


class TestChunkDataclass:
    """Tests for Chunk dataclass."""

    def test_chunk_attributes(self, sample_text):
        """Test that chunks have all required attributes."""
        chunker = FixedSizeChunker(chunk_size=100)
        chunks = chunker.chunk(sample_text, "doc_1", {"source": "test"})

        for chunk in chunks:
            assert hasattr(chunk, "text")
            assert hasattr(chunk, "doc_id")
            assert hasattr(chunk, "chunk_id")
            assert hasattr(chunk, "start_char")
            assert hasattr(chunk, "end_char")
            assert hasattr(chunk, "metadata")

    def test_char_positions_consistent(self, sample_text):
        """Test that character positions are consistent."""
        chunker = FixedSizeChunker(chunk_size=100)
        chunks = chunker.chunk(sample_text, "doc_1")

        for chunk in chunks:
            # Start should be less than end
            assert chunk.start_char < chunk.end_char
            # Length should match approximately
            assert abs((chunk.end_char - chunk.start_char) - len(chunk.text)) < 10
