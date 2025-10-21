"""Unit tests for PaperRAG vector database interface."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from aust.src.rag.chunking import Chunk, PaperCardChunker
from aust.src.rag.vector_db import PaperRAG, SearchResult


class TestPaperCardChunker:
    """Test suite for PaperCardChunker."""

    @pytest.fixture
    def sample_card_content(self) -> str:
        """Sample paper card markdown content."""
        return """# Sample Paper Title

## Metadata
- **Authors**: John Doe, Jane Smith
- **Venue**: Arxiv
- **Year**: 2025
- **ArXiv ID**: 2505.12345
- **Model Type**: T→I
- **Attack Level**: input_level

## Quick Summary (1-2 sentences)
This is a test paper about adversarial attacks.

## Research Problem
How to attack machine learning models.

## Methodology

### Core Method
The core method involves crafting adversarial examples that fool the model.

### Key Techniques
- Gradient-based optimization
- Perturbation minimization

## Experiment Design

### Datasets
We use MNIST and CIFAR-10 datasets.

### Models Tested
We test ResNet-50 and VGG-16 models.

## Key Results (Summary)

### Main Findings
Our attack achieves 95% success rate on MNIST and 87% on CIFAR-10.

### Performance Highlights
The attack is fast and requires minimal computation.

## Relevance to Our Work

### Application to Data-Based Unlearning
This attack can be used to test unlearning robustness.

### Application to Concept Erasure
The attack helps verify concept erasure effectiveness.
"""

    @pytest.fixture
    def temp_paper_cards_dir(self, sample_card_content):
        """Create temporary paper cards directory with sample card."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cards_dir = tmpdir_path / ".paper_cards" / "any-to-t" / "input_level"
            cards_dir.mkdir(parents=True)

            card_file = cards_dir / "2505.12345.md"
            card_file.write_text(sample_card_content)

            yield tmpdir_path / ".paper_cards"

    def test_chunker_initialization(self, temp_paper_cards_dir):
        """Test PaperCardChunker initialization."""
        chunker = PaperCardChunker(temp_paper_cards_dir)
        assert chunker.paper_cards_dir == temp_paper_cards_dir

    def test_chunker_invalid_directory(self):
        """Test PaperCardChunker with invalid directory."""
        with pytest.raises(ValueError, match="Paper cards directory not found"):
            PaperCardChunker(Path("/nonexistent/path"))

    def test_extract_metadata_from_content(self, temp_paper_cards_dir, sample_card_content):
        """Test metadata extraction from card content."""
        chunker = PaperCardChunker(temp_paper_cards_dir)
        card_path = temp_paper_cards_dir / "any-to-t" / "input_level" / "2505.12345.md"

        metadata = chunker._extract_metadata(sample_card_content, card_path)

        assert metadata["arxiv_id"] == "2505.12345"
        assert metadata["attack_level"] == "input_level"
        assert metadata["task_type"] == "any-to-t"

    def test_extract_title(self, temp_paper_cards_dir, sample_card_content):
        """Test paper title extraction."""
        chunker = PaperCardChunker(temp_paper_cards_dir)
        title = chunker._extract_title(sample_card_content)
        assert title == "Sample Paper Title"

    def test_extract_sections(self, temp_paper_cards_dir, sample_card_content):
        """Test section extraction by H2 headings."""
        chunker = PaperCardChunker(temp_paper_cards_dir)
        sections = chunker._extract_sections(sample_card_content)

        assert "Methodology" in sections
        assert "Experiment Design" in sections
        assert "Key Results (Summary)" in sections
        assert "Relevance to Our Work" in sections

        # Check section content
        assert "Core Method" in sections["Methodology"]
        assert "Gradient-based optimization" in sections["Methodology"]

    def test_chunk_card_extracts_target_sections(self, temp_paper_cards_dir):
        """Test that chunk_card extracts only target sections."""
        chunker = PaperCardChunker(temp_paper_cards_dir)
        card_path = temp_paper_cards_dir / "any-to-t" / "input_level" / "2505.12345.md"

        chunks = chunker.chunk_card(card_path)

        # Should extract 4 chunks: Methodology, Experiment Design (mapped to EXPERIMENTS),
        # Key Results (mapped to RESULTS), Relevance to Our Work (mapped to RELEVANCE)
        assert len(chunks) == 4

        sections = [chunk.section for chunk in chunks]
        assert "METHODOLOGY" in sections
        assert "EXPERIMENTS" in sections
        assert "RESULTS" in sections
        assert "RELEVANCE" in sections

    def test_chunk_text_format(self, temp_paper_cards_dir):
        """Test that chunk text has correct format with section prefix."""
        chunker = PaperCardChunker(temp_paper_cards_dir)
        card_path = temp_paper_cards_dir / "any-to-t" / "input_level" / "2505.12345.md"

        chunks = chunker.chunk_card(card_path)
        methodology_chunk = next(c for c in chunks if c.section == "METHODOLOGY")

        assert methodology_chunk.text.startswith("[METHODOLOGY] Sample Paper Title")
        assert "Core Method" in methodology_chunk.text
        assert "Gradient-based optimization" in methodology_chunk.text

    def test_chunk_metadata_fields(self, temp_paper_cards_dir):
        """Test that chunk has all required metadata fields."""
        chunker = PaperCardChunker(temp_paper_cards_dir)
        card_path = temp_paper_cards_dir / "any-to-t" / "input_level" / "2505.12345.md"

        chunks = chunker.chunk_card(card_path)
        chunk = chunks[0]

        assert chunk.arxiv_id == "2505.12345"
        assert chunk.section in ["METHODOLOGY", "EXPERIMENTS", "RESULTS", "RELEVANCE"]
        assert chunk.task_type == "any-to-t"
        assert chunk.attack_level == "input_level"
        assert chunk.paper_title == "Sample Paper Title"
        assert ".paper_cards" in chunk.card_path
        assert len(chunk.text) >= PaperCardChunker.MIN_CHUNK_LENGTH

    def test_chunk_all_cards(self, temp_paper_cards_dir):
        """Test chunking all cards in directory."""
        chunker = PaperCardChunker(temp_paper_cards_dir)
        all_chunks = chunker.chunk_all_cards()

        # Should have 4 chunks from the single test card
        assert len(all_chunks) >= 4

    def test_skip_short_sections(self, temp_paper_cards_dir):
        """Test that sections shorter than MIN_CHUNK_LENGTH are skipped."""
        # Create a card with a very short section
        short_card = """# Short Paper

## Metadata
- **ArXiv ID**: 2505.99999
- **Attack Level**: input_level

## Methodology
Short.

## Key Results (Summary)
This is a longer section with enough content to pass the minimum chunk length threshold.
We add more text here to ensure it exceeds 50 characters easily.
"""
        cards_dir = temp_paper_cards_dir / "any-to-t" / "input_level"
        card_file = cards_dir / "2505.99999.md"
        card_file.write_text(short_card)

        chunker = PaperCardChunker(temp_paper_cards_dir)
        chunks = chunker.chunk_card(card_file)

        # Should only have RESULTS chunk (Methodology is too short)
        sections = [chunk.section for chunk in chunks]
        assert "METHODOLOGY" not in sections
        assert "RESULTS" in sections


class TestPaperRAG:
    """Test suite for PaperRAG interface."""

    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def mock_embedding_model(self):
        """Mock SentenceTransformers embedding model."""
        mock = Mock()
        mock.get_output_dim.return_value = 384
        mock.embed.return_value = [0.1] * 384  # Mock 384-dim vector
        return mock

    @pytest.fixture
    def mock_storage(self):
        """Mock Qdrant storage."""
        mock = Mock()
        mock.client = Mock()
        return mock

    @pytest.fixture
    def mock_retriever(self):
        """Mock VectorRetriever."""
        mock = Mock()
        return mock

    def test_paper_rag_initialization(self, temp_storage_dir):
        """Test PaperRAG initialization."""
        with patch("aust.src.rag.vector_db.SentenceTransformerEncoder"), \
             patch("aust.src.rag.vector_db.QdrantStorage"), \
             patch("aust.src.rag.vector_db.VectorRetriever"):

            rag = PaperRAG(
                storage_path=temp_storage_dir,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                collection_name="test_collection",
            )

            assert rag.collection_name == "test_collection"
            assert rag.similarity_threshold == PaperRAG.DEFAULT_SIMILARITY_THRESHOLD

    def test_build_filter_section_only(self, temp_storage_dir):
        """Test building filter with section filter only."""
        with patch("aust.src.rag.vector_db.SentenceTransformerEncoder"), \
             patch("aust.src.rag.vector_db.QdrantStorage"), \
             patch("aust.src.rag.vector_db.VectorRetriever"):

            rag = PaperRAG(storage_path=temp_storage_dir)
            filter_obj = rag._build_filter(
                section_filter="METHODOLOGY",
                task_type_filter=None,
            )

            assert filter_obj is not None
            assert len(filter_obj.must) == 1

    def test_build_filter_multiple_conditions(self, temp_storage_dir):
        """Test building filter with multiple conditions."""
        with patch("aust.src.rag.vector_db.SentenceTransformerEncoder"), \
             patch("aust.src.rag.vector_db.QdrantStorage"), \
             patch("aust.src.rag.vector_db.VectorRetriever"):

            rag = PaperRAG(storage_path=temp_storage_dir)
            filter_obj = rag._build_filter(
                section_filter="METHODOLOGY",
                task_type_filter="any-to-t",
            )

            assert filter_obj is not None
            assert len(filter_obj.must) == 2

    def test_build_filter_no_conditions(self, temp_storage_dir):
        """Test building filter with no conditions returns None."""
        with patch("aust.src.rag.vector_db.SentenceTransformerEncoder"), \
             patch("aust.src.rag.vector_db.QdrantStorage"), \
             patch("aust.src.rag.vector_db.VectorRetriever"):

            rag = PaperRAG(storage_path=temp_storage_dir)
            filter_obj = rag._build_filter(
                section_filter=None,
                task_type_filter=None,
            )

            assert filter_obj is None

    def test_parse_retriever_results_valid(self, temp_storage_dir):
        """Test parsing valid retriever results."""
        with patch("aust.src.rag.vector_db.SentenceTransformerEncoder"), \
             patch("aust.src.rag.vector_db.QdrantStorage"), \
             patch("aust.src.rag.vector_db.VectorRetriever"):

            rag = PaperRAG(storage_path=temp_storage_dir)

            mock_results = [
                {
                    "text": "Sample chunk text",
                    "similarity score": "0.85",
                    "metadata": {
                        "arxiv_id": "2505.12345",
                        "section": "METHODOLOGY",
                        "task_type": "any-to-t",
                        "attack_level": "input_level",
                        "paper_title": "Test Paper",
                        "card_path": ".paper_cards/any-to-t/input_level/2505.12345.md",
                    },
                }
            ]

            results = rag._parse_retriever_results(mock_results)

            assert len(results) == 1
            result = results[0]
            assert result.arxiv_id == "2505.12345"
            assert result.section == "METHODOLOGY"
            assert result.similarity_score == 0.85
            assert result.text == "Sample chunk text"

    def test_parse_retriever_results_no_results(self, temp_storage_dir):
        """Test parsing no suitable results message."""
        with patch("aust.src.rag.vector_db.SentenceTransformerEncoder"), \
             patch("aust.src.rag.vector_db.QdrantStorage"), \
             patch("aust.src.rag.vector_db.VectorRetriever"):

            rag = PaperRAG(storage_path=temp_storage_dir)

            mock_results = [
                {"text": "No suitable information retrieved from ..."}
            ]

            results = rag._parse_retriever_results(mock_results)

            assert len(results) == 0


class TestSearchResult:
    """Test suite for SearchResult dataclass."""

    def test_search_result_creation(self):
        """Test creating a SearchResult instance."""
        result = SearchResult(
            arxiv_id="2505.12345",
            section="METHODOLOGY",
            text="Sample text",
            similarity_score=0.85,
            task_type="any-to-t",
            attack_level="input_level",
            paper_title="Test Paper",
            card_path=".paper_cards/any-to-t/input_level/2505.12345.md",
        )

        assert result.arxiv_id == "2505.12345"
        assert result.section == "METHODOLOGY"
        assert result.similarity_score == 0.85


class TestChunk:
    """Test suite for Chunk dataclass."""

    def test_chunk_creation(self):
        """Test creating a Chunk instance."""
        chunk = Chunk(
            arxiv_id="2505.12345",
            section="METHODOLOGY",
            task_type="any-to-t",
            attack_level="input_level",
            paper_title="Test Paper",
            card_path=".paper_cards/any-to-t/input_level/2505.12345.md",
            text="[METHODOLOGY] Test Paper\n\nSample methodology text.",
        )

        assert chunk.arxiv_id == "2505.12345"
        assert chunk.section == "METHODOLOGY"
        assert "[METHODOLOGY]" in chunk.text
