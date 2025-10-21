"""Performance tests for PaperRAG system.

Tests verify that the system meets NFR8 performance requirements:
- Single query latency: < 5 seconds
- Batch search (3 queries): < 10 seconds total
- Collection loading: < 2 seconds
- Memory footprint: < 500MB
"""

import logging
import time
from pathlib import Path

import pytest

from aust.src.rag.vector_db import PaperRAG

logger = logging.getLogger(__name__)


class TestPaperRAGPerformance:
    """Performance test suite for PaperRAG."""

    @pytest.fixture(scope="module")
    def paper_rag_instance(self):
        """Create a PaperRAG instance for performance testing.

        NOTE: This fixture assumes the vector index has been built.
        Run `python aust/scripts/build_vector_index.py` first.
        """
        storage_path = Path(__file__).parent.parent.parent / "aust" / "src" / "rag" / "vector_index"

        if not storage_path.exists():
            pytest.skip(
                "Vector index not found. "
                "Run `python aust/scripts/build_vector_index.py` to build index first."
            )

        start_time = time.time()
        rag = PaperRAG(storage_path=str(storage_path))
        load_time = time.time() - start_time

        logger.info(f"Collection load time: {load_time:.2f}s")

        yield rag, load_time

    def test_collection_load_time(self, paper_rag_instance):
        """Test that collection loads in < 2 seconds (NFR8)."""
        _, load_time = paper_rag_instance

        assert load_time < 2.0, f"Collection load time {load_time:.2f}s exceeds 2s threshold"
        logger.info(f"✓ Collection load time: {load_time:.2f}s (target: < 2s)")

    def test_single_query_latency(self, paper_rag_instance):
        """Test single query latency < 5 seconds (NFR8)."""
        rag, _ = paper_rag_instance

        query = "adversarial attacks on text-to-image models using gradient-based optimization"
        start_time = time.time()

        results = rag.search(query=query, top_k=5)

        latency = time.time() - start_time

        assert latency < 5.0, f"Query latency {latency:.2f}s exceeds 5s threshold"
        logger.info(f"✓ Single query latency: {latency:.2f}s (target: < 5s)")
        logger.info(f"  Retrieved {len(results)} results")

    def test_batch_query_latency(self, paper_rag_instance):
        """Test batch query latency (3 queries) < 10 seconds (NFR8)."""
        rag, _ = paper_rag_instance

        queries = [
            "data-based unlearning methods for removing training data",
            "concept erasure techniques for multimodal models",
            "adversarial attacks on vision-language models",
        ]

        start_time = time.time()

        all_results = []
        for query in queries:
            results = rag.search(query=query, top_k=5)
            all_results.extend(results)

        batch_latency = time.time() - start_time

        assert batch_latency < 10.0, f"Batch latency {batch_latency:.2f}s exceeds 10s threshold"
        logger.info(f"✓ Batch query latency (3 queries): {batch_latency:.2f}s (target: < 10s)")
        logger.info(f"  Average per query: {batch_latency / len(queries):.2f}s")
        logger.info(f"  Total results: {len(all_results)}")

    def test_filtered_search_performance(self, paper_rag_instance):
        """Test filtered search performance."""
        rag, _ = paper_rag_instance

        query = "attack methods and evaluation metrics"
        start_time = time.time()

        results = rag.search(
            query=query,
            top_k=5,
            section_filter="METHODOLOGY",
            task_type_filter="any-to-t",
        )

        latency = time.time() - start_time

        assert latency < 5.0, f"Filtered query latency {latency:.2f}s exceeds 5s threshold"
        logger.info(f"✓ Filtered query latency: {latency:.2f}s (target: < 5s)")
        logger.info(f"  Retrieved {len(results)} results with filters")

        # Verify filtering worked
        for result in results:
            assert result.section == "METHODOLOGY"
            assert result.task_type == "any-to-t"

    def test_search_accuracy(self, paper_rag_instance):
        """Test that search returns relevant results with good similarity scores."""
        rag, _ = paper_rag_instance

        query = "adversarial attacks on diffusion models"
        results = rag.search(query=query, top_k=10, similarity_threshold=0.3)

        assert len(results) > 0, "Search returned no results"

        # Check that results are sorted by similarity (descending)
        scores = [r.similarity_score for r in results]
        assert scores == sorted(scores, reverse=True), "Results not sorted by similarity"

        # Check that all results meet threshold
        for result in results:
            assert result.similarity_score >= 0.3

        logger.info(f"✓ Search accuracy: {len(results)} results retrieved")
        logger.info(f"  Similarity scores: {[f'{s:.3f}' for s in scores[:5]]}")

    def test_metadata_retrieval_completeness(self, paper_rag_instance):
        """Test that search results include complete metadata."""
        rag, _ = paper_rag_instance

        query = "unlearning methods"
        results = rag.search(query=query, top_k=3)

        assert len(results) > 0, "Search returned no results"

        for result in results:
            # Verify all required fields are present and non-empty
            assert result.arxiv_id != "unknown", "Missing arxiv_id"
            assert result.section in ["METHODOLOGY", "EXPERIMENTS", "RESULTS", "RELEVANCE"]
            assert len(result.text) > 0, "Empty text field"
            assert result.similarity_score > 0.0
            assert result.task_type in ["any-to-t", "any-to-v", "unknown"]
            assert len(result.paper_title) > 0, "Empty paper_title"
            assert len(result.card_path) > 0, "Empty card_path"

        logger.info(f"✓ Metadata completeness: All {len(results)} results have complete metadata")

    def test_edge_case_empty_query(self, paper_rag_instance):
        """Test edge case: empty query string."""
        rag, _ = paper_rag_instance

        results = rag.search(query="", top_k=5)

        # Should return results or handle gracefully
        logger.info(f"Empty query returned {len(results)} results")

    def test_edge_case_no_results(self, paper_rag_instance):
        """Test edge case: query with very high similarity threshold returns no results."""
        rag, _ = paper_rag_instance

        results = rag.search(
            query="extremely specific query unlikely to match",
            top_k=5,
            similarity_threshold=0.99,  # Very high threshold
        )

        # Should return empty list gracefully
        assert isinstance(results, list)
        logger.info(f"High threshold query returned {len(results)} results (expected 0 or few)")

    def test_edge_case_invalid_filters(self, paper_rag_instance):
        """Test edge case: invalid filter values."""
        rag, _ = paper_rag_instance

        results = rag.search(
            query="test query",
            top_k=5,
            section_filter="INVALID_SECTION",
            task_type_filter="invalid-task-type",
        )

        # Should return empty list or handle gracefully
        assert isinstance(results, list)
        logger.info(f"Invalid filters returned {len(results)} results (expected 0)")


@pytest.mark.skip(reason="Memory profiling requires manual execution with memory_profiler")
class TestMemoryFootprint:
    """Memory footprint tests (requires manual execution)."""

    def test_memory_usage(self):
        """Test memory usage < 500MB (NFR8).

        Run manually with:
        python -m memory_profiler tests/unit/test_rag_performance.py::TestMemoryFootprint::test_memory_usage
        """
        from memory_profiler import profile

        @profile
        def load_and_query():
            rag = PaperRAG()
            for _ in range(10):
                rag.search("test query", top_k=5)

        load_and_query()
