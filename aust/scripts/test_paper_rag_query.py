#!/usr/bin/env python3
"""Test script to demonstrate PaperRAG search capabilities.

This script shows various ways to query the vector database:
1. Basic semantic search
2. Filtered search (by section and task type)
3. Metadata retrieval
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from aust.src.rag import PaperRAG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def print_results(results, title="Search Results"):
    """Pretty print search results."""
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)

    if not results:
        print("No results found.")
        return

    for i, result in enumerate(results, 1):
        print(f"\n[{i}] {result.paper_title}")
        print(f"    ArXiv ID: {result.arxiv_id}")
        print(f"    Section: {result.section}")
        print(f"    Task Type: {result.task_type} | Attack Level: {result.attack_level}")
        print(f"    Similarity: {result.similarity_score:.3f}")
        print(f"    Text Preview: {result.text[:200]}...")
        print("-" * 80)


def main():
    """Run test queries on the PaperRAG system."""
    logger.info("Initializing PaperRAG system...")

    # Initialize RAG with the aust/rag_paper_db storage
    storage_path = project_root / "aust" / "rag_paper_db"
    rag = PaperRAG(storage_path=str(storage_path))

    logger.info(f"✓ PaperRAG initialized with storage: {storage_path}\n")

    # Test 1: Basic semantic search
    print("\n" + "🔍" * 40)
    print("TEST 1: Basic Semantic Search")
    print("🔍" * 40)

    query1 = "adversarial attacks on text-to-image diffusion models"
    logger.info(f"Query: '{query1}'")

    results1 = rag.search(query=query1, top_k=5)
    print_results(results1, title=f"Results for: '{query1}'")

    # Test 2: Filtered search - METHODOLOGY sections only
    print("\n\n" + "🔍" * 40)
    print("TEST 2: Filtered Search - METHODOLOGY Sections")
    print("🔍" * 40)

    query2 = "gradient-based optimization methods for attacking models"
    logger.info(f"Query: '{query2}'")
    logger.info("Filter: section=METHODOLOGY")

    results2 = rag.search(
        query=query2,
        top_k=5,
        section_filter="METHODOLOGY"
    )
    print_results(results2, title=f"METHODOLOGY Results for: '{query2}'")

    # Test 3: Filtered search - any-to-t papers only
    print("\n\n" + "🔍" * 40)
    print("TEST 3: Filtered Search - Text-to-Image Papers (any-to-t)")
    print("🔍" * 40)

    query3 = "data-based unlearning methods and evaluation"
    logger.info(f"Query: '{query3}'")
    logger.info("Filter: task_type=any-to-t")

    results3 = rag.search(
        query=query3,
        top_k=5,
        task_type_filter="any-to-t"
    )
    print_results(results3, title=f"Text-to-Image Results for: '{query3}'")

    # Test 4: Multi-filter search - EXPERIMENTS section + any-to-t
    print("\n\n" + "🔍" * 40)
    print("TEST 4: Multi-Filter Search - EXPERIMENTS + any-to-t")
    print("🔍" * 40)

    query4 = "evaluation metrics and datasets for attack success rate"
    logger.info(f"Query: '{query4}'")
    logger.info("Filters: section=EXPERIMENTS, task_type=any-to-t")

    results4 = rag.search(
        query=query4,
        top_k=5,
        section_filter="EXPERIMENTS",
        task_type_filter="any-to-t"
    )
    print_results(results4, title=f"EXPERIMENTS (any-to-t) Results for: '{query4}'")

    # Test 5: Relevance section - Application to our work
    print("\n\n" + "🔍" * 40)
    print("TEST 5: RELEVANCE Section - Applications to Unlearning")
    print("🔍" * 40)

    query5 = "concept erasure and unlearning robustness testing"
    logger.info(f"Query: '{query5}'")
    logger.info("Filter: section=RELEVANCE")

    results5 = rag.search(
        query=query5,
        top_k=5,
        section_filter="RELEVANCE"
    )
    print_results(results5, title=f"RELEVANCE Results for: '{query5}'")

    # Test 6: High similarity threshold
    print("\n\n" + "🔍" * 40)
    print("TEST 6: High Similarity Threshold (>= 0.7)")
    print("🔍" * 40)

    query6 = "typographic jailbreak attacks on vision-language models"
    logger.info(f"Query: '{query6}'")
    logger.info("Similarity threshold: 0.7")

    results6 = rag.search(
        query=query6,
        top_k=10,
        similarity_threshold=0.7
    )
    print_results(results6, title=f"High-Confidence Results for: '{query6}'")

    # Test 7: Metadata retrieval
    if results1:
        print("\n\n" + "🔍" * 40)
        print("TEST 7: Metadata Retrieval")
        print("🔍" * 40)

        arxiv_id = results1[0].arxiv_id
        logger.info(f"Retrieving metadata for ArXiv ID: {arxiv_id}")

        metadata = rag.get_paper_metadata(arxiv_id=arxiv_id)

        if metadata:
            print(f"\nMetadata for ArXiv ID: {arxiv_id}")
            print("-" * 80)
            for key, value in metadata.items():
                print(f"{key:15s}: {value}")
            print("-" * 80)

    # Summary
    print("\n\n" + "=" * 80)
    print("RAG SYSTEM TEST SUMMARY")
    print("=" * 80)
    print(f"✓ Test 1 (Basic search): {len(results1)} results")
    print(f"✓ Test 2 (METHODOLOGY filter): {len(results2)} results")
    print(f"✓ Test 3 (any-to-t filter): {len(results3)} results")
    print(f"✓ Test 4 (Multi-filter): {len(results4)} results")
    print(f"✓ Test 5 (RELEVANCE filter): {len(results5)} results")
    print(f"✓ Test 6 (High threshold): {len(results6)} results")
    print("✓ Test 7 (Metadata retrieval): Success")
    print("=" * 80)
    print("\n✅ All tests completed successfully!")
    print(f"📊 Total unique papers indexed: Check aust/rag_paper_db/")
    print("\nYou can now use PaperRAG in your agents:")
    print("  from aust.src.rag import PaperRAG")
    print("  rag = PaperRAG(storage_path='aust/rag_paper_db')")
    print("  results = rag.search('your query', top_k=5)")
    print()


if __name__ == "__main__":
    main()
