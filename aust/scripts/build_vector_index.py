#!/usr/bin/env python3
"""Build vector index from paper cards for semantic search.

This script follows CAMEL's VectorRetriever pattern:
1. Chunks paper cards into semantic sections
2. Uses PaperRAG.add_chunks() for embedding and indexing
3. Creates multiple collections: aggregate + per-task-type
4. No manual embedding or VectorRecord creation needed

Collections created:
- aust_papers: All papers (aggregate)
- aust_papers_any_to_v: Text-to-image/video papers
- aust_papers_any_to_t: Text generation papers

Usage:
    python aust/scripts/build_vector_index.py
"""

import logging
import sys
import time
from pathlib import Path

from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from aust.src.rag.chunking import PaperCardChunker
from aust.src.rag.vector_db import PaperRAG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for vector index building."""
    logger.info("Starting vector index build (CAMEL VectorRetriever pattern)")
    start_time = time.time()

    # Configuration
    paper_cards_dir = project_root / ".paper_cards"
    storage_path = project_root / "aust" / "rag_paper_db"
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    collection_name = "aust_papers"

    logger.info(f"Paper cards directory: {paper_cards_dir}")
    logger.info(f"Vector storage: {storage_path}")
    logger.info(f"Embedding model: {embedding_model}")
    logger.info(f"Collection name: {collection_name}")

    # Step 1: Chunk all paper cards
    logger.info("Step 1: Chunking paper cards...")
    chunker = PaperCardChunker(paper_cards_dir)
    all_chunks = chunker.chunk_all_cards()

    if not all_chunks:
        logger.error("No chunks extracted! Exiting.")
        return 1

    logger.info(f"Extracted {len(all_chunks)} chunks from paper cards")

    # Step 2: Organize chunks by task type
    chunks_by_task = {}
    for chunk in all_chunks:
        task_type = chunk.task_type or "unknown"
        chunks_by_task.setdefault(task_type, []).append(chunk)

    logger.info(f"Organized into {len(chunks_by_task)} task types:")
    for task_type, task_chunks in chunks_by_task.items():
        logger.info(f"  - {task_type}: {len(task_chunks)} chunks")

    # Step 3: Build collections (aggregate + per-task)
    collections_built = []

    # Build aggregate collection
    logger.info("\nStep 3a: Building aggregate collection...")
    aggregate_rag = PaperRAG(
        storage_path=str(storage_path),
        embedding_model=embedding_model,
        collection_name=collection_name,
    )

    # Convert chunks to format expected by add_chunks()
    chunk_dicts = []
    for chunk in tqdm(all_chunks, desc="Preparing chunks"):
        chunk_dict = {
            "text": chunk.text,
            "metadata": {
                "arxiv_id": chunk.arxiv_id,
                "section": chunk.section,
                "task_type": chunk.task_type,
                "attack_level": chunk.attack_level,
                "model_type": chunk.model_type,
                "paper_title": chunk.paper_title,
                "card_path": chunk.card_path,
            },
        }
        chunk_dicts.append(chunk_dict)

    # Add to aggregate collection using new method
    aggregate_rag.add_chunks(chunk_dicts)
    collections_built.append((collection_name, len(chunk_dicts)))

    # Build per-task collections
    logger.info("\nStep 3b: Building per-task collections...")
    for task_type, task_chunks in chunks_by_task.items():
        normalized = task_type.replace("-", "_")
        task_collection = f"{collection_name}_{normalized}"

        logger.info(f"Building collection '{task_collection}'...")
        task_rag = PaperRAG(
            storage_path=str(storage_path),
            embedding_model=embedding_model,
            collection_name=task_collection,
        )

        # Convert task chunks
        task_chunk_dicts = []
        for chunk in task_chunks:
            task_chunk_dict = {
                "text": chunk.text,
                "metadata": {
                    "arxiv_id": chunk.arxiv_id,
                    "section": chunk.section,
                    "task_type": chunk.task_type,
                    "attack_level": chunk.attack_level,
                    "model_type": chunk.model_type,
                    "paper_title": chunk.paper_title,
                    "card_path": chunk.card_path,
                },
            }
            task_chunk_dicts.append(task_chunk_dict)

        task_rag.add_chunks(task_chunk_dicts)
        collections_built.append((task_collection, len(task_chunk_dicts)))

    # Step 4: Summary
    elapsed_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Vector Index Build Complete")
    logger.info("=" * 60)
    logger.info(f"Total chunks processed: {len(all_chunks)}")
    logger.info(f"Collections built: {len(collections_built)}")

    for coll_name, chunk_count in collections_built:
        info = aggregate_rag.get_collection_info() if coll_name == collection_name else None
        if info and "vectors_count" in info:
            logger.info(f"  - {coll_name}: {chunk_count} chunks, {info['vectors_count']} vectors")
        else:
            logger.info(f"  - {coll_name}: {chunk_count} chunks")

    logger.info(f"Processing time: {elapsed_time:.2f} seconds")
    total_chunks = sum(count for _, count in collections_built)
    logger.info(f"Throughput: {total_chunks / elapsed_time:.2f} chunks/second")
    logger.info(f"Vector index saved to: {storage_path}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
