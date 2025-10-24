#!/usr/bin/env python3
"""Build vector index from paper cards for semantic search.

This script:
1. Reads all paper cards from .paper_cards/ directory
2. Chunks each card into semantic sections (Methodology, Experiments, Results, Relevance)
3. Embeds chunks using SentenceTransformers
4. Indexes chunks in Qdrant vector database
"""

import logging
import sys
import time
from pathlib import Path
from uuid import uuid4

from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from camel.storages import VectorRecord

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
    logger.info("Starting vector index build")
    start_time = time.time()

    # Configuration
    paper_cards_dir = project_root / ".paper_cards"
    storage_path = project_root / "aust" / "rag_paper_db"
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    collection_name = "aust_papers"
    batch_size = 50  # Upsert 50 chunks per batch

    logger.info(f"Paper cards directory: {paper_cards_dir}")
    logger.info(f"Vector index storage: {storage_path}")
    logger.info(f"Embedding model: {embedding_model}")
    logger.info(f"Collection name: {collection_name}")

    # Step 1: Initialize chunker and extract all chunks
    logger.info("Step 1: Chunking paper cards...")
    chunker = PaperCardChunker(paper_cards_dir)
    all_chunks = chunker.chunk_all_cards()

    if not all_chunks:
        logger.error("No chunks extracted! Exiting.")
        return 1

    logger.info(f"Extracted {len(all_chunks)} chunks from paper cards")

    # Organize chunks by task type for separate collections
    chunks_by_task: dict[str, list] = {}
    for chunk in all_chunks:
        task_type = chunk.task_type or "unknown"
        chunks_by_task.setdefault(task_type, []).append(chunk)

    # Helper to build an individual collection
    def build_collection(target_collection: str, chunks: list) -> tuple[int, list]:
        logger.info(f"Initializing collection '{target_collection}' with {len(chunks)} chunks")
        paper_rag = PaperRAG(
            storage_path=str(storage_path),
            embedding_model=embedding_model,
            collection_name=target_collection,
        )

        failed: list = []
        indexed = 0

        for i in tqdm(range(0, len(chunks), batch_size), desc=f"Indexing {target_collection}"):
            batch = chunks[i : i + batch_size]
            vector_records: list[VectorRecord] = []
            record_chunks: list = []

            for chunk in batch:
                try:
                    metadata = {
                        "arxiv_id": chunk.arxiv_id,
                        "section": chunk.section,
                        "task_type": chunk.task_type,
                        "attack_level": chunk.attack_level,
                        "model_type": chunk.model_type,
                        "paper_title": chunk.paper_title,
                        "card_path": chunk.card_path,
                        "text": chunk.text,
                    }

                    embedding = paper_rag.embedding_model.embed(
                        obj=chunk.text, task="retrieval.passage"
                    )
                    if hasattr(embedding, "tolist"):
                        embedding = embedding.tolist()

                    record = VectorRecord(
                        id=str(uuid4()),
                        vector=embedding,
                        payload=metadata,
                    )
                    vector_records.append(record)
                    record_chunks.append(chunk)
                except Exception as exc:
                    logger.warning(
                        "Failed to prepare chunk %s/%s for %s: %s",
                        chunk.arxiv_id,
                        chunk.section,
                        target_collection,
                        exc,
                    )
                    failed.append((chunk, str(exc)))

            if not vector_records:
                continue

            try:
                paper_rag.storage.add(records=vector_records)
                indexed += len(vector_records)
            except Exception as exc:
                logger.error(
                    "Failed to add batch to %s: %s", target_collection, exc
                )
                for chunk in record_chunks:
                    failed.append((chunk, f"Batch add failed: {exc}"))

        return indexed, failed

    # Step 2: Build aggregate collection (legacy default)
    logger.info("Step 2: Building aggregate collection")
    aggregate_indexed, aggregate_failed = build_collection(collection_name, all_chunks)

    # Step 3: Build per-task collections
    per_task_results: dict[str, tuple[int, list]] = {}
    logger.info("Step 3: Building per-task collections")
    for task_type, task_chunks in chunks_by_task.items():
        normalized = task_type.replace("-", "_")
        task_collection = f"{collection_name}_{normalized}"
        per_task_results[task_type] = build_collection(task_collection, task_chunks)

    # Step 4: Summary
    elapsed_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Vector Index Build Complete")
    logger.info("=" * 60)
    logger.info(f"Total chunks processed: {len(all_chunks)}")
    logger.info(
        "Aggregate collection '%s': %s indexed, %s failed",
        collection_name,
        aggregate_indexed,
        len(aggregate_failed),
    )
    for task_type, (indexed_count, failed) in per_task_results.items():
        logger.info(
            "Collection '%s_%s': %s indexed, %s failed",
            collection_name,
            task_type.replace("-", "_"),
            indexed_count,
            len(failed),
        )

    logger.info(f"Processing time: {elapsed_time:.2f} seconds")
    total_indexed = aggregate_indexed + sum(indexed for indexed, _ in per_task_results.values())
    logger.info(f"Throughput: {total_indexed / elapsed_time:.2f} chunks/second")
    logger.info("=" * 60)

    all_failures = aggregate_failed[:]
    for _, (_, failures) in per_task_results.items():
        all_failures.extend(failures)

    if all_failures:
        logger.warning("Failed chunks (union across collections):")
        for chunk, error in all_failures[:10]:
            logger.warning(f"  - {chunk.arxiv_id}/{chunk.section}: {error}")
        if len(all_failures) > 10:
            logger.warning(f"  ... and {len(all_failures) - 10} more")

    logger.info(f"Vector index saved to: {storage_path}")

    return 0 if not all_failures else 1


if __name__ == "__main__":
    sys.exit(main())
