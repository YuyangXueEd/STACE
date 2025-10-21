#!/usr/bin/env python3
"""Build vector index from paper cards to aust/rag_paper_db.

This script:
1. Reads all paper cards from .paper_cards/ directory
2. Chunks each card into semantic sections (Methodology, Experiments, Results, Relevance)
3. Embeds chunks using SentenceTransformers
4. Indexes chunks in Qdrant vector database at aust/rag_paper_db
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

from aust.src.rag.chunking import PaperCardChunker
from aust.src.rag.vector_db import PaperRAG
from camel.storages import VectorRecord

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for vector index building."""
    logger.info("=" * 70)
    logger.info("Building Paper RAG Vector Index")
    logger.info("=" * 70)
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
    logger.info(f"Batch size: {batch_size}")
    logger.info("=" * 70)

    # Verify paper cards directory exists
    if not paper_cards_dir.exists():
        logger.error(f"Paper cards directory not found: {paper_cards_dir}")
        return 1

    # Create storage directory if needed
    storage_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Initialize chunker and extract all chunks
    logger.info("\n[STEP 1/3] Chunking paper cards...")
    chunker = PaperCardChunker(paper_cards_dir)
    all_chunks = chunker.chunk_all_cards()

    if not all_chunks:
        logger.error("No chunks extracted! Check paper cards directory.")
        return 1

    logger.info(f"✓ Extracted {len(all_chunks)} chunks from paper cards")

    # Step 2: Initialize PaperRAG (creates collection if needed)
    logger.info("\n[STEP 2/3] Initializing Qdrant collection...")
    paper_rag = PaperRAG(
        storage_path=str(storage_path),
        embedding_model=embedding_model,
        collection_name=collection_name,
    )
    logger.info("✓ Qdrant collection initialized")

    # Step 3: Index chunks in batches
    logger.info(f"\n[STEP 3/3] Indexing {len(all_chunks)} chunks...")
    logger.info(f"Processing in batches of {batch_size}...")

    failed_chunks = []
    indexed_count = 0

    # Process chunks in batches with progress bar
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Indexing batches"):
        batch = all_chunks[i : i + batch_size]

        # Prepare batch of VectorRecord objects
        vector_records = []

        for chunk in batch:
            try:
                # Build metadata payload
                metadata = {
                    "arxiv_id": chunk.arxiv_id,
                    "section": chunk.section,
                    "task_type": chunk.task_type,
                    "attack_level": chunk.attack_level,
                    "paper_title": chunk.paper_title,
                    "card_path": chunk.card_path,
                    "text": chunk.text,
                }

                # Embed chunk text
                vector = paper_rag.embedding_model.embed(
                    obj=chunk.text, task="retrieval.passage"
                )

                # Create VectorRecord with UUID (Qdrant requires valid UUID)
                record = VectorRecord(
                    id=str(uuid4()),  # Generate unique UUID for each point
                    vector=vector,
                    payload=metadata
                )
                vector_records.append(record)
                indexed_count += 1

            except Exception as e:
                logger.warning(f"Failed to prepare chunk {chunk.arxiv_id}/{chunk.section}: {e}")
                failed_chunks.append((chunk, str(e)))

        # Batch add all records from this batch
        if vector_records:
            try:
                paper_rag.storage.add(records=vector_records)
            except Exception as e:
                logger.error(f"Failed to add batch to storage: {e}")
                # Mark all chunks in this batch as failed
                for record in vector_records:
                    failed_chunks.append((None, f"Batch add failed: {e}"))

    # Summary
    elapsed_time = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("Vector Index Build Complete!")
    logger.info("=" * 70)
    logger.info(f"Total chunks processed:    {len(all_chunks)}")
    logger.info(f"Successfully indexed:      {indexed_count}")
    logger.info(f"Failed chunks:             {len(failed_chunks)}")
    logger.info(f"Processing time:           {elapsed_time:.2f} seconds")
    logger.info(f"Throughput:                {indexed_count / elapsed_time:.2f} chunks/second")
    logger.info("=" * 70)

    if failed_chunks:
        logger.warning(f"\n⚠️  {len(failed_chunks)} chunks failed to index:")
        for chunk, error in failed_chunks[:10]:  # Show first 10 failures
            if chunk is not None:
                logger.warning(f"  - {chunk.arxiv_id}/{chunk.section}: {error}")
            else:
                logger.warning(f"  - {error}")
        if len(failed_chunks) > 10:
            logger.warning(f"  ... and {len(failed_chunks) - 10} more")

    logger.info(f"\n✓ Vector index saved to: {storage_path}")
    logger.info("\nYou can now use PaperRAG to search papers:")
    logger.info("  from aust.src.rag import PaperRAG")
    logger.info(f"  rag = PaperRAG(storage_path='{storage_path}')")
    logger.info("  results = rag.search('your query here', top_k=5)")

    return 0 if len(failed_chunks) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
