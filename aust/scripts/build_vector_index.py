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
    logger.info("Starting vector index build")
    start_time = time.time()

    # Configuration
    paper_cards_dir = project_root / ".paper_cards"
    storage_path = project_root / "aust" / "src" / "rag" / "vector_index"
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

    # Step 2: Initialize PaperRAG (creates collection if needed)
    logger.info("Step 2: Initializing Qdrant collection...")
    paper_rag = PaperRAG(
        storage_path=str(storage_path),
        embedding_model=embedding_model,
        collection_name=collection_name,
    )

    # Step 3: Index chunks in batches
    logger.info(f"Step 3: Indexing {len(all_chunks)} chunks in batches of {batch_size}...")

    failed_chunks = []
    indexed_count = 0

    # Process chunks in batches with progress bar
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Indexing batches"):
        batch = all_chunks[i : i + batch_size]

        # Prepare batch for CAMEL VectorRetriever
        # VectorRetriever.process() expects either a string content or list of UnstructuredElement
        # For our use case, we need to add chunks to storage directly

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

                # Embed and add to storage
                # Use the storage client directly for more control
                vector = paper_rag.embedding_model.embed(
                    obj=chunk.text, task="retrieval.passage"
                )

                # Add to Qdrant storage
                paper_rag.storage.add(
                    objs=[chunk.text],
                    ids=[f"{chunk.arxiv_id}_{chunk.section}_{indexed_count}"],
                    metadata_list=[metadata],
                    embeddings=[vector],
                )

                indexed_count += 1

            except Exception as e:
                logger.warning(f"Failed to index chunk {chunk.arxiv_id}/{chunk.section}: {e}")
                failed_chunks.append((chunk, str(e)))

    # Step 4: Summary
    elapsed_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Vector Index Build Complete")
    logger.info("=" * 60)
    logger.info(f"Total chunks processed: {len(all_chunks)}")
    logger.info(f"Successfully indexed: {indexed_count}")
    logger.info(f"Failed chunks: {len(failed_chunks)}")
    logger.info(f"Processing time: {elapsed_time:.2f} seconds")
    logger.info(f"Throughput: {indexed_count / elapsed_time:.2f} chunks/second")
    logger.info("=" * 60)

    if failed_chunks:
        logger.warning("Failed chunks:")
        for chunk, error in failed_chunks[:10]:  # Show first 10 failures
            logger.warning(f"  - {chunk.arxiv_id}/{chunk.section}: {error}")
        if len(failed_chunks) > 10:
            logger.warning(f"  ... and {len(failed_chunks) - 10} more")

    logger.info(f"Vector index saved to: {storage_path}")

    return 0 if len(failed_chunks) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
