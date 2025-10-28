"""Vector database interface for paper card retrieval using CAMEL's VectorRetriever pattern.

This module follows CAMEL's RAG cookbook approach while keeping:
- SentenceTransformers for embedding (local, fast)
- OpenRouter for LLM queries (when needed)
- Qdrant for vector storage
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from camel.embeddings import SentenceTransformerEncoder
from camel.retrievers import VectorRetriever
from camel.storages import QdrantStorage

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a search result from the vector database.

    Attributes:
        arxiv_id: ArXiv identifier
        section: Section type (FULL_PAPER or legacy sections)
        text: Full chunk text
        similarity_score: Cosine similarity score (0-1)
        task_type: Task taxonomy (any-to-t, any-to-v)
        attack_level: Attack taxonomy level
        model_type: Model modality tag from metadata (e.g., "T->I")
        paper_title: Full paper title
        card_path: Relative path to paper card
        metadata: Full metadata dict from vector DB
    """

    arxiv_id: str
    section: str
    text: str
    similarity_score: float
    task_type: str
    attack_level: str
    model_type: str
    paper_title: str
    card_path: str
    metadata: Dict[str, Any]


class PaperRAG:
    """Vector-based RAG following CAMEL's VectorRetriever pattern.

    Follows the approach from CAMEL's RAG cookbook:
    - Uses VectorRetriever with QdrantStorage
    - SentenceTransformers for local embedding
    - Simple query() interface with metadata filtering

    Example:
        >>> rag = PaperRAG()
        >>> results = rag.search("diffusion models", top_k=5)
    """

    DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_COLLECTION_NAME = "aust_papers"
    DEFAULT_SIMILARITY_THRESHOLD = 0.5

    def __init__(
        self,
        storage_path: str = "aust/rag_paper_db",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ):
        """Initialize PaperRAG with CAMEL's VectorRetriever.

        Args:
            storage_path: Path to Qdrant local storage directory
            embedding_model: SentenceTransformers model name
            collection_name: Qdrant collection name
            similarity_threshold: Minimum similarity score for results (0-1)
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.embedding_model_name = embedding_model
        self.collection_name = collection_name
        self.similarity_threshold = similarity_threshold

        logger.info(
            f"Initializing PaperRAG with embedding: {embedding_model}, "
            f"storage: {storage_path}, collection: {collection_name}"
        )

        # Create embedding model (SentenceTransformers via CAMEL)
        self.embedding_model = SentenceTransformerEncoder(model_name=embedding_model)
        vector_dim = self.embedding_model.get_output_dim()
        logger.info(f"Embedding dimension: {vector_dim}")

        # Create storage (Qdrant via CAMEL)
        self.storage = QdrantStorage(
            vector_dim=vector_dim,
            path=str(self.storage_path),
            collection_name=collection_name,
        )

        # Create retriever (CAMEL's VectorRetriever)
        self.retriever = VectorRetriever(
            embedding_model=self.embedding_model,
            storage=self.storage,
        )

        logger.info("PaperRAG initialized successfully")

    def search(
        self,
        query: str,
        top_k: int = 5,
        section_filter: Optional[str] = None,
        task_type_filter: Optional[str] = None,
        model_type_filter: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Search for relevant papers using CAMEL's embedding + Qdrant storage.

        Uses CAMEL's embedding model but accesses Qdrant storage directly
        to properly retrieve our custom metadata structure.

        Args:
            query: Query string for semantic search
            top_k: Number of results to return
            section_filter: Filter by section type (FULL_PAPER, etc.)
            task_type_filter: Filter by task type (any-to-t, any-to-v)
            model_type_filter: Filter by model modality (e.g., T->I, T->V)
            similarity_threshold: Override default similarity threshold

        Returns:
            List of SearchResult objects sorted by similarity score
        """
        threshold = similarity_threshold or self.similarity_threshold

        logger.info(
            f"Searching query='{query}', top_k={top_k}, "
            f"filters: section={section_filter}, task={task_type_filter}, "
            f"model={model_type_filter}, threshold={threshold}"
        )

        try:
            # Use CAMEL's embedding model to embed the query
            query_vector = self.embedding_model.embed(obj=query, task="retrieval.query")

            # Access Qdrant storage directly to get full payload
            from camel.storages import VectorDBQuery
            db_query = VectorDBQuery(query_vector=query_vector, top_k=top_k)
            query_results = self.storage.query(query=db_query)

            # Parse Qdrant results (these have full payload)
            results = self._parse_storage_results(
                query_results,
                threshold=threshold,
                section_filter=section_filter,
                task_type_filter=task_type_filter,
                model_type_filter=model_type_filter,
            )

            logger.info(f"Found {len(results)} results after filtering")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return []

    def _parse_storage_results(
        self,
        query_results: List[Any],
        threshold: float,
        section_filter: Optional[str] = None,
        task_type_filter: Optional[str] = None,
        model_type_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """Parse CAMEL QdrantStorage.query() results and apply filters.

        Args:
            query_results: List of VectorDBQueryResult from storage.query()
            threshold: Similarity threshold to filter results
            section_filter: Filter by section
            task_type_filter: Filter by task type
            model_type_filter: Filter by model type

        Returns:
            Filtered list of SearchResult objects
        """
        results = []

        for result in query_results:
            # Check similarity threshold
            if result.similarity < threshold:
                continue

            # Extract payload (our custom metadata)
            payload = result.record.payload
            if not payload:
                logger.warning(f"Empty payload for result ID: {result.record.id}")
                continue

            # Extract metadata directly from payload
            arxiv_id = payload.get("arxiv_id", "unknown")
            section = payload.get("section", "unknown")
            task_type = payload.get("task_type", "unknown")
            attack_level = payload.get("attack_level", "unknown")
            model_type = payload.get("model_type", "unknown")
            paper_title = payload.get("paper_title", "unknown")
            card_path = payload.get("card_path", "unknown")
            text = payload.get("text", "")

            # Apply metadata filters
            if section_filter and section != section_filter:
                continue
            if task_type_filter and task_type != task_type_filter:
                continue
            if model_type_filter and model_type != model_type_filter:
                continue

            # Create SearchResult
            search_result = SearchResult(
                arxiv_id=arxiv_id,
                section=section,
                text=text,
                similarity_score=float(result.similarity),
                task_type=task_type,
                attack_level=attack_level,
                model_type=model_type,
                paper_title=paper_title,
                card_path=card_path,
                metadata=payload,
            )
            results.append(search_result)

        return results

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Add paper chunks to the vector database using CAMEL's process() pattern.

        This follows CAMEL's RAG cookbook approach for adding content.

        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata' keys

        Example:
            >>> chunks = [
            ...     {
            ...         "text": "Paper content...",
            ...         "metadata": {
            ...             "arxiv_id": "2505.11842",
            ...             "section": "FULL_PAPER",
            ...             "task_type": "any-to-v",
            ...             ...
            ...         }
            ...     }
            ... ]
            >>> rag.add_chunks(chunks)
        """
        logger.info(f"Adding {len(chunks)} chunks to vector database")

        for chunk in chunks:
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})

            if not text:
                logger.warning("Skipping chunk with empty text")
                continue

            # Use CAMEL's VectorRetriever to store content
            # This handles embedding and storage automatically
            self.storage.save([text], [metadata])

        logger.info(f"Successfully added {len(chunks)} chunks")

    def get_paper_metadata(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a specific paper by ArXiv ID.

        Uses Qdrant scroll to find exact matches by arxiv_id field.

        Args:
            arxiv_id: ArXiv identifier (e.g., "2505.11842")

        Returns:
            Dictionary with paper metadata or None if not found
        """
        try:
            # Use Qdrant scroll with filter for exact arxiv_id match
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            client = self.storage.client
            scroll_result = client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="arxiv_id",
                            match=MatchValue(value=arxiv_id),
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )

            points = scroll_result[0]
            if not points:
                logger.warning(f"No paper found with ArXiv ID: {arxiv_id}")
                return None

            # Extract metadata from first matching point
            payload = points[0].payload
            return {
                "arxiv_id": payload.get("arxiv_id"),
                "paper_title": payload.get("paper_title"),
                "task_type": payload.get("task_type"),
                "attack_level": payload.get("attack_level"),
                "model_type": payload.get("model_type"),
                "card_path": payload.get("card_path"),
            }

        except Exception as e:
            logger.error(f"Failed to get metadata for {arxiv_id}: {e}")
            return None

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection.

        Returns:
            Dictionary with collection stats (name, vector count, etc.)
        """
        try:
            client = self.storage.client
            collection_info = client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "status": collection_info.status,
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}
