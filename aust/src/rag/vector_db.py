"""Vector database interface for paper card retrieval using CAMEL + Qdrant."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from camel.embeddings import SentenceTransformerEncoder
from camel.retrievers import VectorRetriever
from camel.storages import QdrantStorage
from qdrant_client.models import Filter, FieldCondition, MatchValue

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a search result from the vector database.

    Attributes:
        arxiv_id: ArXiv identifier
        section: Section type (METHODOLOGY, EXPERIMENTS, RESULTS, RELEVANCE)
        text: Full chunk text
        similarity_score: Cosine similarity score (0-1)
        task_type: Task taxonomy (any-to-t, any-to-v)
        attack_level: Attack taxonomy level
        model_type: Model modality tag from metadata (e.g., "T→I")
        paper_title: Full paper title
        card_path: Relative path to paper card
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


class PaperRAG:
    """Vector-based RAG interface for semantic search over paper cards.

    Uses CAMEL's VectorRetriever with Qdrant storage and SentenceTransformers
    embeddings for fast, local semantic search.
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
        """Initialize PaperRAG with existing or new collection.

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
            f"Initializing PaperRAG with embedding model: {embedding_model}, "
            f"storage: {storage_path}, collection: {collection_name}"
        )

        # Initialize SentenceTransformers embedding via CAMEL
        self.embedding_model = SentenceTransformerEncoder(model_name=embedding_model)

        # Get embedding dimension
        vector_dim = self.embedding_model.get_output_dim()
        logger.info(f"Embedding dimension: {vector_dim}")

        # Initialize Qdrant storage via CAMEL
        self.storage = QdrantStorage(
            vector_dim=vector_dim,
            path=str(self.storage_path),
            collection_name=collection_name,
        )

        # Initialize VectorRetriever
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
        """Search for relevant paper chunks by semantic similarity.

        Args:
            query: Query string for semantic search
            top_k: Number of results to return
            section_filter: Filter by section type (METHODOLOGY, EXPERIMENTS, etc.)
            task_type_filter: Filter by task type (any-to-t, any-to-v)
            model_type_filter: Filter by model modality (e.g., T→I)
            similarity_threshold: Override default similarity threshold

        Returns:
            List of SearchResult objects sorted by similarity
        """
        threshold = similarity_threshold or self.similarity_threshold

        logger.info(
            f"Searching with query='{query[:50]}...', top_k={top_k}, "
            f"section_filter={section_filter}, task_type_filter={task_type_filter}, "
            f"model_type_filter={model_type_filter}, threshold={threshold}"
        )

        # Build Qdrant filter conditions
        filter_conditions = self._build_filter(
            section_filter, task_type_filter, model_type_filter
        )

        # Always use direct Qdrant client for better control over metadata
        try:
            results = self._search_with_filter(
                query, top_k, threshold, filter_conditions
            )

            logger.info(f"Found {len(results)} results above threshold {threshold}")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def _search_with_filter(
        self,
        query: str,
        top_k: int,
        threshold: float,
        filter_conditions: Optional[Filter],
    ) -> List[SearchResult]:
        """Execute filtered search using Qdrant client directly.

        Args:
            query: Query string
            top_k: Number of results
            threshold: Similarity threshold
            filter_conditions: Qdrant filter object (or None for no filter)

        Returns:
            List of SearchResult objects
        """
        # Embed the query
        query_vector = self.embedding_model.embed(obj=query, task="retrieval.query")

        # Search with filter using underlying Qdrant client
        # Note: Access the private _client attribute from CAMEL's QdrantStorage
        qdrant_client = self.storage._client

        # Build search parameters
        search_params = {
            "collection_name": self.collection_name,
            "query_vector": query_vector,
            "limit": top_k,
            "score_threshold": threshold,
            "with_payload": True,  # Explicitly request payloads
        }

        # Only add filter if it's not None
        if filter_conditions is not None:
            search_params["query_filter"] = filter_conditions

        search_results = qdrant_client.search(**search_params)

        # Parse Qdrant results
        results = []
        for hit in search_results:
            payload = hit.payload if hit.payload else {}

            # Debug: Log if payload is empty
            if not payload:
                logger.warning(f"Empty payload for hit ID: {hit.id}")

            result = SearchResult(
                arxiv_id=payload.get("arxiv_id", "unknown"),
                section=payload.get("section", "unknown"),
                text=payload.get("text", ""),
                similarity_score=float(hit.score),
                task_type=payload.get("task_type", "unknown"),
                attack_level=payload.get("attack_level", "unknown"),
                model_type=payload.get("model_type", "unknown"),
                paper_title=payload.get("paper_title", "unknown"),
                card_path=payload.get("card_path", "unknown"),
            )
            results.append(result)

        return results

    def _parse_retriever_results(
        self, retrieved_info: List[dict]
    ) -> List[SearchResult]:
        """Parse results from CAMEL VectorRetriever.

        Args:
            retrieved_info: List of dicts from VectorRetriever.query()

        Returns:
            List of SearchResult objects
        """
        results = []
        for item in retrieved_info:
            # Check for "no suitable information" message
            if "No suitable information retrieved" in item.get("text", ""):
                logger.debug("No suitable information found for query")
                continue

            metadata = item.get("metadata", {})
            result = SearchResult(
                arxiv_id=metadata.get("arxiv_id", "unknown"),
                section=metadata.get("section", "unknown"),
                text=item.get("text", ""),
                similarity_score=float(item.get("similarity score", 0.0)),
                task_type=metadata.get("task_type", "unknown"),
                attack_level=metadata.get("attack_level", "unknown"),
                model_type=metadata.get("model_type", "unknown"),
                paper_title=metadata.get("paper_title", "unknown"),
                card_path=metadata.get("card_path", "unknown"),
            )
            results.append(result)

        return results

    def _build_filter(
        self,
        section_filter: Optional[str],
        task_type_filter: Optional[str],
        model_type_filter: Optional[str],
    ) -> Optional[Filter]:
        """Build Qdrant filter from optional parameters.

        Args:
            section_filter: Section type filter
            task_type_filter: Task type filter
            model_type_filter: Model modality filter

        Returns:
            Qdrant Filter object or None if no filters
        """
        conditions = []

        if section_filter:
            conditions.append(
                FieldCondition(key="section", match=MatchValue(value=section_filter))
            )

        if task_type_filter:
            conditions.append(
                FieldCondition(
                    key="task_type", match=MatchValue(value=task_type_filter)
                )
            )

        if model_type_filter:
            conditions.append(
                FieldCondition(
                    key="model_type", match=MatchValue(value=model_type_filter)
                )
            )

        if not conditions:
            return None

        return Filter(must=conditions)

    def get_paper_metadata(self, arxiv_id: str) -> Optional[dict]:
        """Retrieve metadata for a specific paper by ArXiv ID.

        Args:
            arxiv_id: ArXiv identifier (e.g., "2505.11842")

        Returns:
            Dictionary with paper metadata or None if not found
        """
        # Search for any chunk from this paper
        results = self._search_with_filter(
            query=arxiv_id,  # Use arxiv_id as query
            top_k=1,
            threshold=0.0,  # No threshold for metadata lookup
            filter_conditions=Filter(
                must=[FieldCondition(key="arxiv_id", match=MatchValue(value=arxiv_id))]
            ),
        )

        if not results:
            logger.warning(f"No metadata found for ArXiv ID: {arxiv_id}")
            return None

        # Return metadata from first result
        result = results[0]
        return {
            "arxiv_id": result.arxiv_id,
            "paper_title": result.paper_title,
            "task_type": result.task_type,
            "attack_level": result.attack_level,
            "model_type": result.model_type,
            "card_path": result.card_path,
        }
