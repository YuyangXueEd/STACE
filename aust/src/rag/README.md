# AUST RAG System

Vector-based Retrieval Augmented Generation (RAG) system for semantic search over research paper cards.

## Architecture Overview

The AUST RAG system uses:

- **Vector Database**: Qdrant (local persistence)
- **Embedding Model**: SentenceTransformers (`sentence-transformers/all-MiniLM-L6-v2`, 384-dim)
- **Agent Framework Integration**: CAMEL-AI `VectorRetriever` and `QdrantStorage`
- **Chunking Strategy**: Semantic section-based chunking (Methodology, Experiments, Results, Relevance)

### Components

1. **PaperCardChunker** ([chunking.py](./chunking.py)): Parses paper card markdown files and extracts semantic sections with metadata
2. **PaperRAG** ([vector_db.py](./vector_db.py)): Vector search interface with filtering support
3. **Build Script** ([../scripts/build_vector_index.py](../../scripts/build_vector_index.py)): Indexes all paper cards into Qdrant

## Quick Start

### 1. Build Vector Index

Before using the RAG system, build the vector index from paper cards:

```bash
# From project root
python aust/scripts/build_vector_index.py
```

This will:
- Read all paper cards from `.paper_cards/`
- Extract semantic sections (Methodology, Experiments, Results, Relevance)
- Embed chunks using SentenceTransformers
- Index ~400-500 chunks into Qdrant at `aust/rag_paper_db/`

Expected output:
```
INFO - Found 101 paper cards to process
INFO - Successfully chunked 101 cards into 450 chunks
INFO - Indexing 450 chunks in batches of 50...
INFO - Total chunks processed: 450
INFO - Successfully indexed: 450
INFO - Processing time: 45.23 seconds
```

### 2. Basic Search

```python
from aust.src.rag import PaperRAG

# Initialize RAG system
rag = PaperRAG()

# Simple semantic search
results = rag.search(
    query="adversarial attacks on text-to-image diffusion models",
    top_k=5
)

# Print results
for result in results:
    print(f"Paper: {result.paper_title}")
    print(f"Section: {result.section}")
    print(f"Similarity: {result.similarity_score:.3f}")
    print(f"Text: {result.text[:200]}...")
    print("-" * 80)
```

### 3. Filtered Search

Search with section and task type filters:

```python
# Search only METHODOLOGY sections for any-to-t papers
results = rag.search(
    query="gradient-based attack optimization methods",
    top_k=5,
    section_filter="METHODOLOGY",
    task_type_filter="any-to-t"
)

# All results will be from METHODOLOGY sections of any-to-t papers
for result in results:
    assert result.section == "METHODOLOGY"
    assert result.task_type == "any-to-t"
```

### 4. Custom Similarity Threshold

```python
# Only return highly relevant results (similarity >= 0.7)
results = rag.search(
    query="unlearning methods for removing training data",
    top_k=10,
    similarity_threshold=0.7
)
```

## Configuration

### PaperRAG Initialization

```python
rag = PaperRAG(
    storage_path="aust/rag_paper_db",  # Path to Qdrant storage
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # Embedding model
    collection_name="aust_papers",  # Qdrant collection name
    similarity_threshold=0.5  # Default similarity threshold (0-1)
)
```

### Search Parameters

- **query** (str): Query string for semantic search
- **top_k** (int): Number of results to return (default: 5)
- **section_filter** (Optional[str]): Filter by section type
  - Valid values: `"METHODOLOGY"`, `"EXPERIMENTS"`, `"RESULTS"`, `"RELEVANCE"`
- **task_type_filter** (Optional[str]): Filter by task taxonomy
  - Valid values: `"any-to-t"`, `"any-to-v"`
- **similarity_threshold** (Optional[float]): Override default threshold (0-1)

## Use Cases

### Use Case 1: Query Generator Agent (Story 2.3)

The Query Generator Agent uses PaperRAG to retrieve relevant attack methods:

```python
# Retrieve similar attack methodologies
methodology_results = rag.search(
    query="text-based adversarial attacks on diffusion models",
    top_k=3,
    section_filter="METHODOLOGY",
    task_type_filter="any-to-t"
)

# Retrieve experimental setups
experiment_results = rag.search(
    query="evaluation metrics for unlearning robustness",
    top_k=3,
    section_filter="EXPERIMENTS"
)

# Generate queries based on retrieved papers
for result in methodology_results:
    print(f"Retrieved attack method from: {result.paper_title}")
    # Use result.text to inform query generation...
```

### Use Case 2: Hypothesis Generator with RAG Context

```python
# Retrieve relevant prior work for hypothesis generation
relevance_results = rag.search(
    query="concept erasure for text-to-image models",
    top_k=5,
    section_filter="RELEVANCE"
)

# Use retrieved context to generate informed hypotheses
for result in relevance_results:
    print(f"Prior work application: {result.text}")
```

### Use Case 3: Paper Metadata Lookup

```python
# Get metadata for a specific paper
metadata = rag.get_paper_metadata(arxiv_id="2505.11842")

print(f"Title: {metadata['paper_title']}")
print(f"Task Type: {metadata['task_type']}")
print(f"Attack Level: {metadata['attack_level']}")
print(f"Card Path: {metadata['card_path']}")
```

## Data Model

### Chunk

Represents a semantic section from a paper card:

```python
@dataclass
class Chunk:
    arxiv_id: str          # "2505.11842"
    section: str           # "METHODOLOGY" | "EXPERIMENTS" | "RESULTS" | "RELEVANCE"
    task_type: str         # "any-to-t" | "any-to-v"
    attack_level: str      # "input_level" | "encoder_level" | "generator_level" | ...
    model_type: str        # "T->I" | ... (copied from paper card metadata)
    paper_title: str       # "Video-SafetyBench: A Benchmark for..."
    card_path: str         # ".paper_cards/any-to-t/input_level/2505.11842.md"
    text: str              # "[METHODOLOGY] Video-SafetyBench...\n\nA three-stage..."
```

### SearchResult

Represents a search result with similarity score:

```python
@dataclass
class SearchResult:
    arxiv_id: str
    section: str
    text: str
    similarity_score: float  # Cosine similarity (0-1)
    task_type: str
    attack_level: str
    model_type: str
    paper_title: str
    card_path: str
```

## Performance

The RAG system meets NFR8 performance targets:

| Metric | Target | Typical Performance |
|--------|--------|---------------------|
| Collection load time | < 2s | ~0.5s |
| Single query latency | < 5s | ~0.3s |
| Batch query (3 queries) | < 10s | ~1.0s |
| Memory footprint | < 500MB | ~150MB |

Run performance tests:

```bash
pytest tests/unit/test_rag_performance.py -v
```

## Story 2.3 Interface Contract

The PaperRAG interface provides the following contract for Story 2.3 (Query Generator Agent):

### Required Interface

```python
class PaperRAG:
    def search(
        self,
        query: str,
        top_k: int = 5,
        section_filter: Optional[str] = None,
        task_type_filter: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Search for relevant paper chunks by semantic similarity."""
        ...
```

### Expected Usage in Query Generator

```python
# Story 2.3: Query Generator Agent
from aust.src.rag import PaperRAG

rag = PaperRAG()

# Step 1: Retrieve similar attack methodologies
similar_attacks = rag.search(
    query=hypothesis.attack_description,
    top_k=3,
    section_filter="METHODOLOGY",
    task_type_filter=task_config.task_type
)

# Step 2: Generate queries inspired by retrieved methods
for attack_paper in similar_attacks:
    # Extract attack techniques from attack_paper.text
    # Generate variations based on hypothesis + retrieved context
    ...
```

### Guarantees

1. **Fast retrieval**: < 5s per query (NFR8)
2. **Relevant results**: Cosine similarity threshold ensures semantic relevance
3. **Structured metadata**: All results include arxiv_id, section, task_type, attack_level
4. **Filtering support**: Can filter by section type and task taxonomy
5. **CAMEL-AI compatible**: Uses CAMEL VectorRetriever for consistency with agent framework

## Testing

### Unit Tests

```bash
pytest tests/unit/test_paper_rag.py -v
```

Tests cover:
- Paper card chunking logic
- Metadata extraction
- Section parsing
- Filter building
- Search result parsing
- Edge cases (empty query, no results, invalid filters)

### Performance Tests

```bash
pytest tests/unit/test_rag_performance.py -v
```

Performance tests verify:
- Collection load time < 2s
- Single query latency < 5s
- Batch query latency < 10s
- Filtered search performance
- Search accuracy and relevance
- Metadata completeness

## Troubleshooting

### Vector index not found

**Error**: `Vector index not found`

**Solution**: Build the index first:
```bash
python aust/scripts/build_vector_index.py
```

### No results returned

**Possible causes**:
1. Similarity threshold too high -> Lower `similarity_threshold` parameter
2. Filters too restrictive -> Remove `section_filter` or `task_type_filter`
3. Query not semantically similar -> Rephrase query to match paper card content

### Slow query performance

**Possible causes**:
1. Large `top_k` value -> Reduce to 5-10 results
2. Cold start (first query) -> Subsequent queries will be faster due to caching
3. Complex filters -> Simplify filter conditions

## Implementation Details

### Chunking Strategy

Paper cards are chunked by semantic sections:

1. **Target Sections**: Methodology, Experiment Design, Key Results, Relevance to Our Work
2. **Chunk Format**: `[SECTION_TYPE] Paper Title\n\nSection content...`
3. **Minimum Length**: 50 characters (shorter sections skipped)
4. **Metadata Extraction**: ArXiv ID, attack level, task type from card frontmatter and directory structure

### Embedding

- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimension**: 384
- **Task**: Retrieval (query vs passage)
- **Local inference**: No API calls required

### Vector Database

- **Storage**: Qdrant with local disk persistence
- **Distance Metric**: Cosine similarity
- **Index**: HNSW (approximate nearest neighbor)
- **Filtering**: Native Qdrant filter DSL with metadata payloads

## Future Enhancements

Potential improvements for Phase 2:

1. **GraphRAG**: Add graph-based reasoning for multi-hop queries
2. **Hybrid Search**: Combine vector search with keyword/BM25 search
3. **Re-ranking**: Use cross-encoder for re-ranking top results
4. **Query Expansion**: Expand queries with synonyms/related terms
5. **Caching**: Cache frequent queries for faster response
6. **Incremental Indexing**: Add new papers without rebuilding full index

## References

- [CAMEL-AI RAG Cookbook](../../../external/camel/docs/cookbooks/advanced_features/agents_with_rag.ipynb)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [SentenceTransformers](https://www.sbert.net/)
- [Story 2.2 Specification](../../../docs/stories/2.2.vector-database-embedding-system.md)
