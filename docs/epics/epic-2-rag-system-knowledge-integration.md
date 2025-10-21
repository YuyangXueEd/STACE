# Epic 2: RAG System & Knowledge Integration

Expanded Goal: Add RAG-based paper retrieval with query generation to enhance hypothesis quality through literature-based knowledge. Integrate CAMEL-AI long-term memory to store successful vulnerability discoveries for future reference. By the end of this epic, hypothesis generation should leverage relevant research papers and past successes, significantly improving the quality and novelty of proposed stress tests.

Status: ✅ Partially Complete (2.1.1 Completed; 2.2 Ready for Implementation; 2.3-2.6 TBD)

## Story 2.1.1: Paper Card Generation for RAG System – Completed

See `docs/stories/2.1.1.paper-corpus-collection-storage.md`.

## Story 2.2: Vector Database & Embedding System – Ready for Implementation

See `docs/stories/2.2.vector-database-embedding-system.md` (Qdrant + SentenceTransformers; section-level chunking of paper cards).

### Acceptance Criteria

1. Vector database implemented using Qdrant in `rag/vector_db.py`
2. Embedding model integrated using Sentence-Transformers (all-MiniLM-L6-v2)
3. Paper cards chunked by section (Methodology, Experiments, Results, Relevance) with metadata payloads
4. Vector index persisted to `rag/vector_index/` (local collection)
5. Query interface: `search(query_text, top_k=5, filters)` returns top-k relevant chunks with metadata
6. Search latency < 5 seconds per query (NFR8)

## Story 2.3: Query Generator Agent

As a **researcher**,
I want **a Query Generator agent that converts evaluation feedback and hypothesis needs into RAG search queries**,
so that **the system retrieves relevant papers automatically**.

### Acceptance Criteria

1. Query Generator agent implemented in `agents/query_generator.py`
2. Agent accepts inputs: current hypothesis (if any), evaluation feedback, task type (data-based or concept-erasure)
3. Agent generates 1-3 search queries focusing on: attack methods, unlearning vulnerabilities, relevant evaluation metrics
4. Queries are logged to `outputs/queries/` with timestamp
5. Query Generator calls RAG search interface and returns top-5 relevant paper chunks per query

## Story 2.4: Enhance Hypothesis Generator with RAG

As a **researcher**,
I want **the Hypothesis Generator to incorporate RAG retrieval results**,
so that **hypotheses are informed by relevant research and more novel**.

### Acceptance Criteria

1. Hypothesis Generator modified to accept RAG retrieval results as additional context
2. Generator workflow: receive feedback → trigger Query Generator → receive paper chunks → generate hypothesis using seeds + RAG context + past results
3. Generated hypotheses cite relevant papers (e.g., "inspired by membership inference attack from [Smith et al., 2023]")
4. Hypothesis novelty improves measurably (can be validated in Story 2.6 with critic scoring)
5. Backward compatibility: system still works if RAG returns no results (falls back to seed templates)

## Story 2.5: CAMEL-AI Long-Term Memory Integration

As a **researcher**,
I want **to integrate CAMEL-AI's long-term memory to store successful vulnerability discoveries**,
so that **the system learns from past successes across multiple runs**.

### Acceptance Criteria

1. Memory system implemented in `memory/long_term_memory.py` using CAMEL-AI's memory API
2. Successful experiments (VULNERABILITY_FOUND) are stored with: hypothesis, experiment parameters, results, attack trace reference
3. Memory retrieval interface: `get_successful_attacks(task_type)` returns past successful attacks for given task
4. Hypothesis Generator queries memory at start of each run to inform initial hypothesis generation
5. Memory persisted to `outputs/memory_store/` (survives container restarts per NFR14)
6. Successful memories are exported into the Qdrant collection as section="experience" for future retrieval

## Story 2.6: RAG-Enhanced End-to-End Test

As a **researcher**,
I want **to run the complete inner loop with RAG and memory integration**,
so that **we validate that knowledge integration improves hypothesis quality**.

### Acceptance Criteria

1. Run inner loop with RAG enabled on data-based unlearning method (5-10 iterations)
2. Verify Query Generator is called and RAG results are incorporated into hypotheses
3. Compare hypothesis quality: with RAG vs without RAG (use critic agent scoring: novelty, feasibility, rigor)
4. At least one hypothesis cites relevant retrieved papers in its description
5. If VULNERABILITY_FOUND, memory system successfully stores the result for future retrieval
6. Performance: RAG retrieval adds < 5 seconds per loop iteration (NFR8)
