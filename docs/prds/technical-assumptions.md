# Technical Assumptions

## Repository Structure: Monorepo

The AUST project will use a monorepo structure at https://github.com/vios-s/CAUST containing all components (agents, tools, RAG, memory, loop orchestration, experiments) with DeepUnlearn as a git submodule and CAMEL-AI installed in dev/editable mode in an external/ directory.

**Rationale**: Monorepo simplifies dependency management, enables atomic commits across components, and aligns with the MVP timeline where tight integration is prioritized over service independence.

## Service Architecture

**Containerized Monolithic Application**: Single Python application running in Docker with Kubernetes orchestration. The application is stateful with memory persisting across loop iterations via mounted volumes and attack traces saved incrementally.

**Rationale**: Monolithic architecture for MVP reduces complexity compared to microservices. Container isolation provides reproducibility and security. Future migration to orchestrator-worker architecture (Phase 2) is possible but deferred to focus on core research loop functionality.

## Testing Requirements

**Unit + Integration Testing**: Unit tests for individual agent components (hypothesis generator, critic, query generator, evaluator, reporter, judges) and integration tests for full inner/outer loop workflows.

- Unit tests: Test agent logic, prompt generation, RAG retrieval, evaluation metrics
- Integration tests: Test complete vulnerability discovery workflow end-to-end
- Manual validation: Attack trace reproducibility testing with external users

**Rationale**: Given the 3-week timeline, focus testing on critical path components. Full E2E automation is lower priority than functional system delivery. Manual validation ensures attack traces meet usability requirements (NFR9).

## Additional Technical Assumptions and Requests

- **GPU Resource Management**: H200 GPUs are available via Kubernetes job scheduling. Experiment execution should handle job queue delays gracefully (timeout/retry logic).

- **Paper Corpus Management**: 10-20 papers per task domain (data-based unlearning, concept-erasure, shared attack methods) stored as PDFs or text. RAG system uses FAISS or Chroma for vector database with embedding model (OpenRouter or Sentence-Transformers).

- **Agent Prompting Strategy**: Agents use prompt-based task differentiation for data-based vs concept-erasure workflows. Prompts stored in `configs/` directory for easy iteration and versioning.

- **Seed Hypothesis Templates**: Pre-load 3-5 known attack patterns (membership inference, model inversion, data extraction) as seed templates to mitigate hypothesis quality risk (identified as HIGH risk in brief).

- **Attack Trace Format**: Attack traces output in dual format - JSON (machine-readable for analysis) + Markdown (human-readable for paper integration and user reproduction).

- **Judge Persona Definitions**: 3-5 judge personas pre-defined: Security Expert, ML Researcher, Privacy Advocate, Skeptical Reviewer, Industry Practitioner. Each has specific evaluation criteria and perspective.

- **Evaluation Metric Thresholds**: Configurable thresholds in `configs/` - data-based: forget accuracy delta; concept-erasure: generation-based leakage probability, CLIP score changes.

- **Error Handling & Logging**: Comprehensive logging for debugging (all agent interactions, API calls, experiment results). Graceful degradation for API failures, GPU unavailability, or tool integration errors.

- **Output Persistence**: All outputs (reports, attack traces, judge evaluations, experiment results) saved to persistent Kubernetes volumes to survive container restarts and enable post-hoc analysis.
