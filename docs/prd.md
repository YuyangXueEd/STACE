# AUST Product Requirements Document (PRD)

## Goals and Background Context

### Goals

- Demonstrate autonomous end-to-end AI scientific workflow (hypothesis generation → experimentation → interpretation → reporting → judging) on machine unlearning vulnerabilities
- Discover at least one exploitable vulnerability in both data-based unlearning and concept-based erasure methods
- Generate reproducible attack traces showing step-by-step vulnerability discovery paths
- Produce multi-perspective LLM judge evaluations from diverse angles (novelty, rigor, reproducibility, impact, exploitability)
- Complete full system implementation by November 7th, 2025 with paper draft ready by November 14th, 2025
- Establish reusable evaluation methodology for adaptive adversarial testing that outperforms static benchmarks

### Background Context

Machine unlearning methods promise to remove specific data or concepts from trained models—critical for GDPR compliance, bias mitigation, and preventing misuse of generative AI. However, current evaluation approaches rely on static benchmarks that cannot anticipate creative, adaptive attacks from motivated adversaries. AUST (AI Scientist for Autonomous Unlearning Security Testing) addresses this gap by implementing an autonomous research system with nested inner/outer loops: the inner loop iteratively generates hypotheses, retrieves relevant research, executes experiments, and evaluates results until vulnerabilities are discovered; the outer loop generates academic reports and multi-perspective judge evaluations. This demonstrates AI conducting rigorous, end-to-end scientific research on a societally relevant privacy/compliance problem.

The project builds on comprehensive brainstorming and planning completed with the analyst, including detailed system architecture (critic-generator agents, RAG-based knowledge retrieval, multi-modal evaluation), aggressive 3-week implementation timeline, and clear MVP scope prioritization. AUST will run in Docker/Kubernetes with H200 GPUs, integrate DeepUnlearn as a git submodule, use CAMEL-AI in dev mode for multi-agent orchestration, and leverage OpenRouter APIs for flexible LLM/VLM access.

### Change Log

| Date | Version | Description | Author |
|------|---------|-------------|--------|
| 2025-10-16 | 0.1 | Initial PRD creation from Project Brief | John (PM Agent) |

## Requirements

### Functional

**FR1**: The system shall implement an Inner Research Loop that iterates through Hypothesis Generation → Query Generation → RAG Retrieval → Experiment Execution → Multi-Modal Evaluation → Feedback Integration until a vulnerability is discovered or maximum iterations (10) are reached.

**FR2**: The system shall provide a Hypothesis Generator agent that proposes targeted stress tests for unlearning methods based on retrieved papers and past experiment results.

**FR3**: The system shall provide a Critic Agent that debates with the Hypothesis Generator after the first iteration to challenge and improve hypothesis quality.

**FR4**: The system shall provide a Query Generator agent that converts evaluation feedback and hypothesis needs into RAG search queries.

**FR5**: The system shall implement a RAG system with 10-20 key papers per task domain (data-based unlearning, concept-erasure) plus a shared attack methods corpus, supporting semantic search over paper content.

**FR6**: The system shall provide an Experiment Executor that integrates DeepUnlearn (via git submodule as MCP FunctionTool) for data-based unlearning experiments.

**FR7**: The system shall provide an Experiment Executor that integrates concept-erasure methods from GitHub repositories (as MCP FunctionTools or submodules) for concept-based unlearning experiments.

**FR8**: The system shall provide an Evaluator that uses threshold-based metrics (forget accuracy) for data-based unlearning vulnerability detection.

**FR9**: The system shall provide an Evaluator that uses VLM-based analysis (generation-based leakage detection) for concept-erasure vulnerability detection.

**FR10**: The system shall implement a Memory System using CAMEL-AI long-term storage to capture successful vulnerability discoveries for future reference.

**FR11**: The system shall generate Attack Traces documenting step-by-step records of the inner research loop showing hypothesis evolution and vulnerability discovery paths.

**FR12**: The system shall provide a Reporter agent that generates academic-format reports (Introduction, Methods, Experiments, Results, Discussion, Conclusion) with citation integration from retrieved papers.

**FR13**: The system shall provide 3-5 Judge LLM personas that evaluate findings from multiple perspectives (novelty, rigor, reproducibility, impact, exploitability).

**FR14**: The system shall support both data-based unlearning and concept-erasure tasks using unified workflow with prompt-based task differentiation.

**FR15**: The system shall exit the inner research loop when either a vulnerability is discovered OR maximum iterations are reached, then proceed to the outer loop (Reporter → Judges).

### Non Functional

**NFR1**: The system shall run in Docker containers orchestrated by Kubernetes with H200 GPU access via job.yaml configuration.

**NFR2**: The system shall use CAMEL-AI installed in dev/editable mode (`pip install -e`) to allow source code modifications as needed.

**NFR3**: The system shall integrate DeepUnlearn as a git submodule for version control and local modifications.

**NFR4**: The system shall use OpenRouter API for LLM/VLM access supporting multiple models (GPT-4o, Claude 3.5 Sonnet, etc.).

**NFR5**: The system shall use Python 3.11+ as the primary implementation language.

**NFR6**: The system shall complete inner research loop iterations in < 30 minutes per cycle.

**NFR7**: The system shall discover vulnerabilities within < 5 hours for one complete research loop (initial hypothesis to final report).

**NFR8**: The system shall achieve RAG retrieval latency < 5 seconds per query.

**NFR9**: Attack traces shall be reproducible by independent users with 80%+ success rate without additional clarification.

**NFR10**: The system shall operate with ≥ 90% autonomy (without human intervention) after initial setup.

**NFR11**: The system shall pin all dependency versions in requirements.txt and git submodule commits for reproducibility.

**NFR12**: The system shall implement container isolation for experiment execution to prevent code injection or security issues.

**NFR13**: The system shall handle OpenRouter API rate limits gracefully with retry logic and error handling.

**NFR14**: The system shall store outputs (reports, attack traces, judge evaluations) to persistent volumes for preservation across container restarts.

**NFR15**: The system implementation shall be completed by November 7th, 2025 to allow one week for paper writing before the November 14th deadline.

## Technical Assumptions

### Repository Structure: Monorepo

The AUST project will use a monorepo structure at https://github.com/vios-s/CAUST containing all components (agents, tools, RAG, memory, loop orchestration, experiments) with DeepUnlearn as a git submodule and CAMEL-AI installed in dev/editable mode in an external/ directory.

**Rationale**: Monorepo simplifies dependency management, enables atomic commits across components, and aligns with the MVP timeline where tight integration is prioritized over service independence.

### Service Architecture

**Containerized Monolithic Application**: Single Python application running in Docker with Kubernetes orchestration. The application is stateful with memory persisting across loop iterations via mounted volumes and attack traces saved incrementally.

**Rationale**: Monolithic architecture for MVP reduces complexity compared to microservices. Container isolation provides reproducibility and security. Future migration to orchestrator-worker architecture (Phase 2) is possible but deferred to focus on core research loop functionality.

### Testing Requirements

**Unit + Integration Testing**: Unit tests for individual agent components (hypothesis generator, critic, query generator, evaluator, reporter, judges) and integration tests for full inner/outer loop workflows.

- Unit tests: Test agent logic, prompt generation, RAG retrieval, evaluation metrics
- Integration tests: Test complete vulnerability discovery workflow end-to-end
- Manual validation: Attack trace reproducibility testing with external users

**Rationale**: Given the 3-week timeline, focus testing on critical path components. Full E2E automation is lower priority than functional system delivery. Manual validation ensures attack traces meet usability requirements (NFR9).

### Additional Technical Assumptions and Requests

- **GPU Resource Management**: H200 GPUs are available via Kubernetes job scheduling. Experiment execution should handle job queue delays gracefully (timeout/retry logic).

- **Paper Corpus Management**: 10-20 papers per task domain (data-based unlearning, concept-erasure, shared attack methods) stored as PDFs or text. RAG system uses FAISS or Chroma for vector database with embedding model (OpenRouter or Sentence-Transformers).

- **Agent Prompting Strategy**: Agents use prompt-based task differentiation for data-based vs concept-erasure workflows. Prompts stored in `configs/` directory for easy iteration and versioning.

- **Seed Hypothesis Templates**: Pre-load 3-5 known attack patterns (membership inference, model inversion, data extraction) as seed templates to mitigate hypothesis quality risk (identified as HIGH risk in brief).

- **Attack Trace Format**: Attack traces output in dual format - JSON (machine-readable for analysis) + Markdown (human-readable for paper integration and user reproduction).

- **Judge Persona Definitions**: 3-5 judge personas pre-defined: Security Expert, ML Researcher, Privacy Advocate, Skeptical Reviewer, Industry Practitioner. Each has specific evaluation criteria and perspective.

- **Evaluation Metric Thresholds**: Configurable thresholds in `configs/` - data-based: forget accuracy delta; concept-erasure: generation-based leakage probability, CLIP score changes.

- **Error Handling & Logging**: Comprehensive logging for debugging (all agent interactions, API calls, experiment results). Graceful degradation for API failures, GPU unavailability, or tool integration errors.

- **Output Persistence**: All outputs (reports, attack traces, judge evaluations, experiment results) saved to persistent Kubernetes volumes to survive container restarts and enable post-hoc analysis.

## Epic List

### Epic 1: Foundation & Inner Loop MVP (Week 1: Oct 24-30)
**Goal**: Establish project infrastructure, CAMEL-AI integration, DeepUnlearn integration, and implement a complete inner research loop for data-based unlearning only.

### Epic 2: RAG System & Knowledge Integration (Week 2: Oct 31-Nov 6, Part 1)
**Goal**: Add RAG-based paper retrieval with query generation, integrate CAMEL-AI long-term memory, and enhance hypothesis generation with literature-based knowledge.

### Epic 3: Concept-Erasure Task & Multi-Modal Evaluation (Week 2: Oct 31-Nov 6, Part 2)
**Goal**: Extend system to support concept-erasure unlearning methods with VLM-based evaluation, demonstrating generality across both unlearning paradigms.

### Epic 4: Outer Loop & Multi-Perspective Judging (Week 3: Nov 7-13)
**Goal**: Implement reporter agent for academic paper generation and multi-perspective LLM judges, completing the end-to-end autonomous research workflow.

## Epic 1: Foundation & Inner Loop MVP

**Expanded Goal**: Establish the foundational project infrastructure including Docker/Kubernetes setup, CAMEL-AI framework integration in dev mode, DeepUnlearn as a git submodule, and implement a complete inner research loop (Hypothesis Generator → Critic → Experiment Executor → Evaluator → Feedback) for data-based unlearning only. By the end of this epic, the system should be able to run 3-5 manual loop iterations and attempt to discover at least one vulnerability in a data-based unlearning method.

### Story 1.1: Project Setup & Infrastructure

As a **developer**,
I want **to set up the CAUST repository with Docker/Kubernetes configuration, Python 3.11+ environment, and basic project structure**,
so that **the team has a working development environment ready for agent implementation**.

#### Acceptance Criteria

1. Repository at https://github.com/vios-s/CAUST is initialized with directory structure: `agents/`, `tools/`, `rag/`, `memory/`, `loop/`, `outputs/`, `configs/`, `experiments/`, `submodules/`, `external/`, `docker/`
2. Docker conda_Dockerfile builds successfully with Python 3.11+ and core dependencies (PyTorch, CAMEL-AI requirements)
3. Kubernetes job.yaml is configured for H200 GPU access and persistent volume mounts for outputs
4. `requirements.txt` includes pinned versions of core dependencies (PyTorch, CAMEL-AI, OpenRouter client)
5. Basic logging framework is configured (Python logging to file + console)
6. README.md documents setup instructions and project structure

### Story 1.2: CAMEL-AI Integration in Dev Mode

As a **developer**,
I want **to integrate CAMEL-AI in dev/editable mode and verify basic agent functionality**,
so that **we can create and modify agents as needed throughout development**.

#### Acceptance Criteria

1. CAMEL-AI is cloned/installed in `external/camel/` with `pip install -e external/camel`
2. Basic test agent (simple echo agent) successfully instantiates using CAMEL-AI's agent API
3. OpenRouter API integration is tested with at least one model call (GPT-4o or Claude 3.5)
4. Agent prompt configuration system is implemented in `configs/` directory (YAML or JSON format)
5. Basic agent interaction logging captures all LLM API calls and responses

### Story 1.3: DeepUnlearn Integration as Git Submodule

As a **developer**,
I want **to integrate DeepUnlearn as a git submodule and wrap it as a CAMEL-AI MCP FunctionTool**,
so that **agents can programmatically trigger unlearning and evaluation experiments**.

#### Acceptance Criteria

1. DeepUnlearn repository is added as git submodule in `submodules/DeepUnlearn/`
2. DeepUnlearn dependencies are integrated into requirements.txt without conflicts
3. MCP FunctionTool wrapper in `tools/deepunlearn_tool.py` exposes at minimum: `unlearn_model(method, dataset, params)` and `evaluate_model(model, metrics)`
4. Successful test execution: trigger unlearning on a simple dataset and verify evaluation metrics are returned
5. Error handling for GPU unavailability, invalid parameters, and DeepUnlearn failures

### Story 1.4: Hypothesis Generator Agent

As a **researcher**,
I want **a Hypothesis Generator agent that proposes stress tests for data-based unlearning methods**,
so that **the system can autonomously generate testable vulnerability hypotheses**.

#### Acceptance Criteria

1. Hypothesis Generator agent implemented in `agents/hypothesis_generator.py`
2. Agent accepts context inputs: task type (data-based), past experiment results (empty for first iteration), feedback from evaluator
3. Agent generates hypothesis in structured format: attack method, target unlearning method, experiment parameters, expected outcome
4. Seed templates (3-5 known attack patterns: membership inference, model inversion, data extraction) are pre-loaded in `configs/seed_hypotheses.yaml`
5. For MVP, hypothesis generation uses seed templates + basic LLM variation (no RAG yet - deferred to Epic 2)
6. Output hypothesis is logged to `outputs/hypotheses/` with timestamp

### Story 1.5: Critic Agent

As a **researcher**,
I want **a Critic Agent that debates with the Hypothesis Generator after the first iteration**,
so that **hypothesis quality improves through adversarial questioning**.

#### Acceptance Criteria

1. Critic Agent implemented in `agents/critic.py`
2. Critic activates after first inner loop iteration (when feedback is available)
3. Critic challenges hypothesis on: novelty (is this just repeating seed templates?), feasibility (can this be executed?), rigor (is the expected outcome testable?)
4. Critic provides structured feedback to Hypothesis Generator: strengths, weaknesses, suggestions for improvement
5. Hypothesis Generator incorporates critic feedback in next iteration
6. Debate exchange (Hypothesis → Critic → Revised Hypothesis) is logged to `outputs/debates/`

### Story 1.6: Experiment Executor for Data-Based Unlearning

As a **researcher**,
I want **an Experiment Executor that runs unlearning experiments using DeepUnlearn**,
so that **hypotheses can be tested automatically on real unlearning methods**.

#### Acceptance Criteria

1. Experiment Executor implemented in `agents/experiment_executor.py`
2. Executor parses hypothesis structure and translates to DeepUnlearn FunctionTool calls
3. Executor handles experiment execution: trigger unlearning, wait for completion, collect results
4. Timeout handling (max 30 minutes per experiment as per NFR6) with graceful failure
5. Experiment results (metrics, model artifacts, logs) saved to `experiments/results/` with unique run ID
6. Executor reports success/failure status and metrics back to loop orchestrator

### Story 1.7: Evaluator with Threshold-Based Metrics

As a **researcher**,
I want **an Evaluator that assesses experiment results using forget accuracy thresholds**,
so that **the system can automatically detect when vulnerabilities are discovered**.

#### Acceptance Criteria

1. Evaluator implemented in `agents/evaluator.py` for data-based unlearning
2. Evaluator computes forget accuracy from experiment results using DeepUnlearn's evaluation functions
3. Configurable threshold in `configs/evaluation_thresholds.yaml` (e.g., forget accuracy delta > 10%)
4. Evaluator determines: VULNERABILITY_FOUND, INCONCLUSIVE, or NO_VULNERABILITY
5. Evaluator generates structured feedback: what worked, what failed, suggestions for next hypothesis
6. Evaluation results logged to `outputs/evaluations/` with experiment run ID reference

### Story 1.8: Inner Loop Orchestrator

As a **researcher**,
I want **an Inner Loop Orchestrator that coordinates the research cycle**,
so that **the system autonomously iterates until a vulnerability is discovered or max iterations reached**.

#### Acceptance Criteria

1. Loop Orchestrator implemented in `loop/inner_loop.py`
2. Orchestrator manages loop state: iteration count, experiment history, current hypothesis, feedback
3. Loop flow: Hypothesis Generator → (Critic if iteration > 1) → Experiment Executor → Evaluator → check exit condition
4. Exit conditions: VULNERABILITY_FOUND OR iteration count >= 10 (configurable in `configs/loop_config.yaml`)
5. Loop state persisted to `outputs/loop_state.json` after each iteration (supports restart)
6. Attack trace generation: each iteration's hypothesis, experiment parameters, results, and feedback appended to `outputs/attack_traces/trace_{run_id}.md`

### Story 1.9: End-to-End Inner Loop Test

As a **researcher**,
I want **to run the complete inner loop end-to-end on a data-based unlearning method**,
so that **we validate the MVP can discover at least one vulnerability**.

#### Acceptance Criteria

1. Manual execution script `run_inner_loop.py` triggers full loop with configurable task (data-based unlearning method selection)
2. System runs 3-5 iterations successfully without crashes
3. Attack trace file is generated in Markdown format documenting all iterations
4. At minimum, system reaches INCONCLUSIVE or VULNERABILITY_FOUND state (not just failures)
5. All agent interactions, experiment results, and evaluations are logged for debugging
6. Performance: loop iteration completes in < 30 minutes per cycle (NFR6)

## Epic 2: RAG System & Knowledge Integration

**Expanded Goal**: Add RAG-based paper retrieval with query generation to enhance hypothesis quality through literature-based knowledge. Integrate CAMEL-AI long-term memory to store successful vulnerability discoveries for future reference. By the end of this epic, hypothesis generation should leverage relevant research papers and past successes, significantly improving the quality and novelty of proposed stress tests.

### Story 2.1: Paper Corpus Collection & Storage

As a **researcher**,
I want **to collect and store 10-20 key papers per task domain in an accessible format**,
so that **the RAG system has a knowledge base to retrieve from**.

#### Acceptance Criteria

1. Paper corpus directory created: `rag/papers/data_based/`, `rag/papers/concept_erasure/`, `rag/papers/attack_methods/`
2. Minimum 10 papers collected per directory (PDFs or text files) covering relevant research: data-based unlearning (SISA, machine unlearning), concept-erasure (EraseDiff, concept ablation), attack methods (membership inference, model inversion)
3. Paper metadata file `rag/paper_metadata.json` contains: title, authors, venue, year, file path, summary for each paper
4. PDF parsing utility in `rag/pdf_parser.py` extracts text from PDFs (using PyMuPDF or pdfplumber)
5. Extracted paper text stored in `rag/papers_text/` for faster retrieval

### Story 2.2: Vector Database & Embedding System

As a **developer**,
I want **to set up a vector database with embeddings for semantic search over papers**,
so that **agents can retrieve relevant research based on query similarity**.

#### Acceptance Criteria

1. Vector database implemented using FAISS or Chroma in `rag/vector_db.py`
2. Embedding model integrated (OpenRouter embeddings or Sentence-Transformers)
3. All paper text chunked (by paragraph or section) and embedded with unique IDs
4. Vector index built and persisted to `rag/vector_index/` for fast loading
5. Query interface: `search(query_text, top_k=5)` returns top-k most relevant paper chunks with metadata
6. Search latency < 5 seconds per query (NFR8)

### Story 2.3: Query Generator Agent

As a **researcher**,
I want **a Query Generator agent that converts evaluation feedback and hypothesis needs into RAG search queries**,
so that **the system retrieves relevant papers automatically**.

#### Acceptance Criteria

1. Query Generator agent implemented in `agents/query_generator.py`
2. Agent accepts inputs: current hypothesis (if any), evaluation feedback, task type (data-based or concept-erasure)
3. Agent generates 1-3 search queries focusing on: attack methods, unlearning vulnerabilities, relevant evaluation metrics
4. Queries are logged to `outputs/queries/` with timestamp
5. Query Generator calls RAG search interface and returns top-5 relevant paper chunks per query

### Story 2.4: Enhance Hypothesis Generator with RAG

As a **researcher**,
I want **the Hypothesis Generator to incorporate RAG retrieval results**,
so that **hypotheses are informed by relevant research and more novel**.

#### Acceptance Criteria

1. Hypothesis Generator modified to accept RAG retrieval results as additional context
2. Generator workflow: receive feedback → trigger Query Generator → receive paper chunks → generate hypothesis using seeds + RAG context + past results
3. Generated hypotheses cite relevant papers (e.g., "inspired by membership inference attack from [Smith et al., 2023]")
4. Hypothesis novelty improves measurably (can be validated in Story 2.6 with critic scoring)
5. Backward compatibility: system still works if RAG returns no results (falls back to seed templates)

### Story 2.5: CAMEL-AI Long-Term Memory Integration

As a **researcher**,
I want **to integrate CAMEL-AI's long-term memory to store successful vulnerability discoveries**,
so that **the system learns from past successes across multiple runs**.

#### Acceptance Criteria

1. Memory system implemented in `memory/long_term_memory.py` using CAMEL-AI's memory API
2. Successful experiments (VULNERABILITY_FOUND) are stored with: hypothesis, experiment parameters, results, attack trace reference
3. Memory retrieval interface: `get_successful_attacks(task_type)` returns past successful attacks for given task
4. Hypothesis Generator queries memory at start of each run to inform initial hypothesis generation
5. Memory persisted to `outputs/memory_store/` (survives container restarts per NFR14)

### Story 2.6: RAG-Enhanced End-to-End Test

As a **researcher**,
I want **to run the complete inner loop with RAG and memory integration**,
so that **we validate that knowledge integration improves hypothesis quality**.

#### Acceptance Criteria

1. Run inner loop with RAG enabled on data-based unlearning method (5-10 iterations)
2. Verify Query Generator is called and RAG results are incorporated into hypotheses
3. Compare hypothesis quality: with RAG vs without RAG (use critic agent scoring: novelty, feasibility, rigor)
4. At least one hypothesis cites relevant retrieved papers in its description
5. If VULNERABILITY_FOUND, memory system successfully stores the result for future retrieval
6. Performance: RAG retrieval adds < 5 seconds per loop iteration (NFR8)

## Epic 3: Concept-Erasure Task & Multi-Modal Evaluation

**Expanded Goal**: Extend the system to support concept-erasure unlearning methods from GitHub repositories with VLM-based evaluation for generation-based leakage detection. Adapt prompts and agents for concept-erasure domain while maintaining unified workflow architecture. By the end of this epic, AUST should successfully run the inner loop on both data-based and concept-erasure tasks, demonstrating generality across unlearning paradigms.

### Story 3.1: Concept-Erasure Method Integration

As a **developer**,
I want **to integrate concept-erasure methods from GitHub repositories as MCP FunctionTools**,
so that **agents can programmatically trigger concept-erasure experiments**.

#### Acceptance Criteria

1. Identify and select 1-2 concept-erasure methods from GitHub (e.g., EraseDiff, concept ablation tools) for MVP
2. Clone/submodule concept-erasure repositories into `submodules/concept_erasure/`
3. MCP FunctionTool wrapper in `tools/concept_erasure_tool.py` exposes: `erase_concept(model, concept, params)` and `generate_samples(model, prompt, count)`
4. Successful test execution: erase a concept from a generative model and generate samples to verify
5. Error handling for method-specific failures, GPU issues, and invalid parameters

### Story 3.2: VLM-Based Evaluator for Concept Leakage

As a **researcher**,
I want **an Evaluator that uses VLM analysis to detect concept leakage in generated images**,
so that **the system can automatically identify when concept-erasure fails**.

#### Acceptance Criteria

1. VLM Evaluator added to `agents/evaluator.py` (or new `agents/vlm_evaluator.py`) for concept-erasure task
2. Evaluator generates test prompts designed to elicit the supposedly-erased concept
3. Evaluator analyzes generated images using VLM (via OpenRouter: GPT-4V, Claude 3.5) to detect concept presence
4. Configurable thresholds in `configs/evaluation_thresholds.yaml`: concept leakage probability, CLIP score changes (if applicable)
5. Evaluator determines: VULNERABILITY_FOUND (concept leaked), INCONCLUSIVE, or NO_VULNERABILITY (concept successfully erased)
6. Evaluation results include both VLM qualitative assessment and quantitative metrics (if available)

### Story 3.3: Prompt Adaptation for Concept-Erasure

As a **developer**,
I want **to adapt agent prompts for concept-erasure domain terminology and workflows**,
so that **agents understand the specific requirements of concept-erasure tasks**.

#### Acceptance Criteria

1. New prompt configurations in `configs/prompts_concept_erasure.yaml` for: Hypothesis Generator, Critic, Query Generator, Evaluator
2. Prompts use concept-erasure terminology: "concept leakage", "generation-based attacks", "concept resurgence", "visual probing"
3. Seed hypothesis templates for concept-erasure added to `configs/seed_hypotheses.yaml`: adversarial prompts, concept combination attacks, style transfer probing
4. Hypothesis Generator selects appropriate prompt set based on task type parameter
5. All concept-erasure-specific prompts tested with sample inputs/outputs

### Story 3.4: Concept-Erasure Experiment Executor

As a **researcher**,
I want **the Experiment Executor to support concept-erasure workflows**,
so that **hypotheses targeting concept-erasure can be executed automatically**.

#### Acceptance Criteria

1. Experiment Executor extended to handle concept-erasure hypotheses (or new executor module added)
2. Executor workflow for concept-erasure: parse hypothesis → call concept_erasure_tool to erase concept → generate test samples → pass samples to VLM Evaluator
3. Experiment parameters specific to concept-erasure are correctly translated: target concept, erasure strength, generation prompts, sample count
4. Timeout handling (30 minutes per experiment as per NFR6)
5. Results (generated images, VLM analysis, metrics) saved to `experiments/results_concept_erasure/` with unique run ID

### Story 3.5: Unified Task Type Handling

As a **developer**,
I want **the Inner Loop Orchestrator to support both data-based and concept-erasure tasks with prompt-based differentiation**,
so that **the system can switch between task types seamlessly**.

#### Acceptance Criteria

1. Loop Orchestrator modified to accept task_type parameter: "data_based" or "concept_erasure"
2. Orchestrator routes to appropriate tools and evaluators based on task_type
3. Loop configuration `configs/loop_config.yaml` specifies task-specific settings: max iterations, evaluation thresholds, target methods
4. Attack traces clearly indicate task type and use appropriate terminology
5. System can run data-based and concept-erasure tasks sequentially without code changes (only config changes)

### Story 3.6: Concept-Erasure End-to-End Test

As a **researcher**,
I want **to run the complete inner loop on a concept-erasure method**,
so that **we validate the system works for both unlearning paradigms**.

#### Acceptance Criteria

1. Execute inner loop with task_type="concept_erasure" (5-10 iterations)
2. System successfully generates hypotheses, erases concepts, generates test samples, and evaluates leakage using VLM
3. Attack trace for concept-erasure task is generated showing hypothesis evolution and VLM assessments
4. At minimum, system reaches INCONCLUSIVE or VULNERABILITY_FOUND state
5. Compare performance: data-based vs concept-erasure (loop iteration time, hypothesis quality)
6. Verify multi-modal evaluation (VLM + metrics) provides meaningful signal for vulnerability detection

## Epic 4: Outer Loop & Multi-Perspective Judging

**Expanded Goal**: Implement the Reporter agent that generates academic-format papers (Introduction, Methods, Experiments, Results, Discussion, Conclusion) with citation integration, and the multi-perspective LLM Judge system with 3-5 personas evaluating findings from different angles. By the end of this epic, AUST completes the full end-to-end autonomous research workflow, generating publishable reports with multi-faceted evaluation.

### Story 4.1: Reporter Agent - Report Structure & Template

As a **researcher**,
I want **a Reporter agent that generates academic paper structure with proper sections**,
so that **the system produces publication-ready reports automatically**.

#### Acceptance Criteria

1. Reporter agent implemented in `agents/reporter.py`
2. Report template defined in `configs/report_template.md` with sections: Introduction, Methods, Experiments, Results, Discussion, Conclusion
3. Reporter accepts inputs: attack traces (all iterations), successful experiments, evaluation results, retrieved paper references
4. Reporter generates section outlines based on inputs (e.g., Methods describes hypothesis generation + experiment execution workflow)
5. Generated report saved to `outputs/reports/report_{run_id}.md` in Markdown format
6. Report includes metadata: date, task type, target unlearning method, number of iterations

### Story 4.2: Reporter Agent - Content Generation with Citations

As a **researcher**,
I want **the Reporter to populate report sections with detailed content and cite retrieved papers**,
so that **the generated report is comprehensive and properly attributed**.

#### Acceptance Criteria

1. Reporter generates Introduction: problem statement, why unlearning security matters, objectives of the stress test
2. Reporter generates Methods: describes AUST architecture, hypothesis generation process (including RAG and critic debate), experiment execution, evaluation criteria
3. Reporter generates Experiments: details the specific unlearning method tested, datasets used, experiment parameters
4. Reporter generates Results: presents vulnerability discovered (or lack thereof), attack trace summary, evaluation metrics, key observations
5. Reporter generates Discussion: interprets findings, compares to static benchmarks (if applicable), discusses implications for unlearning robustness
6. Reporter generates Conclusion: summarizes contribution, limitations, future work
7. Citations integrated using BibTeX format references from `rag/paper_metadata.json`

### Story 4.3: Attack Trace Enhancement for Report Integration

As a **developer**,
I want **attack traces to include sufficient detail for direct inclusion in academic reports**,
so that **the Reporter can use traces as primary evidence**.

#### Acceptance Criteria

1. Attack trace format enhanced to include: iteration number, hypothesis rationale, experiment design justification, quantitative results, qualitative observations
2. Traces document hypothesis evolution showing how critic feedback improved proposals
3. Traces include failure analysis: why certain hypotheses didn't lead to vulnerabilities
4. Dual format: JSON (machine-readable) saved to `outputs/attack_traces/trace_{run_id}.json` + Markdown (human-readable) saved to `outputs/attack_traces/trace_{run_id}.md`
5. Reporter can parse and extract key sections from traces for Results and Discussion sections

### Story 4.4: Judge Persona Definitions

As a **researcher**,
I want **to define 3-5 LLM judge personas with specific evaluation criteria**,
so that **the judging system provides diverse, meaningful perspectives**.

#### Acceptance Criteria

1. Judge persona definitions created in `configs/judge_personas.yaml` with 3-5 personas:
   - Security Expert: evaluates practical exploitability, real-world attack feasibility
   - ML Researcher: evaluates novelty, methodological rigor, scientific contribution
   - Privacy Advocate: evaluates privacy implications, compliance relevance
   - Skeptical Reviewer: challenges claims, identifies weaknesses, suggests improvements
   - Industry Practitioner: evaluates deployment relevance, risk assessment utility
2. Each persona has: name, role description, evaluation criteria (bullet list), scoring dimensions (e.g., novelty 1-5, rigor 1-5)
3. Persona prompts specify tone and focus area (e.g., Security Expert is pragmatic and risk-focused)

### Story 4.5: Judge Agent Implementation

As a **researcher**,
I want **Judge agents that evaluate generated reports from their persona's perspective**,
so that **the system provides multi-faceted evaluation of findings**.

#### Acceptance Criteria

1. Judge agent implemented in `agents/judge.py` that can instantiate any persona from `judge_personas.yaml`
2. Judge accepts inputs: generated report, attack traces, experiment results
3. Judge generates structured evaluation: summary assessment, strengths, weaknesses, scoring (per persona's dimensions), recommendations
4. Each judge evaluation saved to `outputs/judgments/judge_{persona_name}_{run_id}.md`
5. Judges run independently (can be parallelized if time permits)
6. All judge outputs aggregated into `outputs/judgments/summary_{run_id}.md`

### Story 4.6: Outer Loop Orchestrator

As a **researcher**,
I want **an Outer Loop Orchestrator that coordinates Reporter → Judges workflow**,
so that **the system automatically generates and evaluates reports after vulnerability discovery**.

#### Acceptance Criteria

1. Outer Loop Orchestrator implemented in `loop/outer_loop.py`
2. Orchestrator triggered when inner loop exits with VULNERABILITY_FOUND or max iterations
3. Orchestrator workflow: call Reporter → wait for report generation → call all Judges → aggregate judgments
4. Orchestrator logs all outputs and timestamps for reproducibility
5. Final output package saved to `outputs/final_{run_id}/` containing: report, attack traces (JSON + MD), all judge evaluations, aggregated summary

### Story 4.7: End-to-End Full System Test

As a **researcher**,
I want **to run the complete AUST system end-to-end (inner loop + outer loop)**,
so that **we validate autonomous research workflow from hypothesis to judged report**.

#### Acceptance Criteria

1. Execute full system on both tasks: data-based unlearning and concept-erasure
2. Inner loop discovers at least one vulnerability per task (or reaches max iterations with meaningful attempt)
3. Reporter generates comprehensive academic-format report for each task
4. All 3-5 judges evaluate each report and provide diverse perspectives
5. Attack traces are reproducible (verified with manual re-execution of key steps)
6. System autonomy validated: ≥ 90% of workflow runs without human intervention (NFR10)
7. Performance: full system (inner + outer loop) completes within < 5 hours per task (NFR7)
8. All outputs (reports, traces, judgments) meet quality standards for paper integration

## Next Steps

### Architect Prompt

You are the Architect for the AUST (AI Scientist for Autonomous Unlearning Security Testing) project. Please review this PRD and the Project Brief (docs/brief.md) to create a comprehensive architecture document.

Focus on:
- Detailed system architecture for the two-loop design (inner research loop + outer reporting loop)
- Agent interaction patterns and data flow
- CAMEL-AI framework integration and MCP FunctionTool implementations
- Docker/Kubernetes deployment architecture with H200 GPU orchestration
- Data persistence and state management across loop iterations
- Error handling and retry strategies
- Performance optimization for meeting NFR6-NFR8 targets

Start in architecture creation mode and use both the PRD and Project Brief as your foundation.

