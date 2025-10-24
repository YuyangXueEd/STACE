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

Machine unlearning methods promise to remove specific data or concepts from trained models—critical for GDPR compliance, bias mitigation, and preventing misuse of generative AI. However, current evaluation approaches rely on static benchmarks that cannot anticipate creative, adaptive attacks from motivated adversaries. AUST (AI Scientist for Autonomous Unlearning Security Testing) addresses this gap by implementing an autonomous research system following a 6-step workflow:

**AUST 6-Step Workflow:**
1. **Input Validation**: Parse and validate user input (model name/details + unlearned concept, optional method)
2. **Hypothesis Generation & Critic Loop** (Story 1.5): Generate naive hypothesis from template → Critic challenges → Query Generator creates RAG queries using hypothesis+feedback → Retrieve relevant attack papers → Refine hypothesis. Run 2-round micro-loop minimum.
3. **Attack Code Synthesis & Execution** (Story 1.6a): Translate hypothesis into executable code → Execute → If fails, analyze error and self-repair (3-5 retry budget) → Generate attack results
4. **MLLM Evaluation** (Story 1.4): Use VLM to evaluate if attack succeeded (concept leaked from unlearned model)
5. **Outer Loop Iteration** (default 5 times): Feed evaluation results back to Step 2 to refine hypothesis → Steps 2-4 repeat up to 5 times to improve attack quality
6. **Report Generation & Judging** (Epic 4): After 5 outer loops, generate final report from all interactions → Multi-persona judge group evaluates → If attack successful, index as "experience" memory in RAG for future reference

This demonstrates AI conducting rigorous, end-to-end scientific research on a societally relevant privacy/compliance problem.

The project builds on comprehensive brainstorming and planning completed with the analyst, including detailed system architecture (critic-generator agents, RAG-based knowledge retrieval, multi-modal evaluation), aggressive 3-week implementation timeline, and clear MVP scope prioritization. AUST will run in Docker/Kubernetes with H200 GPUs, use CAMEL-AI in dev mode for multi-agent orchestration, and leverage OpenRouter APIs for flexible LLM/VLM access.

**Project Pivot (October 2025)**: Initial planning focused on data-based unlearning (DeepUnlearn integration, forget accuracy metrics). During Epic 1 implementation, the project pivoted to **concept-erasure unlearning for Text-to-Image (T2I) models** as the primary research direction, with ESD (Erasing Stable Diffusion) as the baseline method. This pivot better aligns with multimodal jailbreak research opportunities and leverages CAMEL-AI's vision-language capabilities (MLLM evaluation, image analysis toolkit). Data-based unlearning remains a future extension (Epic 3 or post-MVP).

### Change Log

| Date | Version | Description | Author |
|------|---------|-------------|--------|
| 2025-10-20 | 0.4 | Major workflow clarification: documented 6-step AUST pipeline (Input→Hypothesis+Critic+RAG→Code+Execute→Evaluate→Outer Loop×5→Report+Judge+Memory), clarified outer loop repeats steps 2-4 five times before final reporting, emphasized Query Generator role in RAG retrieval during hypothesis refinement | John (PM Agent) |
| 2025-10-20 | 0.3 | Align with new 6-step AUST workflow: add Input Parser (Story 1.0), add Attack Code Synthesis & Self-Repair (Story 1.6a), set default outer loop to 5, index successful memories into RAG, clarify Hypothesis↔Critic↔RAG interplay | John (PM Agent) |
| 2025-10-20 | 0.2 | Updated Epic 1 & 2 to reflect actual implementation: T2I concept-erasure focus (not data-based), Story 1.3.1/1.4/1.5 changes, Story 2.1.1/2.2 updates with Qdrant, Paper Corpus Management technical assumptions updated | John (PM Agent) |
| 2025-10-16 | 0.1 | Initial PRD creation from Project Brief | John (PM Agent) |

## Requirements

### Functional

**FR0 (Input Parser)**: The system shall validate user task inputs and require at least two elements: target model details and the unlearned content/concept. Example pattern: "A Stable Diffusion 1.4 model unlearned with concept Cat {with ESD method}" where braces denote optional info. Invalid inputs must return actionable errors.

**FR1 (Core Workflow)**: The system shall implement the 6-step AUST workflow:
1. Input validation (FR0)
2. Hypothesis Generation & Critic with RAG retrieval (FR2, FR3, FR4, FR5) - minimum 2-round micro-loop
3. Attack Code Synthesis & Execution with self-repair (FR7a) - 3-5 retry budget
4. MLLM Evaluation (FR9)
5. Outer loop: repeat steps 2-4 for 5 iterations with evaluation feedback (FR15)
6. Report generation, judging, and memory indexing (FR12, FR13, FR10)

**FR2**: The system shall provide a Hypothesis Generator agent that proposes targeted stress tests for unlearning methods based on retrieved papers and past experiment results.

**FR3**: The system shall provide a Critic Agent that debates with the Hypothesis Generator after the first iteration to challenge and improve hypothesis quality. The critic shall drive RAG query generation based on the current hypothesis and evaluator feedback; the hypothesis–critic module shall run a two-round micro-loop per iteration.

**FR4**: The system shall provide a Query Generator agent that converts evaluation feedback and hypothesis needs into RAG search queries.

**FR5**: The system shall implement a RAG system with 10-20 key papers per task domain (data-based unlearning, concept-erasure) plus a shared attack methods corpus, supporting semantic search over paper content.

**FR6**: The system shall provide an Experiment Executor that integrates DeepUnlearn (via git submodule as MCP FunctionTool) for data-based unlearning experiments.

**FR7**: The system shall provide an Experiment Executor that integrates concept-erasure methods from GitHub repositories (as MCP FunctionTools or submodules) for concept-based unlearning experiments.
**FR7a (Code Synthesis & Self-Repair)**: The system shall include an Attack Code Synthesis agent that generates executable attack code from hypotheses and retries with targeted self-repair upon execution failures (3–5 attempts).

**FR8**: The system shall provide an Evaluator that uses threshold-based metrics (forget accuracy) for data-based unlearning vulnerability detection.

**FR9**: The system shall provide an Evaluator that uses VLM-based analysis (generation-based leakage detection) for concept-erasure vulnerability detection.

**FR10**: The system shall implement a Memory System using CAMEL-AI long-term storage to capture successful vulnerability discoveries for future reference. Successful attacks shall be indexed into the RAG as "experience" chunks for future retrieval.

**FR11**: The system shall generate Attack Traces documenting step-by-step records of the inner research loop showing hypothesis evolution and vulnerability discovery paths.

**FR12**: The system shall provide a Reporter agent that generates academic-format reports (Introduction, Methods, Experiments, Results, Discussion, Conclusion) with citation integration from retrieved papers.

**FR13**: The system shall provide 3-5 Judge LLM personas that evaluate findings from multiple perspectives (novelty, rigor, reproducibility, impact, exploitability).

**FR14**: The system shall support both data-based unlearning and concept-erasure tasks using unified workflow with prompt-based task differentiation.

**FR15 (Outer Loop Orchestration)**: The system shall implement a 5-iteration outer loop that repeats Steps 2-4 of the workflow:
- **Step 2**: Hypothesis Generation & Critic Loop with RAG-based paper retrieval
- **Step 3**: Attack Code Synthesis & Execution with self-repair
- **Step 4**: MLLM Evaluation feeding results back to next iteration

After 5 outer loop iterations complete, proceed to **Step 6** (Report Generation with all interaction history → Multi-Persona Judging → Memory Indexing if successful).

### Non Functional

**NFR1**: The system shall run in Docker containers orchestrated by Kubernetes with H200 GPU access via job.yaml configuration.

**NFR2**: The system shall use CAMEL-AI installed in dev/editable mode (`pip install -e`) to allow source code modifications as needed.

**NFR3**: The system shall integrate DeepUnlearn as a git submodule for version control and local modifications.

**NFR4**: The system shall use OpenRouter API for LLM/VLM access supporting multiple models (GPT-5, Claude 3.5 Sonnet, etc.).

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

- **Paper Corpus Management**: 101 paper cards (structured markdown with Methodology, Experiments, Results sections) generated in Story 2.1.1 from collected PDFs. RAG system uses **Qdrant** for vector database with **SentenceTransformers** (all-MiniLM-L6-v2, 384-dim) for local embeddings. Section-level chunking (~400-500 chunks) with metadata filtering (section type, task type, attack level). Decision rationale: Vector RAG (not GraphRAG) meets MVP timeline and semantic search requirements; Qdrant chosen for integrated metadata storage, local persistence, CAMEL-AI compatibility, and <5s retrieval (NFR8).
 - **Paper Corpus Management**: 101 paper cards (structured markdown with Methodology, Experiments, Results sections) generated in Story 2.1.1 from collected PDFs. RAG system uses **Qdrant** for vector database with **SentenceTransformers** (all-MiniLM-L6-v2, 384-dim) for local embeddings. Section-level chunking (~400-500 chunks) with metadata filtering (section type, task type, attack level). Decision rationale: Vector RAG (not GraphRAG) meets MVP timeline and semantic search requirements; Qdrant chosen for integrated metadata storage, local persistence, CAMEL-AI compatibility, and <5s retrieval (NFR8). In addition to papers, successful memories are indexed as an "experience" section for retrieval.

- **Agent Prompting Strategy**: Agents use prompt-based task differentiation for data-based vs concept-erasure workflows. Prompts stored in `aust/configs/` directory for easy iteration and versioning.

- **Seed Hypothesis Templates**: Pre-load 3-5 known attack patterns (membership inference, model inversion, data extraction) as seed templates to mitigate hypothesis quality risk (identified as HIGH risk in brief).

- **Attack Trace Format**: Attack traces output in dual format - JSON (machine-readable for analysis) + Markdown (human-readable for paper integration and user reproduction).

- **Judge Persona Definitions**: 3-5 judge personas pre-defined: Security Expert, ML Researcher, Privacy Advocate, Skeptical Reviewer, Industry Practitioner. Each has specific evaluation criteria and perspective.

- **Evaluation Metric Thresholds**: Configurable thresholds in `aust/configs/` - data-based: forget accuracy delta; concept-erasure: generation-based leakage probability, CLIP score changes.

- **Error Handling & Logging**: Comprehensive logging for debugging (all agent interactions, API calls, experiment results). Graceful degradation for API failures, GPU unavailability, or tool integration errors.

- **Output Persistence**: All outputs (reports, attack traces, judge evaluations, experiment results) saved to persistent Kubernetes volumes to survive container restarts and enable post-hoc analysis.

## Epic List

### Epic 1: Foundation & Inner Loop MVP (Week 1: Oct 24-30)
**Goal**: Establish project infrastructure, CAMEL-AI integration, concept-erasure integration (ESD baseline), Input Parser (Step 1), and Attack Code Synthesis with self-repair (Step 3); implement complete inner research loop components for concept-erasure focus.

### Epic 2: RAG System & Knowledge Integration (Week 2: Oct 31-Nov 6, Part 1)
**Goal**: Implement Hypothesis Generation & Critic Loop with RAG-based paper retrieval (Step 2), integrate Query Generator for literature search, and add CAMEL-AI long-term memory for experience indexing.

### Epic 3: Concept-Erasure Task & Multi-Modal Evaluation (Week 2: Oct 31-Nov 6, Part 2)
**Goal**: Implement MLLM-based evaluation for concept leakage detection (Step 4), enabling automated assessment of attack success for concept-erasure unlearning methods.

### Epic 4: Outer Loop & Multi-Perspective Judging (Week 3: Nov 7-13)
**Goal**: Implement 5-iteration outer loop orchestration (Step 5) and final report generation with multi-perspective judge evaluation and memory indexing (Step 6), completing the end-to-end autonomous research workflow.

## Epic 1: Foundation & Inner Loop MVP

**Status**: ✅ Partially Complete (Stories 1.3.1 Ready for Review; 1.4 Completed; 1.5 In Progress; 1.0 New; 1.6a New; 1.1-1.2, 1.6-1.8 TBD)

**Expanded Goal**: Establish the foundational project infrastructure including Docker/Kubernetes setup, CAMEL-AI framework integration in dev mode, concept-erasure toolkits (ESD unlearning), strict Task Input Parser (**Step 1: Input Validation**), and attack code synthesis with self-repair (**Step 3: Code Synthesis & Execution**). Implement a complete hypothesis refinement workforce with evaluation capabilities for concept-erasure unlearning methods. **Note**: Focus shifted from data-based unlearning (DeepUnlearn) to concept-erasure unlearning (T2I models) based on project pivot.

**Key Changes from Original Plan:**
- **Story 1.3** replaced by **Story 1.3.1** (T2I Unlearning Attack Pipeline with ESD integration) ✅ Ready for Review
- **Story 1.4** updated to MLLM Unlearning Toolkit & Evaluator ✅ Completed
- **Story 1.5** expanded from simple Critic Agent to Hypothesis Refinement Workforce (Draft; includes Query Generator integration during debate)
- **Story 1.0 (New)** Input Parser & Task Validation
- **Story 1.6a (New)** Attack Code Synthesis & Self-Repair Loop
- Stories 1.1-1.2, 1.6-1.8 remain in backlog (infrastructure, orchestrator)
### Story 1.0: Task Parser & Input Validation (New)

As a developer,
I want to validate and parse user inputs into a structured TaskSpec,
so that the system only runs when at least the model details and the unlearned concept/content are specified.

#### Acceptance Criteria

1. Input Parser implemented in `aust/src/loop/task_parser.py` with Pydantic TaskSpec model
2. Accepts natural language prompt and extracts: model_name/version, unlearning_target (concept/content), optional method info
3. Validates required fields and returns actionable errors on failure
4. Parses example pattern: "A Stable Diffusion 1.4 model unlearned with concept Cat {with ESD method}"
5. Emits normalized TaskSpec JSON into `aust/outputs/{task_id}/task_spec.json`
6. Integrated pre-check in Inner Loop Orchestrator; loop does not start if validation fails


### Story 1.1: Project Setup & Infrastructure

As a **developer**,
I want **to set up the CAUST repository with Docker/Kubernetes configuration, Python 3.11+ environment, and basic project structure**,
so that **the team has a working development environment ready for agent implementation**.

#### Acceptance Criteria

1. Repository at https://github.com/vios-s/CAUST is initialized with directory structure: `aust/src/agents`, `aust/src/toolkits`, `aust/src/rag`, `aust/src/memory`, `aust/src/loop`, `aust/outputs`, `aust/configs`, `aust/experiments`, `external/`, `docker/`
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
3. OpenRouter API integration is tested with at least one model call (GPT-5 or Claude 3.5)
4. Agent prompt configuration system is implemented in `aust/configs/` directory (YAML or JSON format)
5. Basic agent interaction logging captures all LLM API calls and responses

### Story 1.3.1: T2I Unlearning Attack Pipeline ✅ Ready for Review

**Note**: This story replaces original Story 1.3 (DeepUnlearn Integration). Project pivoted to concept-erasure unlearning for T2I models instead of data-based unlearning.

As a **researcher**,
I want **to integrate ESD unlearning method as an MCP toolkit and create an agent to orchestrate concept erasure across multiple T2I models**,
so that **I can programmatically erase concepts from text-to-image models (Stable Diffusion, SDXL, Flux) via natural language prompts**.

#### Acceptance Criteria

1. ESD repository cloned to `external/esd/` (from official GitHub repository)
2. ConceptUnlearnToolkit in `aust/src/toolkits/concept_unlearn_toolkit.py` wraps ESD as MCP toolkit with FunctionTools supporting multiple T2I models
3. Support for T2I models: Stable Diffusion (default), Stable Diffusion XL (SDXL), and Flux
4. ConceptUnlearnAgent in `aust/src/agents/concept_unlearn_agent.py` receives prompts like "using {unlearning_method} to erase {concept} [from {model}] [with {method_variant}]" and triggers appropriate tool calls
5. Support for ESD method with optional method variants (e.g., 'xattn', 'noxattn', 'full')
6. Default parameters when model/variant not specified in prompt (model: Stable Diffusion, variant: 'xattn' for ESD)
7. Model-specific configuration and checkpoint management (`data/unlearned_models/esd/{model}/{concept}/`)
8. Toolkit designed to be extensible for future unlearning methods (SalUn, Receler, etc.) and T2I models

**See**: [docs/stories/1.3.1.t2i-unlearning-attack-pipeline.md](docs/stories/1.3.1.t2i-unlearning-attack-pipeline.md)

### Story 1.4: Concept Unlearning Evaluation Toolkit & MLLM Evaluator Agent ✅ Completed

**Note**: This story was updated from original "Hypothesis Generator Agent" to focus on evaluation capabilities for concept-erasure unlearning.

As a **researcher**,
I want **a comprehensive evaluation toolkit with metrics from Awesome-Multimodal-Jailbreak and an MLLM-powered evaluator agent**,
so that **I can systematically assess whether concepts have been successfully unlearned from text-to-image models and evaluate their robustness against jailbreak attacks**.

#### Acceptance Criteria

1. ConceptUnlearnEvaluationToolkit in `aust/src/toolkits/concept_unlearn_evaluation_toolkit.py` implements evaluation metrics from Awesome-Multimodal-Jailbreak repository
2. Robustness metrics: Attack Success Rate (ASR), concept leakage detection (CLIP-based, detector-based)
3. Utility metrics: FID (Frechet Inception Distance), CLIP Score, Prompt Perplexity
4. MLLM assessment using Large Vision Language models to recognize whether the concept is leaked
5. Integration with `camel.toolkits.ImageAnalysisToolkit` for image-based analysis
6. Evaluator agent in `aust/src/agents/mllm_evaluator.py` orchestrates evaluation workflow
7. Evaluation script `aust/scripts/run_mllm_evaluation.py` provides CLI for evaluating unlearned models

**See**: [docs/stories/1.4.mllm-unlearning-toolkit-evaluator.md](docs/stories/1.4.mllm-unlearning-toolkit-evaluator.md)

### Story 1.5: Hypothesis Refinement Workforce (Draft, In Progress)

**Note**: This story expanded from original "Critic Agent" to a full workforce pattern with iterative debate and Chain-of-Thought reasoning.

As a **researcher**,
I want **a collaborative Hypothesis Refinement Workforce that uses iterative debate between a Hypothesis Generator and Critic agent with self-improving Chain-of-Thought reasoning**,
so that **the system autonomously generates high-quality, testable vulnerability hypotheses through structured adversarial refinement**.

#### Acceptance Criteria

1. Hypothesis Refinement Workforce implemented as a cohesive module in `aust/src/agents/hypothesis_workforce.py`
2. Workforce uses CAMEL-AI Workforce pattern to coordinate Hypothesis Generator and Critic agents
3. Workforce implements 2-round debate cycle with self-improving CoT reasoning
4. Hypothesis Generator accepts context inputs: task type, past experiment results, evaluator feedback, retrieved papers chunks from RAG
5. Critic triggers the Query Generator to produce 1–3 RAG queries based on current hypothesis; retrieved chunks are fed into refinement
6. In the first round, the generator uses seed templates (3-5 known attack patterns) with LLM-based variation
6. Hypothesis data model with structured format: attack method, target, experiment parameters, expected outcome, confidence score, novelty score, reasoning trace
7. Critic Agent challenges hypotheses on novelty, feasibility, and rigor dimensions
8. Critic provides structured feedback: strengths, weaknesses, specific suggestions for improvement
9. Hypothesis Generator incorporates critic feedback using CoT reasoning
10. Workforce activates after first inner loop iteration (when feedback from evaluator is available)
11. Full debate exchange logged to `aust/outputs/{task_id}/debates/iteration_{n}.json`
12. Final hypothesis from workforce integrates with existing Inner Loop Orchestrator state machine

**See**: [docs/stories/1.5.hypothesis-refinement-workforce.md](docs/stories/1.5.hypothesis-refinement-workforce.md)

### Story 1.6: Experiment Executor for Data-Based Unlearning

As a **researcher**,
I want **an Experiment Executor that runs unlearning experiments using DeepUnlearn**,
so that **hypotheses can be tested automatically on real unlearning methods**.

#### Acceptance Criteria

1. Experiment Executor implemented in `aust/src/agents/experiment_executor.py`
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

1. Evaluator implemented in `aust/src/agents/evaluator.py` for data-based unlearning
2. Evaluator computes forget accuracy from experiment results using DeepUnlearn's evaluation functions
3. Configurable threshold in `aust/configs/evaluation_thresholds.yaml` (e.g., forget accuracy delta > 10%)
4. Evaluator determines: VULNERABILITY_FOUND, INCONCLUSIVE, or NO_VULNERABILITY
5. Evaluator generates structured feedback: what worked, what failed, suggestions for next hypothesis
6. Evaluation results logged to `aust/outputs/evaluations/` with experiment run ID reference

### Story 1.8: Inner Loop Orchestrator

As a **researcher**,
I want **an Inner Loop Orchestrator that coordinates the research cycle**,
so that **the system autonomously iterates until a vulnerability is discovered or max iterations reached**.

#### Acceptance Criteria

1. Loop Orchestrator implemented in `aust/src/loop/inner_loop_orchestrator.py`
2. Orchestrator manages loop state: iteration count, experiment history, current hypothesis, feedback
3. Loop flow: Hypothesis Generator → (Critic if iteration > 1) → Experiment Executor → Evaluator → check exit condition
4. Exit conditions: VULNERABILITY_FOUND OR iteration count >= 10 (configurable in `aust/configs/loop_config.yaml`)
5. Loop state persisted to `aust/outputs/loop_state.json` after each iteration (supports restart)
6. Attack trace generation: each iteration's hypothesis, experiment parameters, results, and feedback appended to `aust/outputs/attack_traces/trace_{run_id}.md`

### Story 1.9: End-to-End Inner Loop Test
### Story 1.6a: Attack Code Synthesis & Self-Repair Loop (New)

As a researcher,
I want an agent that translates hypotheses into executable attack code/scripts and automatically repairs code on failures,
so that failed executions are retried up to 3–5 times before giving up.

#### Acceptance Criteria

1. Code Synthesis agent implemented in `aust/src/agents/code_synthesizer.py` with prompts tailored for our toolkits and environment
2. Execution runner integrated with sandboxed subprocess and timeouts; logs captured to `aust/outputs/{task_id}/runs/{run_id}/`
3. On failure (non-zero exit, exception), the agent inspects error logs and regenerates code with targeted fixes; retry budget configurable (3–5)
4. Success/failure with artifacts and logs reported back to Inner Loop Orchestrator
5. Works for both concept-erasure attack scripts and data-based DeepUnlearn wrappers


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

**Status**: ✅ Partially Complete (Story 2.1.1 done; Story 2.2 ready for implementation; 2.3-2.5 TBD)

**Expanded Goal**: Implement Hypothesis Generation & Critic Loop with RAG-based paper retrieval (**Step 2**) to enhance hypothesis quality through literature-based knowledge. Integrate Query Generator for transforming hypothesis+feedback into search queries. Add CAMEL-AI long-term memory to store successful vulnerability discoveries (**Step 6** memory indexing) for future reference. By the end of this epic, hypothesis generation should leverage relevant research papers and past successes, significantly improving the quality and novelty of proposed stress tests.

**Key Changes from Original Plan:**
- **Story 2.1** replaced by **Story 2.1.1** (Paper Card Generation for RAG System) ✅ Completed - 101 papers processed into structured markdown cards
- **Story 2.2** updated to use **Qdrant** (not FAISS/Chroma) with SentenceTransformers for local embeddings ⏳ Ready for Implementation
- Successful memories from Story 2.5 will be indexed into Qdrant as "experience" chunks

### Story 2.1.1: Paper Card Generation for RAG System ✅ Completed

**Note**: This story replaces original Story 2.1 (Paper Corpus Collection & Storage). Instead of storing raw PDFs, we generate AI-extracted structured paper cards for better RAG retrieval.

As a **researcher**,
I want **an AI agent to generate structured paper cards from collected PDFs that extract key information (methodology, experiments, results, GitHub links) for RAG retrieval**,
so that **the hypothesis generation agent can efficiently retrieve relevant research insights without processing messy unstructured PDFs**.

#### Acceptance Criteria

1. Paper card markdown template created in `.paper_cards/TEMPLATE.md` with sections: Metadata, Quick Summary, Research Problem, Methodology, Experiment Design, Key Results Summary, Implementation Details, Relevance to Our Work, Key Quotes, Related Work Mentioned, Citation
2. Template includes metadata fields: generation_date, agent_model for traceability
3. Paper Card Agent implemented in `aust/src/agents/paper_card_agent.py` extending CAMEL-AI ChatAgent
4. Agent uses LLM (gpt-5-nano via OpenRouter) to extract structured information from PDFs
5. Agent prompt configuration in `aust/configs/prompts/paper_card_extraction.yaml`
6. Agent handles PDF processing edge cases (figures, tables, references)
7. Batch processing script `scripts/generate_paper_cards.py` generates cards for all 101 papers
8. Paper card metadata file `.paper_cards/card_metadata.json` tracks generation status
9. Error handling and logging for failed extractions
10. Paper cards designed for RAG chunking: each section (Methodology, Experiments, Results) can be embedded separately

**Achievements:**
- ✅ 101 paper cards generated in `.paper_cards/` directory
- ✅ Organized by taxonomy: `any-to-t/` and `any-to-v/` with subdirectories by attack level
- ✅ Structured markdown format ready for section-based chunking in Story 2.2

**See**: [docs/stories/2.1.1.paper-corpus-collection-storage.md](docs/stories/2.1.1.paper-corpus-collection-storage.md)

### Story 2.2: Vector Database & Embedding System ⏳ Ready for Implementation

**Note**: Updated to use **Qdrant** vector database with local SentenceTransformers embeddings (not OpenRouter).

As a **developer**,
I want **to set up a Qdrant vector database with local embeddings for semantic search over paper cards**,
so that **agents can retrieve relevant research based on query similarity to support hypothesis generation**.

#### Acceptance Criteria

1. Qdrant collection created with collection name `"aust_papers"`, 384-dim vectors (all-MiniLM-L6-v2), local disk persistence at `rag/vector_index/`
2. Paper card chunking by semantic sections: Methodology, Experiments, Results, Relevance (~400-500 chunks from 101 cards)
3. Each chunk includes paper title prefix and metadata payload: arxiv_id, section, task_type, attack_level, paper_title, card_path
4. Index building script `scripts/build_vector_index.py` processes all paper cards with progress logging
5. PaperRAG query interface: `search(query, top_k=5, section_filter, task_type_filter)` with Qdrant filtering DSL
6. Search performance: < 5 seconds per query, < 10 seconds for 3-query batch (NFR8)
7. Story 2.3 integration ready: PaperRAG class importable from `aust.rag.vector_db`

**Technical Decisions:**
- **Vector RAG** (not GraphRAG) - meets MVP timeline and semantic search requirements
- **Qdrant** chosen for integrated metadata storage, local persistence, CAMEL-AI compatibility
- **Section-level chunking** - leverages structured paper cards from Story 2.1.1

**See**: [docs/stories/2.2.vector-database-embedding-system.md](docs/stories/2.2.vector-database-embedding-system.md)

### Story 2.3: Query Generator Agent (TBD)

As a **researcher**,
I want **a Query Generator agent that converts evaluation feedback and hypothesis needs into RAG search queries**,
so that **the system retrieves relevant papers automatically from the Qdrant-based RAG system**.

#### Acceptance Criteria

1. Query Generator agent implemented in `aust/src/agents/query_generator.py`
2. Agent accepts inputs: current hypothesis (if any), evaluation feedback, task type (concept-erasure or data-based), iteration number
3. Agent generates 1-3 search queries focusing on: attack methods, unlearning vulnerabilities, relevant evaluation metrics
4. Queries are logged to `aust/outputs/{task_id}/queries/iteration_{n}.json` with timestamp
5. Query Generator calls `PaperRAG.search()` interface from Story 2.2:
   - Uses section filtering (e.g., `section_filter="METHODOLOGY"` for attack methods)
   - Uses task type filtering (e.g., `task_type_filter="any-to-t"` for T2I papers)
   - Returns top-5 relevant paper chunks per query (15 chunks total for 3 queries)
6. Integration with Hypothesis Refinement Workforce (Story 1.5): retrieved papers passed as context

**Dependencies:**
- Story 2.2 (PaperRAG interface) must be completed first
- Story 1.5 (Hypothesis Workforce) integration point

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
5. Memory persisted to `aust/outputs/memory_store/` (survives container restarts per NFR14)
6. Successful memories are exported to the Qdrant collection `aust_papers` with section="experience" for PaperRAG retrieval

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

**Expanded Goal**: Implement MLLM-based evaluation for concept leakage detection (**Step 4: MLLM Evaluation**). Extend the system to support concept-erasure unlearning methods from GitHub repositories with VLM-based evaluation for generation-based leakage detection. Adapt prompts and agents for concept-erasure domain while maintaining unified workflow architecture. By the end of this epic, AUST should successfully run the complete inner loop (Steps 2-4) on concept-erasure tasks.

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

1. VLM Evaluator added to `aust/src/agents/evaluator.py` (or new `aust/src/agents/vlm_evaluator.py`) for concept-erasure task
2. Evaluator generates test prompts designed to elicit the supposedly-erased concept
3. Evaluator analyzes generated images using VLM (via OpenRouter: GPT-4V, Claude 3.5) to detect concept presence
4. Configurable thresholds in `aust/configs/evaluation_thresholds.yaml`: concept leakage probability, CLIP score changes (if applicable)
5. Evaluator determines: VULNERABILITY_FOUND (concept leaked), INCONCLUSIVE, or NO_VULNERABILITY (concept successfully erased)
6. Evaluation results include both VLM qualitative assessment and quantitative metrics (if available)

### Story 3.3: Prompt Adaptation for Concept-Erasure

As a **developer**,
I want **to adapt agent prompts for concept-erasure domain terminology and workflows**,
so that **agents understand the specific requirements of concept-erasure tasks**.

#### Acceptance Criteria

1. New prompt configurations in `aust/configs/prompts_concept_erasure.yaml` for: Hypothesis Generator, Critic, Query Generator, Evaluator
2. Prompts use concept-erasure terminology: "concept leakage", "generation-based attacks", "concept resurgence", "visual probing"
3. Seed hypothesis templates for concept-erasure added to `aust/configs/seed_hypotheses.yaml`: adversarial prompts, concept combination attacks, style transfer probing
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
3. Loop configuration `aust/configs/loop_config.yaml` specifies task-specific settings: max iterations, evaluation thresholds, target methods
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

**Expanded Goal**: Implement 5-iteration outer loop orchestration (**Step 5**) that feeds evaluation results back to refine hypothesis generation (Steps 2-4). After outer loop completes, implement final report generation, multi-perspective LLM Judge system with 3-5 personas, and memory indexing for successful attacks (**Step 6**). By the end of this epic, AUST completes the full end-to-end autonomous research workflow (all 6 steps), generating publishable reports with multi-faceted evaluation and learning from successes.

### Story 4.1: Reporter Agent - Report Structure & Template

As a **researcher**,
I want **a Reporter agent that generates academic paper structure with proper sections**,
so that **the system produces publication-ready reports automatically**.

#### Acceptance Criteria

1. Reporter agent implemented in `aust/src/agents/reporter.py`
2. Report template defined in `aust/configs/report_template.md` with sections: Introduction, Methods, Experiments, Results, Discussion, Conclusion
3. Reporter accepts inputs: attack traces (all iterations), successful experiments, evaluation results, retrieved paper references
4. Reporter generates section outlines based on inputs (e.g., Methods describes hypothesis generation + experiment execution workflow)
5. Generated report saved to `aust/outputs/reports/report_{run_id}.md` in Markdown format
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
4. Dual format: JSON (machine-readable) saved to `aust/outputs/attack_traces/trace_{run_id}.json` + Markdown (human-readable) saved to `aust/outputs/attack_traces/trace_{run_id}.md`
5. Reporter can parse and extract key sections from traces for Results and Discussion sections

### Story 4.4: Judge Persona Definitions

As a **researcher**,
I want **to define 3-5 LLM judge personas with specific evaluation criteria**,
so that **the judging system provides diverse, meaningful perspectives**.

#### Acceptance Criteria

1. Judge persona definitions created in `aust/configs/personas/judge_personas.yaml` with 3-5 personas:
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

1. Judge agent implemented in `aust/src/agents/judge.py` that can instantiate any persona from `aust/configs/personas/judge_personas.yaml`
2. Judge accepts inputs: generated report, attack traces, experiment results
3. Judge generates structured evaluation: summary assessment, strengths, weaknesses, scoring (per persona's dimensions), recommendations
4. Each judge evaluation saved to `aust/outputs/judgments/judge_{persona_name}_{run_id}.md`
5. Judges run independently (can be parallelized if time permits)
6. All judge outputs aggregated into `aust/outputs/judgments/summary_{run_id}.md`

### Story 4.6: Outer Loop Orchestrator

As a **researcher**,
I want **an Outer Loop Orchestrator that coordinates Reporter → Judges workflow**,
so that **the system automatically generates and evaluates reports after vulnerability discovery**.

#### Acceptance Criteria

1. Outer Loop Orchestrator implemented in `aust/src/loop/outer_loop.py`
2. Orchestrator triggered when inner loop exits with VULNERABILITY_FOUND or max iterations
3. Orchestrator workflow: call Reporter → wait for report generation → call all Judges → aggregate judgments
4. Orchestrator logs all outputs and timestamps for reproducibility
5. Default outer loop iteration budget is 5; configurable via `aust/configs/loop_config.yaml`
6. Final output package saved to `aust/outputs/final_{run_id}/` containing: report, attack traces (JSON + MD), all judge evaluations, aggregated summary

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
