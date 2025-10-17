# Epic 3: Concept-Erasure Task & Multi-Modal Evaluation

**Expanded Goal**: Extend the system to support concept-erasure unlearning methods from GitHub repositories with VLM-based evaluation for generation-based leakage detection. Adapt prompts and agents for concept-erasure domain while maintaining unified workflow architecture. By the end of this epic, AUST should successfully run the inner loop on both data-based and concept-erasure tasks, demonstrating generality across unlearning paradigms.

## Story 3.1: Concept-Erasure Method Integration

As a **developer**,
I want **to integrate concept-erasure methods from GitHub repositories as MCP FunctionTools**,
so that **agents can programmatically trigger concept-erasure experiments**.

### Acceptance Criteria

1. Identify and select 1-2 concept-erasure methods from GitHub (e.g., EraseDiff, concept ablation tools) for MVP
2. Clone/submodule concept-erasure repositories into `submodules/concept_erasure/`
3. MCP FunctionTool wrapper in `tools/concept_erasure_tool.py` exposes: `erase_concept(model, concept, params)` and `generate_samples(model, prompt, count)`
4. Successful test execution: erase a concept from a generative model and generate samples to verify
5. Error handling for method-specific failures, GPU issues, and invalid parameters

## Story 3.2: VLM-Based Evaluator for Concept Leakage

As a **researcher**,
I want **an Evaluator that uses VLM analysis to detect concept leakage in generated images**,
so that **the system can automatically identify when concept-erasure fails**.

### Acceptance Criteria

1. VLM Evaluator added to `agents/evaluator.py` (or new `agents/vlm_evaluator.py`) for concept-erasure task
2. Evaluator generates test prompts designed to elicit the supposedly-erased concept
3. Evaluator analyzes generated images using VLM (via OpenRouter: GPT-4V, Claude 3.5) to detect concept presence
4. Configurable thresholds in `configs/evaluation_thresholds.yaml`: concept leakage probability, CLIP score changes (if applicable)
5. Evaluator determines: VULNERABILITY_FOUND (concept leaked), INCONCLUSIVE, or NO_VULNERABILITY (concept successfully erased)
6. Evaluation results include both VLM qualitative assessment and quantitative metrics (if available)

## Story 3.3: Prompt Adaptation for Concept-Erasure

As a **developer**,
I want **to adapt agent prompts for concept-erasure domain terminology and workflows**,
so that **agents understand the specific requirements of concept-erasure tasks**.

### Acceptance Criteria

1. New prompt configurations in `configs/prompts_concept_erasure.yaml` for: Hypothesis Generator, Critic, Query Generator, Evaluator
2. Prompts use concept-erasure terminology: "concept leakage", "generation-based attacks", "concept resurgence", "visual probing"
3. Seed hypothesis templates for concept-erasure added to `configs/seed_hypotheses.yaml`: adversarial prompts, concept combination attacks, style transfer probing
4. Hypothesis Generator selects appropriate prompt set based on task type parameter
5. All concept-erasure-specific prompts tested with sample inputs/outputs

## Story 3.4: Concept-Erasure Experiment Executor

As a **researcher**,
I want **the Experiment Executor to support concept-erasure workflows**,
so that **hypotheses targeting concept-erasure can be executed automatically**.

### Acceptance Criteria

1. Experiment Executor extended to handle concept-erasure hypotheses (or new executor module added)
2. Executor workflow for concept-erasure: parse hypothesis → call concept_erasure_tool to erase concept → generate test samples → pass samples to VLM Evaluator
3. Experiment parameters specific to concept-erasure are correctly translated: target concept, erasure strength, generation prompts, sample count
4. Timeout handling (30 minutes per experiment as per NFR6)
5. Results (generated images, VLM analysis, metrics) saved to `experiments/results_concept_erasure/` with unique run ID

## Story 3.5: Unified Task Type Handling

As a **developer**,
I want **the Inner Loop Orchestrator to support both data-based and concept-erasure tasks with prompt-based differentiation**,
so that **the system can switch between task types seamlessly**.

### Acceptance Criteria

1. Loop Orchestrator modified to accept task_type parameter: "data_based" or "concept_erasure"
2. Orchestrator routes to appropriate tools and evaluators based on task_type
3. Loop configuration `configs/loop_config.yaml` specifies task-specific settings: max iterations, evaluation thresholds, target methods
4. Attack traces clearly indicate task type and use appropriate terminology
5. System can run data-based and concept-erasure tasks sequentially without code changes (only config changes)

## Story 3.6: Concept-Erasure End-to-End Test

As a **researcher**,
I want **to run the complete inner loop on a concept-erasure method**,
so that **we validate the system works for both unlearning paradigms**.

### Acceptance Criteria

1. Execute inner loop with task_type="concept_erasure" (5-10 iterations)
2. System successfully generates hypotheses, erases concepts, generates test samples, and evaluates leakage using VLM
3. Attack trace for concept-erasure task is generated showing hypothesis evolution and VLM assessments
4. At minimum, system reaches INCONCLUSIVE or VULNERABILITY_FOUND state
5. Compare performance: data-based vs concept-erasure (loop iteration time, hypothesis quality)
6. Verify multi-modal evaluation (VLM + metrics) provides meaningful signal for vulnerability detection
