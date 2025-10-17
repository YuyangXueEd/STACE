# Epic 1: Foundation & Inner Loop MVP

**Expanded Goal**: Establish the foundational project infrastructure including Docker/Kubernetes setup, CAMEL-AI framework integration in dev mode, DeepUnlearn as a git submodule, and implement a complete inner research loop (Hypothesis Generator → Critic → Experiment Executor → Evaluator → Feedback) for data-based unlearning only. By the end of this epic, the system should be able to run 3-5 manual loop iterations and attempt to discover at least one vulnerability in a data-based unlearning method.

## Story 1.1: Project Setup & Infrastructure

As a **developer**,
I want **to set up the CAUST repository with Docker/Kubernetes configuration, Python 3.11+ environment, and basic project structure**,
so that **the team has a working development environment ready for agent implementation**.

### Acceptance Criteria

1. Repository at https://github.com/vios-s/CAUST is initialized with directory structure: `agents/`, `tools/`, `rag/`, `memory/`, `loop/`, `outputs/`, `configs/`, `experiments/`, `submodules/`, `external/`, `docker/`
2. Docker conda_Dockerfile builds successfully with Python 3.11+ and core dependencies (PyTorch, CAMEL-AI requirements)
3. Kubernetes job.yaml is configured for H200 GPU access and persistent volume mounts for outputs
4. `requirements.txt` includes pinned versions of core dependencies (PyTorch, CAMEL-AI, OpenRouter client)
5. Basic logging framework is configured (Python logging to file + console)
6. README.md documents setup instructions and project structure

## Story 1.2: CAMEL-AI Integration in Dev Mode

As a **developer**,
I want **to integrate CAMEL-AI in dev/editable mode and verify basic agent functionality**,
so that **we can create and modify agents as needed throughout development**.

### Acceptance Criteria

1. CAMEL-AI is cloned/installed in `external/camel/` with `pip install -e external/camel`
2. Basic test agent (simple echo agent) successfully instantiates using CAMEL-AI's agent API
3. OpenRouter API integration is tested with at least one model call (GPT-4o or Claude 3.5)
4. Agent prompt configuration system is implemented in `configs/` directory (YAML or JSON format)
5. Basic agent interaction logging captures all LLM API calls and responses

## Story 1.3: DeepUnlearn Integration as Git Submodule

As a **developer**,
I want **to integrate DeepUnlearn as a git submodule and wrap it as a CAMEL-AI MCP FunctionTool**,
so that **agents can programmatically trigger unlearning and evaluation experiments**.

### Acceptance Criteria

1. DeepUnlearn repository is added as git submodule in `submodules/DeepUnlearn/`
2. DeepUnlearn dependencies are integrated into requirements.txt without conflicts
3. MCP FunctionTool wrapper in `tools/deepunlearn_tool.py` exposes at minimum: `unlearn_model(method, dataset, params)` and `evaluate_model(model, metrics)`
4. Successful test execution: trigger unlearning on a simple dataset and verify evaluation metrics are returned
5. Error handling for GPU unavailability, invalid parameters, and DeepUnlearn failures

## Story 1.4: Hypothesis Generator Agent

As a **researcher**,
I want **a Hypothesis Generator agent that proposes stress tests for data-based unlearning methods**,
so that **the system can autonomously generate testable vulnerability hypotheses**.

### Acceptance Criteria

1. Hypothesis Generator agent implemented in `agents/hypothesis_generator.py`
2. Agent accepts context inputs: task type (data-based), past experiment results (empty for first iteration), feedback from evaluator
3. Agent generates hypothesis in structured format: attack method, target unlearning method, experiment parameters, expected outcome
4. Seed templates (3-5 known attack patterns: membership inference, model inversion, data extraction) are pre-loaded in `configs/seed_hypotheses.yaml`
5. For MVP, hypothesis generation uses seed templates + basic LLM variation (no RAG yet - deferred to Epic 2)
6. Output hypothesis is logged to `outputs/hypotheses/` with timestamp

## Story 1.5: Critic Agent

As a **researcher**,
I want **a Critic Agent that debates with the Hypothesis Generator after the first iteration**,
so that **hypothesis quality improves through adversarial questioning**.

### Acceptance Criteria

1. Critic Agent implemented in `agents/critic.py`
2. Critic activates after first inner loop iteration (when feedback is available)
3. Critic challenges hypothesis on: novelty (is this just repeating seed templates?), feasibility (can this be executed?), rigor (is the expected outcome testable?)
4. Critic provides structured feedback to Hypothesis Generator: strengths, weaknesses, suggestions for improvement
5. Hypothesis Generator incorporates critic feedback in next iteration
6. Debate exchange (Hypothesis → Critic → Revised Hypothesis) is logged to `outputs/debates/`

## Story 1.6: Experiment Executor for Data-Based Unlearning

As a **researcher**,
I want **an Experiment Executor that runs unlearning experiments using DeepUnlearn**,
so that **hypotheses can be tested automatically on real unlearning methods**.

### Acceptance Criteria

1. Experiment Executor implemented in `agents/experiment_executor.py`
2. Executor parses hypothesis structure and translates to DeepUnlearn FunctionTool calls
3. Executor handles experiment execution: trigger unlearning, wait for completion, collect results
4. Timeout handling (max 30 minutes per experiment as per NFR6) with graceful failure
5. Experiment results (metrics, model artifacts, logs) saved to `experiments/results/` with unique run ID
6. Executor reports success/failure status and metrics back to loop orchestrator

## Story 1.7: Evaluator with Threshold-Based Metrics

As a **researcher**,
I want **an Evaluator that assesses experiment results using forget accuracy thresholds**,
so that **the system can automatically detect when vulnerabilities are discovered**.

### Acceptance Criteria

1. Evaluator implemented in `agents/evaluator.py` for data-based unlearning
2. Evaluator computes forget accuracy from experiment results using DeepUnlearn's evaluation functions
3. Configurable threshold in `configs/evaluation_thresholds.yaml` (e.g., forget accuracy delta > 10%)
4. Evaluator determines: VULNERABILITY_FOUND, INCONCLUSIVE, or NO_VULNERABILITY
5. Evaluator generates structured feedback: what worked, what failed, suggestions for next hypothesis
6. Evaluation results logged to `outputs/evaluations/` with experiment run ID reference

## Story 1.8: Inner Loop Orchestrator

As a **researcher**,
I want **an Inner Loop Orchestrator that coordinates the research cycle**,
so that **the system autonomously iterates until a vulnerability is discovered or max iterations reached**.

### Acceptance Criteria

1. Loop Orchestrator implemented in `loop/inner_loop.py`
2. Orchestrator manages loop state: iteration count, experiment history, current hypothesis, feedback
3. Loop flow: Hypothesis Generator → (Critic if iteration > 1) → Experiment Executor → Evaluator → check exit condition
4. Exit conditions: VULNERABILITY_FOUND OR iteration count >= 10 (configurable in `configs/loop_config.yaml`)
5. Loop state persisted to `outputs/loop_state.json` after each iteration (supports restart)
6. Attack trace generation: each iteration's hypothesis, experiment parameters, results, and feedback appended to `outputs/attack_traces/trace_{run_id}.md`

## Story 1.9: End-to-End Inner Loop Test

As a **researcher**,
I want **to run the complete inner loop end-to-end on a data-based unlearning method**,
so that **we validate the MVP can discover at least one vulnerability**.

### Acceptance Criteria

1. Manual execution script `run_inner_loop.py` triggers full loop with configurable task (data-based unlearning method selection)
2. System runs 3-5 iterations successfully without crashes
3. Attack trace file is generated in Markdown format documenting all iterations
4. At minimum, system reaches INCONCLUSIVE or VULNERABILITY_FOUND state (not just failures)
5. All agent interactions, experiment results, and evaluations are logged for debugging
6. Performance: loop iteration completes in < 30 minutes per cycle (NFR6)
