# Requirements

## Functional

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

## Non Functional

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
