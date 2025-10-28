"""
Inner Loop Orchestrator for AUST Research Cycle (Stories 1.0-1.8).

This orchestrator coordinates the full inner loop workflow:
1. Task initialization and configuration
2. RAG-enhanced hypothesis generation (Story 1.5)
3. Multi-agent debate refinement (Story 1.5)
4. Experiment execution (Story 1.6a - placeholder)
5. Result evaluation (Story 1.7 - placeholder)
6. State persistence and attack trace generation
7. Exit condition checking

Integrates:
- PaperRAG (Story 1.2-1.4) for literature-grounded hypothesis generation
- HypothesisRefinementWorkforce (Story 1.5) for debate-based refinement
- ConfigLoader for seed templates and prompts
- InnerLoopState for state management
"""

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from aust.src.agents.hypothesis_workforce import HypothesisRefinementWorkforce
from aust.src.agents.query_generator import QueryGeneratorAgent
from aust.src.utils.logging_config import get_logger
from aust.src.utils.config_loader import ConfigLoader
from aust.src.data_models import (
    DebateSession,
    ExitCondition,
    HypothesisContext,
    InnerLoopState,
    IterationResult,
    TaskSpec,
)
from aust.src.rag.vector_db import PaperRAG

logger = get_logger(__name__)


class InnerLoopOrchestrator:
    """
    Orchestrates the complete inner loop research cycle.

    The inner loop iteratively:
    - Generates hypotheses using RAG-retrieved papers
    - Refines hypotheses through multi-agent debate
    - Executes experiments to test hypotheses
    - Evaluates results and provides feedback
    - Persists state and generates attack traces
    - Checks exit conditions
    """

    def __init__(
        self,
        task_id: Optional[str] = None,
        task_type: str = "concept_erasure",
        task_description: str = "Evaluate vulnerabilities in machine unlearning",
        task_spec: [Any] = None,
        max_iterations: int = 10,
        enable_debate: bool = True,
        output_dir: Path = Path("aust/outputs"),
        rag_storage_path: Path = Path("aust/rag_paper_db"),
        config_dir: Optional[Path] = None,
        # RAG configuration
        rag_top_k: int = 3,
        # Model configuration
        generator_model: Optional[str] = None,
        critic_model: Optional[str] = None,
        # Debate configuration
        quality_threshold: float = 0.85,
        max_debate_iterations: int = 3, # how many rounds of self-improving CoT
        # Query generator configuration
        query_generator_model: Optional[str] = None,
        query_max_queries: int = 3,
    ):
        """
        Initialize Inner Loop Orchestrator.

        Args:
            task_id: Unique task identifier (auto-generated if None)
            task_type: Type of task (data_based_unlearning or concept_erasure)
            task_description: High-level task description
            task_spec: Structured TaskSpec dictionary for the task (if available)
            max_iterations: Maximum number of inner loop iterations
            enable_debate: Whether to enable multi-agent debate
            output_dir: Base output directory for results
            rag_storage_path: Path to RAG vector database
            config_dir: Path to configuration directory
            rag_top_k: Default retrieval depth hint for the query generator
            generator_model: Optional override for hypothesis generation model
            critic_model: Optional override for hypothesis critique model
            quality_threshold: Quality threshold for debate termination
            max_debate_iterations: Max iterations in self-improving CoT
            query_generator_model: Optional override for query generator model
            query_max_queries: Maximum number of queries the agent should attempt per iteration
        """
        if task_spec is None:
            raise ValueError(
                "TaskSpec is required. Run the Task Parser (Step 1) before initializing the inner loop."
            )

        self.task_spec = self._normalize_task_spec(task_spec)
        self.task_id = task_id or f"task_{uuid.uuid4().hex[:8]}"
        self.task_type = task_type
        self.task_description = task_description
        self.rag_top_k = rag_top_k

        self.generator_model = (
            generator_model or HypothesisRefinementWorkforce.DEFAULT_GENERATOR_MODEL
        )
        self.critic_model = (
            critic_model or HypothesisRefinementWorkforce.DEFAULT_CRITIC_MODEL
        )

        logger.info(f"Initializing InnerLoopOrchestrator for task_id={self.task_id}")

        # Create output directory for this task
        self.output_dir = output_dir / self.task_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize state
        self.state = InnerLoopState(
            task_id=self.task_id,
            task_type=task_type,
            task_description=task_description,
            max_iterations=max_iterations,
            enable_debate=enable_debate,
            output_dir=self.output_dir,
            task_spec=self.task_spec,
        )

        self.debates_dir = self.output_dir / "debates"
        self.debates_dir.mkdir(exist_ok=True)

        self.traces_dir = self.output_dir / "attack_traces"
        self.traces_dir.mkdir(exist_ok=True)

        self.queries_dir = self.output_dir / "queries"
        self.queries_dir.mkdir(exist_ok=True)

        # Initialize attack trace file
        self.attack_trace_file = self.traces_dir / f"trace_{self.task_id}.md"
        self.state.attack_trace_file = self.attack_trace_file
        self._initialize_attack_trace()

        # Initialize RAG system
        logger.info(f"Loading RAG system from {rag_storage_path}")
        self.rag = PaperRAG(storage_path=str(rag_storage_path))
        logger.info("RAG system loaded successfully")

        # Initialize query generator agent
        query_model = query_generator_model
        query_results_top_k = max(self.rag_top_k, QueryGeneratorAgent.DEFAULT_TOP_K)
        logger.info(
            "Initializing Query Generator Agent with model=%s, max_queries=%s, top_k=%s",
            query_model or QueryGeneratorAgent.DEFAULT_MODEL,
            query_max_queries,
            query_results_top_k,
        )
        self.query_generator = QueryGeneratorAgent(
            rag=self.rag,
            model_name=query_model,
            max_queries=query_max_queries,
            top_k=query_results_top_k,
            output_dir=self.queries_dir,
        )

        # Initialize configuration loader
        self.config_loader = ConfigLoader(config_dir=config_dir)

        # Initialize hypothesis refinement workforce
        logger.info("Initializing Hypothesis Refinement Workforce")
        self.workforce = HypothesisRefinementWorkforce(
            generator_model=self.generator_model,
            critic_model=self.critic_model,
            quality_threshold=quality_threshold,
            max_iterations=max_debate_iterations,
        )
        logger.info("Hypothesis Refinement Workforce initialized")

        logger.info(f"InnerLoopOrchestrator initialized for task {self.task_id}")

    def _normalize_task_spec(self, task_spec: Any) -> dict[str, Any]:
        """
        Normalize TaskSpec input (TaskSpec object or dict) into a dictionary.
        """
        if task_spec is None:
            raise ValueError("TaskSpec cannot be None.")

        if isinstance(task_spec, TaskSpec):
            return task_spec.model_dump(mode="json")

        if hasattr(task_spec, "model_dump"):
            return task_spec.model_dump(mode="json")

        if isinstance(task_spec, dict):
            return dict(task_spec)

        if hasattr(task_spec, "dict"):
            return task_spec.dict()

        try:
            return dict(task_spec)
        except Exception as exc:
            raise ValueError(f"Unsupported TaskSpec type: {type(task_spec)}") from exc

    @classmethod
    def resume_from_state(
        cls,
        state_file: Path,
        rag_storage_path: Path = Path("aust/rag_paper_db"),
        config_dir: Optional[Path] = None,
        rag_top_k: int = 3,
        generator_model: Optional[str] = None,
        critic_model: Optional[str] = None,
        quality_threshold: float = 0.85,
        max_debate_iterations: int = 3,
        query_generator_model: Optional[str] = None,
        query_max_queries: int = 3,
    ) -> "InnerLoopOrchestrator":
        """
        Resume orchestrator from a saved state file.

        Args:
            state_file: Path to state.json file
            rag_storage_path: Path to RAG vector database
            config_dir: Path to configuration directory
            rag_top_k: Default retrieval depth hint for the query generator
            generator_model: Optional override for hypothesis generation model
            critic_model: Optional override for hypothesis critique model
            quality_threshold: Quality threshold for debate termination
            max_debate_iterations: Max iterations in self-improving CoT
            query_generator_model: Model for the query generator agent
            query_max_queries: Maximum queries per iteration for query generator

        Returns:
            InnerLoopOrchestrator instance with restored state

        Raises:
            FileNotFoundError: If state file doesn't exist
            ValueError: If state file is invalid
        """
        if not state_file.exists():
            raise FileNotFoundError(f"State file not found: {state_file}")

        logger.info(f"Loading state from {state_file}")

        # Load state
        loaded_state = InnerLoopState.load_from_file(state_file)

        logger.info(f"State loaded: task_id={loaded_state.task_id}, iteration={loaded_state.current_iteration}")

        # Create orchestrator with loaded state
        orchestrator = cls(
            task_id=loaded_state.task_id,
            task_type=loaded_state.task_type,
            task_description=loaded_state.task_description,
            task_spec=loaded_state.task_spec,
            max_iterations=loaded_state.max_iterations,
            enable_debate=loaded_state.enable_debate,
            output_dir=loaded_state.output_dir.parent,  # Parent since output_dir includes task_id
            rag_storage_path=rag_storage_path,
            config_dir=config_dir,
            rag_top_k=rag_top_k,
            generator_model=generator_model,
            critic_model=critic_model,
            quality_threshold=quality_threshold,
            max_debate_iterations=max_debate_iterations,
            query_generator_model=query_generator_model,
            query_max_queries=query_max_queries,
        )

        # Replace state with loaded state
        orchestrator.state = loaded_state

        logger.info(f"Orchestrator resumed from checkpoint (iteration {loaded_state.current_iteration})")

        return orchestrator

    def run(self) -> InnerLoopState:
        """
        Run the complete inner loop until exit condition is met.

        Returns:
            Final InnerLoopState

        Raises:
            RuntimeError: If loop execution fails
        """
        logger.info("=" * 80)
        logger.info(f"Starting Inner Loop for task {self.task_id}")
        logger.info("=" * 80)
        logger.info(f"Task type: {self.task_type}")
        logger.info(f"Task description: {self.task_description}")
        logger.info(f"Max iterations: {self.state.max_iterations}")
        logger.info(f"Debate enabled: {self.state.enable_debate}")
        logger.info("=" * 80)

        self._append_to_attack_trace("\n## Inner Loop Execution\n")
        self._append_to_attack_trace(
            f"**Started**: {datetime.now(timezone.utc).isoformat()}\n"
        )

        iteration_number = 0

        try:
            while True:
                iteration_number += 1

                # Check if should continue
                should_continue, reason = self.state.should_continue()
                if not should_continue:
                    logger.info(f"Loop termination: {reason}")
                    exit_condition = (
                        ExitCondition.VULNERABILITY_FOUND
                        if self.state.vulnerability_found
                        else ExitCondition.MAX_ITERATIONS
                    )
                    self.state.mark_complete(exit_condition, reason)
                    break

                # Run iteration
                logger.info(f"\n{'=' * 80}")
                logger.info(f"Iteration {iteration_number}/{self.state.max_iterations}")
                logger.info(f"{'=' * 80}")

                iteration_result = self._run_iteration(iteration_number)

                # Add result to state
                self.state.add_iteration_result(iteration_result)

                # Save state after each iteration
                self._save_state()

                logger.info(f"\nIteration {iteration_number} complete:")
                logger.info(f"  - Hypothesis: {iteration_result.hypothesis_summary}")
                logger.info(f"  - Duration: {iteration_result.duration_seconds:.1f}s")
                logger.info(
                    f"  - Vulnerability detected: {iteration_result.vulnerability_detected}"
                )

        except KeyboardInterrupt:
            logger.warning("User interrupted loop execution")
            self.state.mark_complete(ExitCondition.USER_STOPPED, "User stopped the loop")
            self._save_state()
        except Exception as e:
            logger.error(f"Inner loop failed: {e}", exc_info=True)
            self.state.mark_complete(ExitCondition.ERROR, f"Error: {str(e)}")
            self._save_state()
            raise

        # Final summary
        self._log_final_summary()
        self._append_final_summary_to_trace()
        self._save_state()

        logger.info(f"\nInner loop complete: {self.state.exit_condition.value}")
        logger.info(f"Total duration: {self.state.total_duration_seconds:.1f}s")
        logger.info(f"Output directory: {self.output_dir}")

        return self.state

    def _run_iteration(self, iteration_number: int) -> IterationResult:
        """
        Run a single iteration of the inner loop.

        Args:
            iteration_number: Current iteration number (1-indexed)

        Returns:
            IterationResult with all iteration data
        """
        iteration_started_at = datetime.now(timezone.utc)
        rag_queries: list[str] = []
        retrieved_paper_count = 0
        retrieved_paper_ids: list[str] = []
        experiment_results: Optional[dict] = None
        evaluator_feedback: Optional[str] = None
        vulnerability_detected = False
        vulnerability_confidence = 0.0

        # Append iteration header to attack trace
        self._append_to_attack_trace(f"\n### Iteration {iteration_number}\n\n")

        # Step 1: Build hypothesis generation context
        logger.info("Step 1: Building hypothesis generation context")
        context, rag_query_list = self._build_hypothesis_context(iteration_number)

        # Log RAG queries
        if rag_query_list:
            rag_queries.extend(rag_query_list)

        if context.retrieved_papers:
            retrieved_paper_count = len(context.retrieved_papers)
            retrieved_paper_ids = [
                paper.get('arxiv_id', 'unknown') for paper in context.retrieved_papers
            ]
            logger.info(
                f"  Retrieved {retrieved_paper_count} papers from RAG: "
                f"{retrieved_paper_ids}"
            )

        # Step 2: Generate and refine hypothesis
        logger.info("Step 2: Generating hypothesis with debate")
        hypothesis, debate_session = self.workforce.generate_refined_hypothesis(
            context=context,
            enable_debate=self.state.enable_debate and iteration_number > 1,
            debate_rounds=self.workforce.max_iterations,
            query_generator=(
                self.query_generator if iteration_number > 1 else None
            ),
        )

        rag_queries.extend(debate_session.rag_queries)
        debate_paper_ids = self._collect_retrieved_paper_ids(debate_session)
        if debate_paper_ids:
            existing_ids = set(retrieved_paper_ids)
            for paper_id in debate_paper_ids:
                if paper_id not in existing_ids:
                    retrieved_paper_ids.append(paper_id)
                    existing_ids.add(paper_id)
            retrieved_paper_count = len(existing_ids)

        # Save debate log
        debate_log_path = self.workforce.save_debate_log(debate_session, self.debates_dir)
        logger.info(f"  Debate log saved: {debate_log_path.name}")

        # Append hypothesis to attack trace
        self._append_hypothesis_to_trace(hypothesis, debate_session)

        # Step 3: Execute experiment (Story 1.6a - placeholder)
        logger.info("Step 3: Executing experiment (PLACEHOLDER)")
        experiment_results = self._execute_experiment_placeholder(hypothesis)
        experiment_executed = experiment_results is not None

        if experiment_results:
            self._append_to_attack_trace(
                f"**Experiment Result**: {experiment_results.get('summary', 'N/A')}\n\n"
            )

        # Step 4: Evaluate results (Story 1.7 - placeholder)
        logger.info("Step 4: Evaluating results (PLACEHOLDER)")
        evaluator_feedback, vulnerability_detected, vulnerability_confidence = (
            self._evaluate_results_placeholder(hypothesis, experiment_results)
        )

        self._append_to_attack_trace(
            f"**Evaluator Feedback**: {evaluator_feedback}\n"
            f"**Vulnerability Detected**: {vulnerability_detected} "
            f"(confidence={vulnerability_confidence:.2f})\n\n"
        )

        iteration_completed_at = datetime.now(timezone.utc)

        return IterationResult(
            iteration_number=iteration_number,
            hypothesis=hypothesis,
            debate_session=debate_session,
            rag_queries=rag_queries,
            retrieved_paper_count=retrieved_paper_count,
            retrieved_paper_ids=retrieved_paper_ids,
            experiment_executed=experiment_executed,
            experiment_results=experiment_results,
            evaluator_feedback=evaluator_feedback,
            vulnerability_detected=vulnerability_detected,
            vulnerability_confidence=vulnerability_confidence,
            started_at=iteration_started_at,
            completed_at=iteration_completed_at,
        )

    def _build_hypothesis_context(
        self, iteration_number: int
    ) -> tuple[HypothesisContext, list[str]]:
        """
        Build HypothesisContext with RAG-retrieved papers and past results.

        Args:
            iteration_number: Current iteration number

        Returns:
            Tuple of (HypothesisContext, rag_queries) for hypothesis generation
        """
        # Get past results summary
        past_results = self.state.get_past_results_summary(num_recent=3)

        # Get evaluator feedback from previous iteration
        evaluator_feedback = self.state.get_evaluator_feedback()

        # Get task template for iteration 1
        seed_template = None
        if iteration_number == 1:
            seed_template = self.config_loader.get_task_template(
                task_type=self.task_type,
                iteration=iteration_number,
            )
            if not seed_template:
                raise RuntimeError(
                    f"No task template available for task_type='{self.task_type}' "
                    "during the first iteration."
                )
            logger.info(
                f"  Using task template: {seed_template.get('id', 'unknown')}"
            )

        # Gather critic feedback summaries from previous iteration
        critic_feedback_summaries = self._collect_recent_critic_feedback()

        retrieved_papers: Optional[list[dict]] = None
        rag_queries: list[str] = []

        if iteration_number == 1:
            logger.info(
                "  Iteration 1: Skipping RAG query (using TaskSpec + task template only)"
            )
        else:
            logger.info(
                "  Iteration %s: Deferring new RAG queries until post-critic feedback",
                iteration_number,
            )
            previous_iteration = self.state.latest_iteration
            if (
                previous_iteration
                and previous_iteration.debate_session
                and previous_iteration.debate_session.retrieved_papers
            ):
                retrieved_papers = list(
                    previous_iteration.debate_session.retrieved_papers
                )
                logger.info(
                    "  Carrying forward %s previously retrieved papers",
                    len(retrieved_papers),
                )

        # Build context
        context = HypothesisContext(
            iteration_number=iteration_number,
            past_results=past_results if past_results else None,
            evaluator_feedback=evaluator_feedback,
            retrieved_papers=retrieved_papers if retrieved_papers else None,
            memory_entries=None,  # TODO: Integrate memory system
            seed_template=seed_template,
            task_spec=self.task_spec,
            critic_feedback_summaries=critic_feedback_summaries,
        )

        return context, rag_queries

    def _collect_recent_critic_feedback(self) -> Optional[list[dict[str, Any]]]:
        """
        Collect critic feedback summaries from the latest iteration for context building.
        """
        latest_iteration = self.state.latest_iteration
        if not latest_iteration or not latest_iteration.debate_session:
            return None

        summaries: list[dict[str, Any]] = []
        for exchange in latest_iteration.debate_session.exchanges:
            feedback = exchange.critic_feedback
            summaries.append(
                {
                    "round": exchange.round_number,
                    "overall_assessment": feedback.overall_assessment,
                    "overall_assumption": feedback.overall_assumption,
                    "novelty_score": feedback.novelty_score,
                    "feasibility_score": feedback.feasibility_score,
                    "rigor_score": feedback.rigor_score,
                    "strengths": feedback.strengths,
                    "weaknesses": feedback.weaknesses,
                    "suggestions": feedback.suggestions,
                }
            )

        return summaries or None

    @staticmethod
    def _collect_retrieved_paper_ids(session: DebateSession) -> list[str]:
        """Flatten retrieved paper identifiers from a debate session."""
        if not session or not session.exchanges:
            return []
        paper_ids: list[str] = []
        for exchange in session.exchanges:
            if not exchange.retrieval_context:
                continue
            for paper in exchange.retrieval_context:
                candidate = (
                    paper.get("arxiv_id")
                    or paper.get("paper_id")
                    or paper.get("id")
                    or "unknown"
                )
                paper_ids.append(str(candidate))
        return paper_ids

    def _execute_experiment_placeholder(self, hypothesis) -> Optional[dict]:
        """
        Placeholder for experiment execution (Story 1.6a).

        In the full implementation, this would:
        - Translate hypothesis to experiment configuration
        - Execute attack using appropriate toolkit
        - Collect metrics and results

        Args:
            hypothesis: Hypothesis to test

        Returns:
            Experiment results dictionary
        """
        logger.info(f"  [PLACEHOLDER] Would execute: {hypothesis.attack_type}")
        logger.info(f"  [PLACEHOLDER] Hypothesis description: {hypothesis.description}")

        # Simulate experiment result
        return {
            "summary": "Experiment not executed (placeholder)",
            "metrics": {},
            "success": False,
        }

    def _evaluate_results_placeholder(
        self, hypothesis, experiment_results
    ) -> tuple[str, bool, float]:
        """
        Placeholder for result evaluation (Story 1.7).

        In the full implementation, this would:
        - Analyze experiment metrics
        - Compare to expected outcomes
        - Generate structured feedback
        - Assess vulnerability confidence

        Args:
            hypothesis: Tested hypothesis
            experiment_results: Results from experiment execution

        Returns:
            Tuple of (evaluator_feedback, vulnerability_detected, confidence)
        """
        logger.info("  [PLACEHOLDER] Evaluating results...")

        # Simulate evaluation
        feedback = (
            f"Hypothesis {hypothesis.attack_type} tested. "
            f"Experiment placeholder - no real evaluation performed yet."
        )

        # Simulate no vulnerability for now
        vulnerability_detected = False
        confidence = 0.0

        return feedback, vulnerability_detected, confidence

    def _initialize_attack_trace(self):
        """Initialize attack trace markdown file."""
        content = f"""# Attack Trace: {self.task_id}

**Task Type**: {self.task_type}
**Description**: {self.task_description}
**Created**: {datetime.now(timezone.utc).isoformat()}

---

## Configuration

- **Max Iterations**: {self.state.max_iterations}
- **Debate Enabled**: {self.state.enable_debate}
- **Generator Model**: {self.generator_model}
- **Critic Model**: {self.critic_model}

---
"""
        self.attack_trace_file.write_text(content, encoding='utf-8')

    def _append_to_attack_trace(self, content: str):
        """Append content to attack trace file."""
        with open(self.attack_trace_file, 'a', encoding='utf-8') as f:
            f.write(content)

    def _append_hypothesis_to_trace(self, hypothesis, debate_session):
        """Append hypothesis details to attack trace."""
        confidence = (
            f"{float(hypothesis.confidence_score):.2f}"
            if hypothesis.confidence_score is not None
            else "N/A"
        )
        novelty = (
            f"{float(hypothesis.novelty_score):.2f}"
            if hypothesis.novelty_score is not None
            else "N/A"
        )

        duration = debate_session.duration_seconds
        duration_text = f"{duration:.1f}s" if duration is not None else "N/A"

        quality_score = debate_session.final_quality_score
        if quality_score is None and debate_session.exchanges:
            last_feedback = debate_session.exchanges[-1].critic_feedback
            if last_feedback:
                quality_score = last_feedback.average_score

        quality_text = f"{quality_score:.2f}" if quality_score is not None else "N/A"

        content = f"""**Hypothesis ID**: `{hypothesis.hypothesis_id}`
**Attack Type**: {hypothesis.attack_type}
**Description**: {hypothesis.description}
**Target Type**: {hypothesis.target_type}
**Confidence**: {confidence}
**Novelty**: {novelty}

**Debate Rounds**: {debate_session.total_rounds}
**Debate Duration**: {duration_text}
**Quality Score**: {quality_text}

"""
        self._append_to_attack_trace(content)

    def _append_final_summary_to_trace(self):
        """Append final summary to attack trace."""
        content = f"""
---

## Final Summary

**Total Iterations**: {len(self.state.iterations)}
**Exit Condition**: {self.state.exit_condition.value}
**Exit Message**: {self.state.exit_message}
**Total Duration**: {self.state.total_duration_seconds:.1f}s
**Vulnerability Found**: {self.state.vulnerability_found}
**Highest Confidence**: {self.state.highest_vulnerability_confidence:.2f}

**Completed**: {datetime.now(timezone.utc).isoformat()}

---
"""
        self._append_to_attack_trace(content)

    def _save_state(self):
        """Save current state to JSON file."""
        state_file = self.output_dir / "loop_state.json"
        self.state.save_to_file(state_file)
        logger.debug(f"State saved to {state_file}")

    def _log_final_summary(self):
        """Log final summary of inner loop execution."""
        logger.info("\n" + "=" * 80)
        logger.info("INNER LOOP FINAL SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Task ID: {self.task_id}")
        logger.info(f"Total Iterations: {len(self.state.iterations)}")
        logger.info(f"Exit Condition: {self.state.exit_condition.value}")
        logger.info(f"Vulnerability Found: {self.state.vulnerability_found}")
        logger.info(
            f"Highest Confidence: {self.state.highest_vulnerability_confidence:.2f}"
        )
        logger.info(f"Total Duration: {self.state.total_duration_seconds:.1f}s")
        logger.info(f"Output Directory: {self.output_dir}")
        logger.info(f"Attack Trace: {self.attack_trace_file}")
        logger.info("=" * 80)
