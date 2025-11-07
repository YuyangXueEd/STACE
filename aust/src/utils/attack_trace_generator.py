"""
Attack Trace Generator for AUST (Story 4.3).

Generates dual-format attack traces:
1. JSON format - Machine-readable for programmatic access
2. Markdown format - Human-readable with narratives for academic reports

The traces include:
- Iteration details with hypothesis description
- Experiment design justification
- Quantitative and qualitative results
- Hypothesis evolution narratives
- Failure analysis
- Critic feedback integration
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from aust.src.utils.logging_config import get_logger

if TYPE_CHECKING:
    from aust.src.data_models.debate import DebateSession
    from aust.src.data_models.hypothesis import Hypothesis
    from aust.src.data_models.loop_state import InnerLoopState, IterationResult

logger = get_logger(__name__)


class AttackTraceGenerator:
    """
    Generates comprehensive attack traces in dual formats (JSON + Markdown).

    This class provides enhanced tracing capabilities for the inner loop,
    generating detailed documentation suitable for:
    - Academic report integration (Markdown)
    - Programmatic analysis and parsing (JSON)
    - Hypothesis evolution tracking
    - Failure analysis and learning
    """

    def __init__(self, output_dir: Path, task_id: str):
        """
        Initialize attack trace generator.

        Args:
            output_dir: Directory to save trace files
            task_id: Unique task identifier
        """
        self.output_dir = output_dir
        self.task_id = task_id
        self.traces_dir = output_dir / "attack_traces"
        self.traces_dir.mkdir(parents=True, exist_ok=True)

        # Define output paths
        self.json_trace_file = self.traces_dir / f"trace_{task_id}.json"
        self.md_trace_file = self.traces_dir / f"trace_{task_id}.md"

        logger.info(
            f"AttackTraceGenerator initialized for task {task_id} (output: {self.traces_dir})"
        )

    def initialize_trace(
        self,
        task_type: str,
        task_description: str,
        task_spec: Optional[dict[str, Any]],
        max_iterations: int,
        enable_debate: bool,
        generator_model: str,
        critic_model: str,
    ) -> None:
        """
        Initialize trace files with header information.

        Args:
            task_type: Type of task (concept_erasure, data_based_unlearning)
            task_description: High-level task description
            task_spec: TaskSpec dictionary with model paths and parameters
            max_iterations: Maximum number of iterations
            enable_debate: Whether debate is enabled
            generator_model: Hypothesis generator model identifier
            critic_model: Critic model identifier
        """
        timestamp = datetime.now(timezone.utc)

        # Initialize JSON trace
        json_data = {
            "task_id": self.task_id,
            "task_type": task_type,
            "task_description": task_description,
            "task_spec": task_spec or {},
            "configuration": {
                "max_iterations": max_iterations,
                "enable_debate": enable_debate,
                "generator_model": generator_model,
                "critic_model": critic_model,
            },
            "created_at": timestamp.isoformat(),
            "iterations": [],
            "summary": None,
        }
        self._write_json(json_data)

        # Initialize Markdown trace
        md_content = self._generate_md_header(
            task_type,
            task_description,
            task_spec,
            max_iterations,
            enable_debate,
            generator_model,
            critic_model,
            timestamp,
        )
        self.md_trace_file.write_text(md_content, encoding="utf-8")

        logger.info(f"Attack trace initialized: {self.md_trace_file}, {self.json_trace_file}")

    def append_iteration(
        self,
        iteration_result: IterationResult,
        iteration_number: int,
    ) -> None:
        """
        Append iteration details to both JSON and Markdown traces.

        Args:
            iteration_result: Complete iteration result
            iteration_number: Current iteration number
        """
        logger.debug(f"Appending iteration {iteration_number} to attack trace")

        # Append to JSON
        self._append_iteration_json(iteration_result, iteration_number)

        # Append to Markdown
        self._append_iteration_md(iteration_result, iteration_number)

        logger.debug(f"Iteration {iteration_number} appended to attack trace")

    def save_iteration_trace(
        self,
        iteration_result: IterationResult,
        iteration_number: int,
        task_type: str,
        task_description: Optional[str] = None,
        task_spec: Optional[dict[str, Any]] = None,
    ) -> Path:
        """
        Save per-iteration attack trace to individual JSON file (Story 5.2 AC7).

        This method generates standalone traces for each iteration, enabling
        the Reporter to incrementally load and analyze iteration results
        without waiting for the full task to complete.

        Args:
            iteration_result: Complete iteration result
            iteration_number: Current iteration number (1-indexed)
            task_type: Task type (concept_erasure, data_based_unlearning)
            task_description: Optional task description for context
            task_spec: Optional task specification dictionary for context

        Returns:
            Path to the saved iteration trace file

        Example filename:
            attack_trace_iter_01.json, attack_trace_iter_02.json, etc.
        """
        # Build per-iteration trace file path
        filename = f"attack_trace_iter_{iteration_number:02d}.json"
        iteration_trace_file = self.traces_dir / filename

        # Build iteration trace structure matching AC7 schema
        # Include task-level context for standalone trace files
        iteration_trace = {
            "task_id": self.task_id,
            "task_type": task_type,
            "task_description": task_description,
            "task_spec": task_spec or {},
            "iteration": {
                "iteration_number": iteration_number,
                "started_at": iteration_result.started_at.isoformat(),
                "completed_at": (
                    iteration_result.completed_at.isoformat()
                    if iteration_result.completed_at
                    else None
                ),
                "hypothesis": self._serialize_hypothesis(iteration_result.hypothesis),
                "attempts": self._extract_attempts_from_iteration(iteration_result),
                "final_status": "success" if iteration_result.vulnerability_detected else "failure",
                "vulnerability_detected": iteration_result.vulnerability_detected,
                "confidence": iteration_result.vulnerability_confidence,
                "debate_session": (
                    self._serialize_debate_session(iteration_result.debate_session)
                    if iteration_result.debate_session
                    else None
                ),
                "debate_narrative": (
                    self._generate_debate_narrative(iteration_result.debate_session).strip()
                    if iteration_result.debate_session
                    else None
                ),
                "key_learning": iteration_result.key_learning,
                "outcome_summary": iteration_result.outcome,
            },
        }

        # Write to file with atomic write pattern
        temp_file = iteration_trace_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(iteration_trace, f, indent=2, ensure_ascii=False)
        temp_file.replace(iteration_trace_file)

        logger.info(f"Saved iteration {iteration_number} trace: {iteration_trace_file}")
        return iteration_trace_file

    def _extract_attempts_from_iteration(
        self, iteration_result: IterationResult
    ) -> list[dict[str, Any]]:
        """
        Extract attempt records from an iteration result.

        For now, each iteration maps to a single attempt. In future stories,
        code repair loops may generate multiple attempts per iteration.

        Args:
            iteration_result: Iteration result to extract attempts from

        Returns:
            List of attempt records
        """
        attempt = {
            "attempt_number": 1,
            "status": "success" if iteration_result.vulnerability_detected else "failure",
            "images_generated": self._count_images_from_results(iteration_result),
            "evaluator_feedback": iteration_result.evaluator_feedback or "No feedback",
        }
        return [attempt]

    def _count_images_from_results(self, iteration_result: IterationResult) -> int:
        """
        Count number of images generated from experiment results.

        Args:
            iteration_result: Iteration result with experiment_results

        Returns:
            Number of images generated (0 if not available)
        """
        if not iteration_result.experiment_results:
            return 0

        results = iteration_result.experiment_results
        # Check for image_paths or images_count field
        if "image_paths" in results:
            return len(results["image_paths"])
        if "images_count" in results:
            return results["images_count"]
        if "images_generated" in results:
            return results["images_generated"]

        return 0

    def finalize_trace(
        self,
        final_state: InnerLoopState,
    ) -> tuple[Path, Path]:
        """
        Finalize both trace files with summary information.

        Args:
            final_state: Final inner loop state

        Returns:
            Tuple of (json_path, markdown_path)
        """
        logger.info("Finalizing attack trace")

        # Finalize JSON
        self._finalize_json(final_state)

        # Finalize Markdown
        self._finalize_md(final_state)

        logger.info(
            f"Attack trace finalized: JSON={self.json_trace_file}, MD={self.md_trace_file}"
        )

        return self.json_trace_file, self.md_trace_file

    # =========================================================================
    # JSON Trace Generation
    # =========================================================================

    def _write_json(self, data: dict[str, Any]) -> None:
        """Write JSON data to trace file."""
        self.json_trace_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _append_iteration_json(
        self,
        iteration_result: IterationResult,
        iteration_number: int,
    ) -> None:
        """Append iteration to JSON trace."""
        # Load current JSON
        json_data = json.loads(self.json_trace_file.read_text(encoding="utf-8"))

        # Build iteration entry
        iteration_entry = {
            "iteration_number": iteration_number,
            "hypothesis": self._serialize_hypothesis(iteration_result.hypothesis),
            "debate_session": self._serialize_debate_session(iteration_result.debate_session),
            "rag_info": {
                "queries": iteration_result.rag_queries,
                "retrieved_paper_count": iteration_result.retrieved_paper_count,
                "retrieved_paper_ids": iteration_result.retrieved_paper_ids,
            },
            "experiment": {
                "executed": iteration_result.experiment_executed,
                "results": iteration_result.experiment_results,
            },
            "evaluation": {
                "vulnerability_detected": iteration_result.vulnerability_detected,
                "vulnerability_confidence": iteration_result.vulnerability_confidence,
                "evaluator_feedback": iteration_result.evaluator_feedback,
            },
            "timing": {
                "started_at": iteration_result.started_at.isoformat(),
                "completed_at": (
                    iteration_result.completed_at.isoformat()
                    if iteration_result.completed_at
                    else None
                ),
                "duration_seconds": iteration_result.duration_seconds,
            },
            "outcome_summary": iteration_result.outcome,
            "key_learning": iteration_result.key_learning,
        }

        json_data["iterations"].append(iteration_entry)
        self._write_json(json_data)

    def _finalize_json(self, final_state: InnerLoopState) -> None:
        """Add final summary to JSON trace."""
        json_data = json.loads(self.json_trace_file.read_text(encoding="utf-8"))

        json_data["summary"] = {
            "total_iterations": len(final_state.iterations),
            "exit_condition": final_state.exit_condition.value if final_state.exit_condition else None,
            "exit_message": final_state.exit_message,
            "vulnerability_found": final_state.vulnerability_found,
            "highest_vulnerability_confidence": final_state.highest_vulnerability_confidence,
            "total_duration_seconds": final_state.total_duration_seconds,
            "completed_at": (
                final_state.completed_at.isoformat() if final_state.completed_at else None
            ),
        }

        self._write_json(json_data)

    # =========================================================================
    # Markdown Trace Generation
    # =========================================================================

    def _generate_md_header(
        self,
        task_type: str,
        task_description: str,
        task_spec: Optional[dict[str, Any]],
        max_iterations: int,
        enable_debate: bool,
        generator_model: str,
        critic_model: str,
        timestamp: datetime,
    ) -> str:
        """Generate Markdown header section."""
        spec_info = ""
        if task_spec:
            model_name = task_spec.get("model_name", "Unknown")
            model_version = task_spec.get("model_version", "Unknown")
            unlearning_method = task_spec.get("unlearning_method", "Unknown")
            unlearned_target = task_spec.get("unlearned_target", "Unknown")

            spec_info = f"""
### Task Specification

- **Target Model**: {model_name} (version {model_version})
- **Unlearning Method**: {unlearning_method}
- **Unlearned Target**: {unlearned_target}
"""

        content = f"""# Attack Trace: {self.task_id}

**Task Type**: {task_type}
**Description**: {task_description}
**Created**: {timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")}
{spec_info}
---

## Configuration

- **Max Iterations**: {max_iterations}
- **Debate Enabled**: {enable_debate}
- **Generator Model**: {generator_model}
- **Critic Model**: {critic_model}

---

## Iteration History

This section documents the complete evolution of hypotheses through the inner loop,
capturing design intent, debate feedback, experiment results, and key learnings.

"""
        return content

    def _append_iteration_md(
        self,
        iteration_result: IterationResult,
        iteration_number: int,
    ) -> None:
        """Append iteration to Markdown trace with narrative."""
        hypothesis = iteration_result.hypothesis
        debate = iteration_result.debate_session

        # Build iteration narrative
        content = f"""
### Iteration {iteration_number}

**Hypothesis ID**: `{hypothesis.hypothesis_id}`
**Attack Type**: {hypothesis.attack_type}
**Target Type**: {hypothesis.target_type}
**Started**: {iteration_result.started_at.strftime("%Y-%m-%d %H:%M:%S UTC")}

#### Hypothesis Description

{hypothesis.description or "No description provided"}

#### Experiment Design

{hypothesis.experiment_design or "No experiment design specified"}

"""

        # Add debate evolution narrative
        if debate and debate.exchanges:
            content += self._generate_debate_narrative(debate)

        # Add RAG context
        if iteration_result.rag_queries:
            content += self._generate_rag_narrative(iteration_result)

        # Add experiment results
        if iteration_result.experiment_executed:
            content += self._generate_experiment_narrative(iteration_result)
        else:
            content += """#### Experiment Execution

**Status**: Not executed

"""

        # Add evaluation results
        content += self._generate_evaluation_narrative(iteration_result)

        # Add key learning
        content += f"""#### Key Learning

{iteration_result.key_learning}

#### Outcome Summary

{iteration_result.outcome}

---

"""

        # Append to file
        with open(self.md_trace_file, "a", encoding="utf-8") as f:
            f.write(content)

    def _finalize_md(self, final_state: InnerLoopState) -> None:
        """Add final summary section to Markdown trace."""
        # Generate hypothesis evolution analysis
        evolution_narrative = self._generate_evolution_narrative(final_state)

        # Generate failure analysis
        failure_narrative = self._generate_failure_narrative(final_state)

        content = f"""
---

## Final Summary

**Total Iterations**: {len(final_state.iterations)}
**Exit Condition**: {final_state.exit_condition.value if final_state.exit_condition else "Unknown"}
**Exit Message**: {final_state.exit_message or "No message"}
**Total Duration**: {final_state.total_duration_seconds:.1f} seconds
**Vulnerability Found**: {"Yes" if final_state.vulnerability_found else "No"}
**Highest Confidence**: {final_state.highest_vulnerability_confidence:.2%}
**Completed**: {final_state.completed_at.strftime("%Y-%m-%d %H:%M:%S UTC") if final_state.completed_at else "Not completed"}

### Hypothesis Evolution Analysis

{evolution_narrative}

### Failure Analysis

{failure_narrative}

---

**End of Attack Trace**
"""

        with open(self.md_trace_file, "a", encoding="utf-8") as f:
            f.write(content)

    # =========================================================================
    # Narrative Generation Helpers
    # =========================================================================

    def _generate_debate_narrative(self, debate: DebateSession) -> str:
        """Generate narrative for debate session."""
        quality_score = debate.final_quality_score
        if quality_score is None and debate.exchanges:
            last_feedback = debate.exchanges[-1].critic_feedback
            if last_feedback:
                quality_score = last_feedback.average_score

        quality_text = f"{quality_score:.2f}" if quality_score is not None else "N/A"
        duration_text = f"{debate.duration_seconds:.1f}s" if debate.duration_seconds else "N/A"
        outcome_flags: list[str] = []
        if debate.convergence_reached:
            outcome_flags.append("converged")
        if debate.quality_threshold_met:
            outcome_flags.append("quality threshold met")
        outcome_text = ", ".join(outcome_flags) if outcome_flags else "No special status flags"

        narrative = f"""#### Debate Refinement

**Rounds**: {debate.total_rounds}
**Duration**: {duration_text}
**Final Quality Score**: {quality_text}
**Outcome Flags**: {outcome_text}

"""

        # Add exchange details
        for idx, exchange in enumerate(debate.exchanges, 1):
            critic_feedback = exchange.critic_feedback
            if not critic_feedback:
                continue

            score_text = f"{critic_feedback.average_score:.2f}" if critic_feedback.average_score is not None else "N/A"

            narrative += f"""**Round {idx} - Critic Feedback**

- **Overall Assessment**: {critic_feedback.overall_assessment or "No assessment"}
- **Quality Score**: {score_text}

**Strengths**:
{self._format_list(critic_feedback.strengths)}

**Weaknesses**:
{self._format_list(critic_feedback.weaknesses)}

**Suggestions**:
{self._format_list(critic_feedback.suggestions)}

"""

        return narrative

    def _generate_rag_narrative(self, iteration_result: IterationResult) -> str:
        """Generate narrative for RAG retrieval."""
        narrative = f"""#### Literature Context (RAG)

**Queries Generated**: {len(iteration_result.rag_queries)}
**Papers Retrieved**: {iteration_result.retrieved_paper_count}

"""

        if iteration_result.rag_queries:
            narrative += "**Queries**:\n"
            for idx, query in enumerate(iteration_result.rag_queries, 1):
                narrative += f"{idx}. {query}\n"
            narrative += "\n"

        if iteration_result.retrieved_paper_ids:
            narrative += f"**Retrieved Paper IDs**: {', '.join(iteration_result.retrieved_paper_ids[:5])}"
            if len(iteration_result.retrieved_paper_ids) > 5:
                narrative += f" (and {len(iteration_result.retrieved_paper_ids) - 5} more)"
            narrative += "\n\n"

        return narrative

    def _generate_experiment_narrative(self, iteration_result: IterationResult) -> str:
        """Generate narrative for experiment execution."""
        results = iteration_result.experiment_results or {}

        narrative = f"""#### Experiment Execution

**Status**: Executed
**Duration**: {results.get('execution_time_seconds', 'N/A')} seconds

"""

        # Add quantitative results
        if "metrics" in results:
            narrative += "**Quantitative Results**:\n"
            for metric_name, metric_value in results["metrics"].items():
                narrative += f"- {metric_name}: {metric_value}\n"
            narrative += "\n"

        # Add qualitative observations
        if "observations" in results:
            narrative += f"""**Qualitative Observations**:

{results['observations']}

"""

        # Add error information if present
        if results.get("error"):
            narrative += f"""**Error Encountered**:

```
{results['error']}
```

"""

        return narrative

    def _generate_evaluation_narrative(self, iteration_result: IterationResult) -> str:
        """Generate narrative for evaluation results."""
        narrative = f"""#### Evaluation Results

**Vulnerability Detected**: {"Yes" if iteration_result.vulnerability_detected else "No"}
**Confidence Score**: {iteration_result.vulnerability_confidence:.2%}

"""

        if iteration_result.evaluator_feedback:
            narrative += f"""**Evaluator Feedback**:

{iteration_result.evaluator_feedback}

"""

        return narrative

    def _generate_evolution_narrative(self, final_state: InnerLoopState) -> str:
        """Generate narrative analyzing hypothesis evolution across iterations."""
        if not final_state.iterations:
            return "No iterations completed."

        narrative = "This section analyzes how hypotheses evolved through critic feedback and experimentation.\n\n"

        # Track attack type evolution
        attack_types = [iter_res.hypothesis.attack_type for iter_res in final_state.iterations]
        unique_types = list(dict.fromkeys(attack_types))  # Preserve order

        narrative += f"**Attack Type Progression**: {' → '.join(unique_types)}\n\n"

        # Identify confidence trend
        confidences = [
            iter_res.vulnerability_confidence
            for iter_res in final_state.iterations
            if iter_res.vulnerability_detected
        ]

        if confidences:
            narrative += f"**Vulnerability Confidence Trend**: "
            narrative += " → ".join(f"{conf:.2%}" for conf in confidences)
            narrative += f" (Peak: {max(confidences):.2%})\n\n"

        # Summarize key improvements
        narrative += "**Key Improvements Across Iterations**:\n\n"
        for idx, iter_res in enumerate(final_state.iterations, 1):
            if iter_res.debate_session and iter_res.debate_session.exchanges:
                last_feedback = iter_res.debate_session.exchanges[-1].critic_feedback
                if last_feedback and last_feedback.suggestions:
                    top_suggestion = last_feedback.suggestions[0] if last_feedback.suggestions else "N/A"
                    narrative += f"{idx}. {top_suggestion}\n"

        return narrative

    def _generate_failure_narrative(self, final_state: InnerLoopState) -> str:
        """Generate narrative analyzing why certain hypotheses failed."""
        failed_iterations = [
            iter_res
            for iter_res in final_state.iterations
            if not iter_res.vulnerability_detected
        ]

        if not failed_iterations:
            return "All iterations resulted in detected vulnerabilities."

        narrative = "This section analyzes iterations that did not lead to vulnerability detection.\n\n"

        narrative += f"**Failed Iterations**: {len(failed_iterations)} / {len(final_state.iterations)}\n\n"

        narrative += "**Common Failure Patterns**:\n\n"

        # Analyze failure reasons from evaluator feedback
        for idx, iter_res in enumerate(failed_iterations[:3], 1):  # Limit to first 3
            iteration_num = iter_res.iteration_number
            hypothesis_type = iter_res.hypothesis.attack_type

            narrative += f"{idx}. **Iteration {iteration_num}** ({hypothesis_type}): "

            if iter_res.evaluator_feedback:
                # Extract key reason from feedback
                feedback_preview = iter_res.evaluator_feedback[:200]
                if len(iter_res.evaluator_feedback) > 200:
                    feedback_preview += "..."
                narrative += f"{feedback_preview}\n\n"
            elif not iter_res.experiment_executed:
                narrative += "Experiment was not executed.\n\n"
            else:
                narrative += "Experiment executed but no vulnerability detected.\n\n"

        if len(failed_iterations) > 3:
            narrative += f"\n(Analysis limited to first 3 failures; {len(failed_iterations) - 3} additional failures not detailed)\n"

        return narrative

    # =========================================================================
    # Serialization Helpers
    # =========================================================================

    def _serialize_hypothesis(self, hypothesis: Hypothesis) -> dict[str, Any]:
        """Serialize hypothesis to dictionary."""
        return hypothesis.model_dump(mode="json")

    def _serialize_debate_session(self, debate: DebateSession) -> dict[str, Any]:
        """Serialize debate session to dictionary."""
        return debate.model_dump(mode="json")

    def _format_list(self, items: Optional[list[str]]) -> str:
        """Format list of strings as markdown bullet points."""
        if not items:
            return "- None\n"
        return "\n".join(f"- {item}" for item in items) + "\n"
