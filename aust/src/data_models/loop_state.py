"""
Inner loop state data models.

Encapsulates iteration bookkeeping, exit conditions, and serialization utilities
used by the orchestrator to persist and resume research progress.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from aust.src.data_models.debate import DebateSession
from aust.src.data_models.hypothesis import Hypothesis


class ExitCondition(str, Enum):
    """Exit conditions for inner loop termination."""

    VULNERABILITY_FOUND = "vulnerability_found"  # High-confidence vulnerability detected
    INCONCLUSIVE = "inconclusive"  # Max iterations reached without clear result
    MAX_ITERATIONS = "max_iterations"  # Reached maximum iteration limit
    ERROR = "error"  # Unrecoverable error occurred
    USER_STOPPED = "user_stopped"  # User manually stopped the loop


class IterationResult(BaseModel):
    """Results from a single iteration of the inner loop."""

    iteration_number: int = Field(..., ge=1)
    hypothesis: Hypothesis = Field(..., description="Final hypothesis from debate")
    debate_session: DebateSession = Field(..., description="Complete debate session")

    # RAG retrieval info
    rag_queries: list[str] = Field(default_factory=list)
    retrieved_paper_count: int = Field(default=0)
    retrieved_paper_ids: list[str] = Field(default_factory=list)

    # Experiment execution (Story 1.6a - placeholder)
    experiment_executed: bool = Field(default=False)
    experiment_results: Optional[dict] = Field(default=None)

    # Evaluation (Story 1.7)
    evaluator_feedback: Optional[str] = Field(default=None)
    vulnerability_detected: bool = Field(default=False)
    vulnerability_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    # Timing
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = Field(default=None)

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate iteration duration in seconds."""
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def hypothesis_summary(self) -> str:
        """One-line summary of hypothesis for logging."""
        summary = (self.hypothesis.description or "").strip()
        if summary.endswith("...") and self.hypothesis.description:
            summary = self.hypothesis.description.strip()
        if not summary:
            summary = "No hypothesis summary available"
        return f"{self.hypothesis.attack_type}: {summary}"

    @property
    def outcome(self) -> str:
        """Outcome description for this iteration."""
        if not self.experiment_executed:
            return "Hypothesis generated (experiment not executed)"
        if self.vulnerability_detected:
            return f"Vulnerability detected (confidence={self.vulnerability_confidence:.2f})"
        return "No vulnerability detected"

    @property
    def key_learning(self) -> str:
        """Key learning from this iteration for next iteration."""
        if self.evaluator_feedback:
            feedback = self.evaluator_feedback.strip()
            return feedback or "Key learning unavailable"
        if self.debate_session and self.debate_session.exchanges:
            last_feedback = self.debate_session.exchanges[-1].critic_feedback
            if last_feedback.suggestions:
                suggestion = last_feedback.suggestions[0].strip()
                if suggestion:
                    return suggestion
        if self.vulnerability_detected:
            return f"Attack {self.hypothesis.attack_type} succeeded"
        return f"Attack {self.hypothesis.attack_type} did not reveal vulnerability"


class InnerLoopState(BaseModel):
    """
    Complete state of the inner loop research cycle.

    Tracks all iterations, results, and exit conditions.
    """

    # Task identification
    task_id: str = Field(..., description="Unique task identifier")
    task_type: str = Field(
        ..., description="Task type: data_based_unlearning or concept_erasure"
    )
    task_description: str = Field(..., description="High-level task description")

    # Loop configuration
    max_iterations: int = Field(default=10, ge=1, le=50)
    current_iteration: int = Field(default=0, ge=0)
    enable_debate: bool = Field(
        default=True, description="Enable multi-agent debate for hypothesis refinement"
    )

    # State tracking
    iterations: list[IterationResult] = Field(
        default_factory=list, description="Results from all iterations"
    )
    exit_condition: Optional[ExitCondition] = Field(default=None)
    exit_message: Optional[str] = Field(default=None)

    # Timestamps
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = Field(default=None)

    # Output paths
    output_dir: Path = Field(default=Path("aust/outputs"))
    attack_trace_file: Optional[Path] = Field(default=None)

    # Model inputs (for main.py entry point)
    base_model_path: Optional[str] = Field(
        default=None, description="Path to base model (before unlearning)"
    )
    unlearned_model_path: Optional[str] = Field(
        default=None, description="Path to unlearned model (after unlearning/erasure)"
    )
    test_prompt: Optional[str] = Field(
        default=None, description="Test prompt containing keywords/concepts to evaluate"
    )
    task_spec: Optional[dict[str, Any]] = Field(
        default=None,
        description="Structured TaskSpec dictionary associated with this task",
    )

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    @property
    def is_complete(self) -> bool:
        """Check if loop has completed."""
        return self.exit_condition is not None

    @property
    def total_duration_seconds(self) -> Optional[float]:
        """Total duration of loop in seconds."""
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def latest_iteration(self) -> Optional[IterationResult]:
        """Get most recent iteration result."""
        return self.iterations[-1] if self.iterations else None

    @property
    def vulnerability_found(self) -> bool:
        """Check if any iteration found a vulnerability."""
        return any(iter_result.vulnerability_detected for iter_result in self.iterations)

    @property
    def highest_vulnerability_confidence(self) -> float:
        """Get highest vulnerability confidence across all iterations."""
        if not self.iterations:
            return 0.0
        return max(
            iter_result.vulnerability_confidence for iter_result in self.iterations
        )

    def get_past_results_summary(self, num_recent: int = 3) -> list[dict]:
        """
        Get summary of past N iterations for hypothesis context.

        Args:
            num_recent: Number of recent iterations to include

        Returns:
            List of dictionaries with iteration summaries
        """
        recent_iterations = self.iterations[-num_recent:] if self.iterations else []

        summaries: list[dict[str, Any]] = []
        for iter_result in recent_iterations:
            description = (iter_result.hypothesis.description or "").strip()
            critic_assumption: Optional[str] = None
            if iter_result.debate_session and iter_result.debate_session.exchanges:
                latest_feedback = iter_result.debate_session.exchanges[-1].critic_feedback
                if latest_feedback:
                    critic_assumption = (
                        latest_feedback.overall_assumption
                        or latest_feedback.overall_assessment
                    )

            summaries.append(
                {
                    "iteration": iter_result.iteration_number,
                    "description": description,
                    "overall_assumption": critic_assumption,
                    "outcome": iter_result.outcome,
                    "key_learning": iter_result.key_learning,
                    "vulnerability_confidence": iter_result.vulnerability_confidence,
                }
            )

        return summaries

    def get_evaluator_feedback(self) -> Optional[str]:
        """Get evaluator feedback from most recent iteration."""
        if not self.iterations:
            return None
        latest = self.iterations[-1]
        return latest.evaluator_feedback

    def should_continue(self) -> tuple[bool, Optional[str]]:
        """
        Check if loop should continue.

        Returns:
            Tuple of (should_continue, reason)
        """
        # Already completed
        if self.is_complete:
            return False, f"Loop already completed: {self.exit_condition.value}"

        # Max iterations reached
        if self.current_iteration >= self.max_iterations:
            return (
                False,
                f"Maximum iterations reached ({self.max_iterations})",
            )

        # High-confidence vulnerability found
        if self.highest_vulnerability_confidence >= 0.9:
            return (
                False,
                f"High-confidence vulnerability found (confidence={self.highest_vulnerability_confidence:.2f})",
            )

        return True, None

    def add_iteration_result(self, result: IterationResult):
        """
        Add iteration result and update state.

        Args:
            result: Iteration result to add
        """
        self.iterations.append(result)
        self.current_iteration = result.iteration_number

    def mark_complete(self, exit_condition: ExitCondition, exit_message: str):
        """
        Mark loop as complete with exit condition.

        Args:
            exit_condition: Reason for exit
            exit_message: Human-readable exit message
        """
        self.exit_condition = exit_condition
        self.exit_message = exit_message
        self.completed_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump(mode="json")

    def save_to_file(self, file_path: Path):
        """
        Save state to JSON file with atomic write.

        Args:
            file_path: Path to save state file
        """
        import json

        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write
        temp_path = file_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False, default=str)

        temp_path.replace(file_path)

    @classmethod
    def load_from_file(cls, file_path: Path) -> "InnerLoopState":
        """
        Load state from JSON file.

        Args:
            file_path: Path to state file

        Returns:
            Loaded InnerLoopState instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is invalid
        """
        import json

        if not file_path.exists():
            raise FileNotFoundError(f"State file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        return cls(**data)


__all__ = ["ExitCondition", "IterationResult", "InnerLoopState"]

