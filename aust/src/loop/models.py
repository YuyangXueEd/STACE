"""
Data models for AUST loop orchestration and hypothesis refinement workflow.

This module defines Pydantic models for:
- Task specification assembly from LLM parser outputs
- Loop state management
- Hypothesis generation and refinement
- Debate workflow tracking
- Iteration results and evaluation feedback
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError, field_validator

if TYPE_CHECKING:  # pragma: no cover - only for typing
    from aust.src.agents.task_parser_agent import TaskParserResult


class TaskSpec(BaseModel):
    """
    Structured task specification derived from CLI inputs and LLM parser output.

    Fields capture the essential model details, unlearned concept, and
    optional method required by downstream agents.
    """

    task_type: str = Field(
        ..., description="Task type (data_based_unlearning or concept_erasure)"
    )
    model_name: str = Field(
        ..., description="Normalized model name (e.g., Stable Diffusion)"
    )
    model_version: Optional[str] = Field(
        default=None, description="Model version if provided (e.g., 1.4)"
    )
    base_model_path: str = Field(
        ..., description="Path to base/original model"
    )
    unlearned_model_path: str = Field(
        ..., description="Path to unlearned/erased model"
    )
    unlearned_target: str = Field(
        ..., description="Primary concept, dataset, or capability that was unlearned"
    )
    unlearning_method: Optional[str] = Field(
        default=None, description="Method used for unlearning (optional)"
    )
    user_prompt: str = Field(
        ..., description="Raw user prompt provided at the CLI"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp (UTC)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata parsed from the prompt",
    )

    @field_validator("created_at", mode="before")
    @classmethod
    def ensure_timezone(cls, value: datetime) -> datetime:
        """Ensure created_at timestamps are timezone-aware."""
        if value is None:
            return datetime.now(timezone.utc)
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    def save_json(self, path: Path) -> Path:
        """Persist TaskSpec to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def assemble(
        cls,
        *,
        task_type: str,
        base_model_path: str,
        unlearned_model_path: str,
        user_prompt: str,
        parser_result: Optional["TaskParserResult"],
        overrides: Optional[Mapping[str, Optional[str]]] = None,
    ) -> "TaskSpec":
        """
        Combine CLI overrides and task parser output into a validated TaskSpec.

        Args:
            task_type: Task type (concept_erasure or data_based_unlearning).
            base_model_path: Path to the base/original model.
            unlearned_model_path: Path to the unlearned model.
            user_prompt: Raw prompt provided by the user.
            parser_result: Structured output from TaskParserAgent (may be None).
            overrides: Explicit CLI overrides for individual fields.

        Returns:
            A validated TaskSpec object.

        Raises:
            ValueError: If required fields cannot be determined.
        """
        prompt_text = (user_prompt or "").strip()
        if not prompt_text:
            raise ValueError("User prompt cannot be empty.")

        override_map = dict(overrides or {})
        parser_fields = parser_result.normalized_dict() if parser_result else {}

        field_sources: dict[str, str] = {}

        def select(field: str, *, required: bool) -> Optional[str]:
            override_value = override_map.get(field)
            if override_value is not None:
                text = str(override_value).strip()
                if text:
                    field_sources[field] = "cli"
                    return text

            candidate = parser_fields.get(field)
            if isinstance(candidate, str):
                candidate = candidate.strip()
            if candidate:
                field_sources[field] = "task_parser_agent"
                return candidate

            if required:
                raise ValueError(
                    f"Missing required field '{field}'. Provide CLI override or update the prompt."
                )

            field_sources[field] = "unspecified"
            return None

        model_name = select("model_name", required=True)
        model_version = select("model_version", required=False)
        unlearned_target = select("unlearned_target", required=True)
        unlearning_method = select("unlearning_method", required=False)

        metadata: dict[str, Any] = {"field_sources": field_sources}

        if parser_result:
            metadata["task_parser_agent"] = {
                "raw_response": parser_result.raw_response,
                "fields": parser_result.fields,
            }

        metadata["overrides_used"] = {
            key: bool(override_map.get(key))
            for key in ("model_name", "model_version", "unlearned_target", "unlearning_method")
        }

        try:
            return cls(
                task_type=task_type,
                model_name=model_name,  # type: ignore[arg-type]
                model_version=model_version,
                base_model_path=base_model_path,
                unlearned_model_path=unlearned_model_path,
                unlearned_target=unlearned_target,  # type: ignore[arg-type]
                unlearning_method=unlearning_method,
                user_prompt=prompt_text,
                metadata=metadata,
            )
        except ValidationError as exc:
            raise ValueError(f"Invalid TaskSpec: {exc}") from exc


class Hypothesis(BaseModel):
    """
    Represents a proposed vulnerability test for unlearning methods.

    Generated by the Hypothesis Generator agent and refined through
    debate with the Critic agent.
    """

    hypothesis_id: str = Field(default_factory=lambda: str(uuid4()))
    attack_type: str = Field(
        ...,
        description="Type of attack (e.g., 'membership_inference', 'model_inversion', 'concept_leakage')",
    )
    description: str = Field(
        ..., description="Natural language description of the hypothesis"
    )
    target: str = Field(
        ..., description="What the attack targets (e.g., specific data points, concepts)"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Generator's confidence in hypothesis success",
    )
    novelty_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated novelty based on memory/RAG similarity",
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Hypothesis generation timestamp (timezone-aware)",
    )

    @field_validator("confidence_score", "novelty_score", mode="before")
    @classmethod
    def clamp_scores(cls, value: float) -> float:
        """Clamp scores into valid range prior to numeric validation."""
        if value is None:
            return value
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return value
        return max(0.0, min(1.0, numeric_value))

    @field_validator("generated_at", mode="before")
    @classmethod
    def ensure_timezone_aware(cls, v: datetime) -> datetime:
        """Ensure datetime is timezone-aware (use UTC if naive)."""
        if v and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class HypothesisContext(BaseModel):
    """
    Context information for hypothesis generation.

    Provides the Hypothesis Generator with all necessary information
    to create informed, relevant hypotheses.
    """
    iteration_number: int = Field(..., ge=1, description="Current iteration number")
    past_results: Optional[list[dict]] = Field(
        default=None,
        description="Summaries of past iteration results"
    )
    evaluator_feedback: Optional[str] = Field(
        default=None,
        description="Feedback from previous iteration's evaluator (step 4)"
    )
    retrieved_papers: Optional[list[dict]] = Field(
        default=None,
        description="Paper chunks retrieved from RAG system"
    )
    memory_entries: Optional[list[dict]] = Field(
        default=None,
        description="Similar successful hypotheses from memory"
    )
    seed_template: Optional[dict] = Field(
        default=None,
        description="Seed attack template for first iteration"
    )
    task_spec: Optional[dict[str, Any]] = Field(
        default=None,
        description="Structured TaskSpec information for this task",
    )
    critic_feedback_summaries: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Recent critic feedback summaries to guide new hypotheses (inner loop from critic)",
    )


class CriticFeedback(BaseModel):
    """
    Structured feedback from Critic agent during hypothesis debate.

    Evaluates hypotheses on three dimensions: novelty, feasibility, and rigor/detailedness.
    This model serves both as the response parser for Critic agent output and as the
    internal data structure for storing critic feedback in debate sessions.
    """
    novelty_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score for hypothesis novelty (0=derivative, 1=highly novel)"
    )
    feasibility_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score for hypothesis feasibility (0=impractical, 1=highly feasible)"
    )
    rigor_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score for hypothesis rigor/detailedness (0=vague, 1=well-defined)"
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="Identified strengths of the hypothesis"
    )
    weaknesses: list[str] = Field(
        default_factory=list,
        description="Identified weaknesses and concerns"
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Specific suggestions for improvement"
    )
    overall_assessment: str = Field(
        ...,
        description="Overall qualitative assessment of the hypothesis"
    )

    @property
    def average_score(self) -> float:
        """Calculate average score across all three dimensions."""
        return (self.novelty_score + self.feasibility_score + self.rigor_score) / 3.0


class DebateExchange(BaseModel):
    """
    Represents one round of debate between Generator and Critic.

    Tracks the full exchange including initial hypothesis, critique,
    and refined hypothesis.
    """
    round_number: int = Field(..., ge=1, description="Debate round number (1 or 2)")
    initial_hypothesis: Hypothesis = Field(..., description="Hypothesis before critique")
    critic_feedback: CriticFeedback = Field(..., description="Critic's assessment")
    refined_hypothesis: Optional[Hypothesis] = Field(
        default=None,
        description="Hypothesis after incorporating feedback (None if debate ended)"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Exchange timestamp (timezone-aware)"
    )
    generator_model: str = Field(..., description="Model used by generator")
    critic_model: str = Field(..., description="Model used by critic")
    improvement_delta: Optional[float] = Field(
        default=None,
        description="Quality score improvement from initial to refined"
    )

    @field_validator('timestamp', mode='before')
    @classmethod
    def ensure_timezone_aware(cls, v: datetime) -> datetime:
        """Ensure datetime is timezone-aware (use UTC if naive)."""
        if v and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class DebateSession(BaseModel):
    """
    Complete debate session for one iteration of the inner loop.

    Contains all debate rounds, metadata, and final output.
    """
    iteration_number: int = Field(..., ge=1, description="Loop iteration number")
    task_id: str = Field(..., description="Task identifier")
    task_type: str = Field(..., description="Task type (data_based or concept_erasure)")
    exchanges: list[DebateExchange] = Field(
        default_factory=list,
        description="Ordered list of debate exchanges"
    )
    final_hypothesis: Optional[Hypothesis] = Field(
        default=None,
        description="Final hypothesis after all debate rounds"
    )
    total_rounds: int = Field(default=0, description="Total number of debate rounds")
    debate_enabled: bool = Field(
        default=True,
        description="Whether debate was enabled for this iteration"
    )
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Debate start timestamp"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Debate completion timestamp"
    )
    convergence_reached: bool = Field(
        default=False,
        description="Whether debate converged (minimal improvement)"
    )
    quality_threshold_met: bool = Field(
        default=False,
        description="Whether quality threshold was met"
    )

    @field_validator('started_at', 'completed_at', mode='before')
    @classmethod
    def ensure_timezone_aware(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Ensure datetime is timezone-aware (use UTC if naive)."""
        if v and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate debate duration in seconds."""
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def final_quality_score(self) -> Optional[float]:
        """Get final quality score from last exchange."""
        if self.exchanges:
            return self.exchanges[-1].critic_feedback.average_score
        return None


__all__ = [
    "TaskSpec",
    "Hypothesis",
    "HypothesisContext",
    "CriticFeedback",
    "DebateExchange",
    "DebateSession",
]
