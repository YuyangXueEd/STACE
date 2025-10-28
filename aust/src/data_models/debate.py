"""
Debate session data models.

Provide the structured representation of generator/critic exchanges used by the
HypothesisRefinementWorkforce.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from aust.src.data_models.critic import CriticFeedback
from aust.src.data_models.hypothesis import Hypothesis


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
        description="Hypothesis after incorporating feedback (None if debate ended)",
    )
    rag_queries: list[str] = Field(
        default_factory=list,
        description="Queries issued to PaperRAG after the critic round",
    )
    retrieval_context: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Paper snippets returned by RAG to guide the next hypothesis",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Exchange timestamp (timezone-aware)",
    )
    generator_model: str = Field(..., description="Model used by generator")
    critic_model: str = Field(..., description="Model used by critic")
    improvement_delta: Optional[float] = Field(
        default=None,
        description="Quality score improvement from initial to refined",
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_timezone_aware(cls, value: datetime) -> datetime:
        """Ensure datetime is timezone-aware (use UTC if naive)."""
        if value and value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


class DebateSession(BaseModel):
    """
    Complete debate session for one iteration of the inner loop.

    Contains all debate rounds, metadata, and final output.
    """

    iteration_number: int = Field(..., ge=1, description="Loop iteration number")
    task_id: str = Field(..., description="Task identifier")
    task_type: str = Field(..., description="Task type (data_based or concept_erasure)")
    exchanges: list[DebateExchange] = Field(
        default_factory=list, description="Ordered list of debate exchanges"
    )
    final_hypothesis: Optional[Hypothesis] = Field(
        default=None, description="Final hypothesis after all debate rounds"
    )
    total_rounds: int = Field(default=0, description="Total number of debate rounds")
    debate_enabled: bool = Field(
        default=True, description="Whether debate was enabled for this iteration"
    )
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Debate start timestamp",
    )
    completed_at: Optional[datetime] = Field(
        default=None, description="Debate completion timestamp"
    )
    convergence_reached: bool = Field(
        default=False, description="Whether debate converged (minimal improvement)"
    )
    quality_threshold_met: bool = Field(
        default=False, description="Whether quality threshold was met"
    )

    @field_validator("started_at", "completed_at", mode="before")
    @classmethod
    def ensure_timezone_aware(cls, value: Optional[datetime]) -> Optional[datetime]:
        """Ensure datetime is timezone-aware (use UTC if naive)."""
        if value and value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

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

    @property
    def rag_queries(self) -> list[str]:
        """Flatten all RAG queries issued during this debate."""
        queries: list[str] = []
        for exchange in self.exchanges:
            if exchange.rag_queries:
                queries.extend(exchange.rag_queries)
        return queries

    @property
    def retrieved_papers(self) -> list[dict[str, Any]]:
        """Flatten retrieved paper context from every exchange."""
        papers: list[dict[str, Any]] = []
        for exchange in self.exchanges:
            if exchange.retrieval_context:
                papers.extend(exchange.retrieval_context)
        return papers


__all__ = ["DebateExchange", "DebateSession"]
