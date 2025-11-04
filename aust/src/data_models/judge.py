"""Data models representing judge evaluations and committee summaries."""

from __future__ import annotations

from statistics import mean
from typing import Iterable

from pydantic import BaseModel, Field, root_validator


class JudgeScore(BaseModel):
    """Score assigned by a judge along a specific dimension."""

    dimension: str = Field(..., description="Name of the scoring dimension")
    value: float = Field(..., ge=0.0, description="Numeric score provided by the judge")
    scale: str = Field(default="1-5", description="Scale descriptor, e.g., '1-5'")
    justification: str = Field(default="", description="Brief rationale for the score")

    class Config:
        allow_mutation = False


class JudgeEvaluation(BaseModel):
    """Structured evaluation output from a single judge persona."""

    persona_id: str = Field(..., description="Identifier of the persona")
    persona_name: str = Field(..., description="Human readable persona name")
    summary: str = Field(..., description="High level assessment summary")
    strengths: list[str] = Field(default_factory=list, description="Positive findings")
    weaknesses: list[str] = Field(default_factory=list, description="Identified risks or gaps")
    recommendations: list[str] = Field(
        default_factory=list,
        description="Actionable remediation or follow-up steps",
    )
    scores: list[JudgeScore] = Field(default_factory=list, description="Dimension level scores")
    overall_rating: float | None = Field(
        default=None,
        ge=0.0,
        description="Optional holistic rating on a consistent scale",
    )
    raw_response: str | None = Field(
        default=None,
        description="Raw LLM response prior to parsing (for debugging)",
    )

    @property
    def average_score(self) -> float | None:
        """Return the arithmetic mean of available dimension scores."""

        if not self.scores:
            return None
        return mean(score.value for score in self.scores)


class CommitteeAggregate(BaseModel):
    """Aggregate metrics derived from multiple judge evaluations."""

    run_id: str = Field(..., description="Identifier for the evaluation run")
    persona_evaluations: list[JudgeEvaluation] = Field(
        ..., description="Evaluations produced by each persona"
    )
    average_overall_rating: float | None = Field(
        default=None,
        description="Average of all provided overall ratings",
    )
    dimension_averages: dict[str, float] = Field(
        default_factory=dict,
        description="Average score per dimension across personas",
    )

    @root_validator(pre=True)
    def _compute_aggregates(cls, values: dict) -> dict:
        evaluations: Iterable[JudgeEvaluation] | None = values.get("persona_evaluations")
        if not evaluations:
            return values

        ratings = [ev.overall_rating for ev in evaluations if ev.overall_rating is not None]
        if ratings:
            values.setdefault("average_overall_rating", mean(ratings))

        dimension_scores: dict[str, list[float]] = {}
        for evaluation in evaluations:
            for score in evaluation.scores:
                key = score.dimension.lower()
                dimension_scores.setdefault(key, []).append(score.value)

        averages = {dimension: mean(values) for dimension, values in dimension_scores.items()}
        values.setdefault("dimension_averages", averages)
        return values


__all__ = ["JudgeEvaluation", "JudgeScore", "CommitteeAggregate"]