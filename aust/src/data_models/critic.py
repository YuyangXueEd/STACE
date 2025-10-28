"""
Critic agent data models.

Defines the structured feedback schema returned by the Critic agent during
debate rounds with the Hypothesis Generator.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


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
        description="Score for hypothesis novelty (0=derivative, 1=highly novel)",
    )
    feasibility_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score for hypothesis feasibility (0=impractical, 1=highly feasible)",
    )
    rigor_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score for hypothesis rigor/detailedness (0=vague, 1=well-defined)",
    )
    strengths: list[str] = Field(
        default_factory=list, description="Identified strengths of the hypothesis"
    )
    weaknesses: list[str] = Field(
        default_factory=list, description="Identified weaknesses and concerns"
    )
    suggestions: list[str] = Field(
        default_factory=list, description="Specific suggestions for improvement"
    )
    overall_assessment: str = Field(
        ..., description="Overall qualitative assessment of the hypothesis"
    )
    overall_assumption: Optional[str] = Field(
        default=None,
        description="Critic's high-level assumption about why the hypothesis may succeed or fail",
    )

    @property
    def average_score(self) -> float:
        """Calculate average score across all three dimensions."""
        return (self.novelty_score + self.feasibility_score + self.rigor_score) / 3.0


__all__ = ["CriticFeedback"]
