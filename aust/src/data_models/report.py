"""
Data models for academic report generation (Story 4.1, 4.2).

Defines structures for report sections, metadata, and complete academic reports
with citation tracking and section management.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class ReportSectionType(str, Enum):
    """Standard academic report sections."""

    INTRODUCTION = "introduction"
    METHODS = "methods"
    EXPERIMENTS = "experiments"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"


class ReportSection(BaseModel):
    """Individual section of an academic report."""

    section_type: ReportSectionType = Field(..., description="Type of section")
    title: str = Field(..., description="Section title (e.g., 'Introduction')")
    content: str = Field(default="", description="Section content in Markdown")
    citations: list[str] = Field(
        default_factory=list, description="List of citation keys (e.g., arxiv IDs) used in this section"
    )
    subsections: dict[str, str] = Field(
        default_factory=dict,
        description="Optional subsections mapping subsection title to content",
    )
    order: int = Field(default=0, ge=0, description="Display order in report")

    @property
    def word_count(self) -> int:
        """Approximate word count of section content."""
        return len(self.content.split())

    @property
    def has_content(self) -> bool:
        """Check if section has actual content."""
        return bool(self.content.strip())

    def add_citation(self, citation_key: str) -> None:
        """
        Add a citation to this section if not already present.

        Args:
            citation_key: Citation identifier (e.g., 'arxiv:2101.00001')
        """
        if citation_key not in self.citations:
            self.citations.append(citation_key)


class NoveltyInfo(BaseModel):
    """Novelty calculation results for a hypothesis."""

    novelty_score: float = Field(..., ge=0.0, le=1.0, description="Novelty score (1 - max_similarity)")
    max_similarity: float = Field(..., ge=0.0, le=1.0, description="Maximum similarity to existing papers")
    top_similar_papers: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Top 3 most similar papers with scores",
    )
    calculation_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When novelty was calculated",
    )


class ReportMetadata(BaseModel):
    """Metadata for an academic report."""

    report_id: str = Field(..., description="Unique report identifier")
    task_id: str = Field(..., description="Associated task ID from inner loop")
    task_type: str = Field(..., description="Type of task (concept_erasure, data_based_unlearning)")

    # Target information
    target_model_name: Optional[str] = Field(default=None, description="Name of target model")
    target_model_version: Optional[str] = Field(default=None, description="Version of target model")
    unlearning_method: Optional[str] = Field(default=None, description="Unlearning method used")
    unlearned_target: Optional[str] = Field(default=None, description="Target concept/data unlearned")

    # Loop execution info
    total_iterations: int = Field(default=0, ge=0, description="Number of inner loop iterations")
    vulnerability_found: bool = Field(default=False, description="Whether vulnerability was discovered")
    highest_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Highest vulnerability confidence")

    # Timestamps
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Report generation timestamp",
    )
    inner_loop_started_at: Optional[datetime] = Field(default=None, description="Inner loop start time")
    inner_loop_completed_at: Optional[datetime] = Field(default=None, description="Inner loop completion time")

    # Additional metadata
    generator_model: Optional[str] = Field(default=None, description="Model used for hypothesis generation")
    critic_model: Optional[str] = Field(default=None, description="Model used for critique")

    # Novelty information (Story 5.1)
    novelty_info: Optional[NoveltyInfo] = Field(default=None, description="Hypothesis novelty calculation")

    @property
    def duration_hours(self) -> Optional[float]:
        """Calculate inner loop duration in hours."""
        if self.inner_loop_started_at and self.inner_loop_completed_at:
            duration = self.inner_loop_completed_at - self.inner_loop_started_at
            return duration.total_seconds() / 3600
        return None


class AcademicReport(BaseModel):
    """
    Complete academic report with all sections and metadata.

    This model represents a fully-formed academic paper generated from
    inner loop results, including standard sections, citations, and metadata.
    """

    metadata: ReportMetadata = Field(..., description="Report metadata")
    sections: dict[ReportSectionType, ReportSection] = Field(
        default_factory=dict, description="Report sections indexed by type"
    )
    references: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Bibliography mapping citation keys to paper metadata",
    )

    # Output paths
    output_path: Optional[Path] = Field(default=None, description="Path where report was saved")

    @property
    def total_word_count(self) -> int:
        """Calculate total word count across all sections."""
        return sum(section.word_count for section in self.sections.values())

    @property
    def all_citations(self) -> list[str]:
        """Get all unique citations across all sections."""
        all_cites = []
        for section in self.sections.values():
            all_cites.extend(section.citations)
        return list(set(all_cites))  # Remove duplicates

    @property
    def is_complete(self) -> bool:
        """Check if report has all required sections with content."""
        required_sections = [
            ReportSectionType.INTRODUCTION,
            ReportSectionType.METHODS,
            ReportSectionType.RESULTS,
            ReportSectionType.CONCLUSION,
        ]
        return all(
            section_type in self.sections and self.sections[section_type].has_content
            for section_type in required_sections
        )

    def add_section(self, section: ReportSection) -> None:
        """
        Add or update a section in the report.

        Args:
            section: ReportSection to add
        """
        self.sections[section.section_type] = section

    def get_section(self, section_type: ReportSectionType) -> Optional[ReportSection]:
        """
        Retrieve a section by type.

        Args:
            section_type: Type of section to retrieve

        Returns:
            ReportSection if exists, None otherwise
        """
        return self.sections.get(section_type)

    def get_ordered_sections(self) -> list[ReportSection]:
        """
        Get sections in display order.

        Returns:
            List of sections sorted by order field
        """
        return sorted(self.sections.values(), key=lambda s: s.order)

    def add_reference(self, citation_key: str, paper_metadata: dict[str, Any]) -> None:
        """
        Add a reference to the bibliography.

        Args:
            citation_key: Citation identifier
            paper_metadata: Paper metadata dictionary
        """
        self.references[citation_key] = paper_metadata

    def to_markdown(self) -> str:
        """
        Generate complete report in Markdown format.

        Returns:
            Complete report as Markdown string
        """
        lines = []

        # Title and metadata
        lines.append(f"# Vulnerability Assessment Report: {self.metadata.task_id}")
        lines.append("")
        lines.append(f"**Generated**: {self.metadata.generated_at.strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append(f"**Task Type**: {self.metadata.task_type}")

        if self.metadata.target_model_name:
            lines.append(f"**Target Model**: {self.metadata.target_model_name}")
            if self.metadata.target_model_version:
                lines[-1] += f" v{self.metadata.target_model_version}"

        if self.metadata.unlearning_method:
            lines.append(f"**Unlearning Method**: {self.metadata.unlearning_method}")

        if self.metadata.unlearned_target:
            lines.append(f"**Unlearned Target**: {self.metadata.unlearned_target}")

        lines.append(f"**Iterations**: {self.metadata.total_iterations}")
        lines.append(f"**Vulnerability Found**: {'Yes' if self.metadata.vulnerability_found else 'No'}")

        if self.metadata.vulnerability_found:
            lines.append(f"**Highest Confidence**: {self.metadata.highest_confidence:.1%}")

        lines.append("")
        lines.append("---")
        lines.append("")

        # Abstract / Summary
        lines.append("## Abstract")
        lines.append("")
        lines.append(self._generate_abstract())
        lines.append("")

        # Sections in order
        for section in self.get_ordered_sections():
            lines.append(f"## {section.title}")
            lines.append("")
            lines.append(section.content)
            lines.append("")

        # References
        if self.references:
            lines.append("## References")
            lines.append("")
            for idx, (citation_key, paper_meta) in enumerate(sorted(self.references.items()), 1):
                lines.append(self._format_reference(idx, citation_key, paper_meta))
            lines.append("")

        return "\n".join(lines)

    def _generate_abstract(self) -> str:
        """Generate abstract based on metadata."""
        abstract_parts = [
            f"This report presents an automated vulnerability assessment of the {self.metadata.target_model_name or 'target'} model",
        ]

        if self.metadata.unlearning_method:
            abstract_parts.append(
                f"using the {self.metadata.unlearning_method} unlearning method"
            )

        abstract_parts.append(
            f"across {self.metadata.total_iterations} iterative hypothesis-testing cycles."
        )

        if self.metadata.vulnerability_found:
            abstract_parts.append(
                f"The assessment successfully identified vulnerabilities with {self.metadata.highest_confidence:.1%} confidence,"
                " demonstrating potential weaknesses in the unlearning process."
            )
        else:
            abstract_parts.append(
                "The assessment did not identify critical vulnerabilities within the testing scope."
            )

        return " ".join(abstract_parts)

    def _format_reference(self, index: int, citation_key: str, paper_meta: dict) -> str:
        """Format a reference entry."""
        # Basic format: [1] Author et al. (Year). Title. arXiv:XXXX.XXXXX
        author = paper_meta.get("author", "Unknown")
        year = paper_meta.get("year", "n.d.")
        title = paper_meta.get("title", "Untitled")

        # Simplify author if it's a list
        if isinstance(author, list):
            if len(author) > 2:
                author = f"{author[0]} et al."
            else:
                author = " and ".join(author)

        ref = f"[{index}] {author} ({year}). {title}."

        if "arxiv_id" in paper_meta:
            ref += f" arXiv:{paper_meta['arxiv_id']}"
        elif citation_key.startswith("arxiv:"):
            ref += f" {citation_key}"

        return ref

    def save_to_file(self, file_path: Path) -> None:
        """
        Save report to Markdown file.

        Args:
            file_path: Path to save report
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(self.to_markdown(), encoding="utf-8")
        self.output_path = file_path
