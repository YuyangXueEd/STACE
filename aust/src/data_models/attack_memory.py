"""
Data models for long-term attack memory system.

This module defines the structured format for storing and retrieving
successful vulnerability discoveries across multiple runs.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class AttackMemoryCard(BaseModel):
    """
    Structured memory card for a successful vulnerability discovery.

    Stored as markdown files in outputs/memory_store/attacks/{attack_id}.md
    """

    attack_id: str = Field(..., description="Unique identifier for this attack")
    task_type: str = Field(
        ..., description="Task type: concept_erasure or data_based_unlearning"
    )
    unlearned_target: str = Field(
        ..., description="Target concept or content that was unlearned"
    )
    unlearning_method: Optional[str] = Field(
        None, description="Method used (e.g., ESD, SCRUB, etc.)"
    )
    model_name: Optional[str] = Field(
        None, description="Model tested (e.g., Stable Diffusion 1.4)"
    )

    # Hypothesis information
    hypothesis_attack_type: str = Field(..., description="Type of attack used")
    hypothesis_summary: str = Field(
        ..., description="Brief summary of the hypothesis"
    )
    hypothesis_reasoning: str = Field(
        ..., description="Reasoning behind the hypothesis"
    )
    hypothesis_full: dict[str, Any] = Field(
        ..., description="Complete hypothesis object as dict"
    )

    # Experiment results
    experiment_parameters: dict[str, Any] = Field(
        ..., description="Parameters used in the experiment"
    )
    detection_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Vulnerability detection rate (0-1)"
    )
    max_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Maximum confidence score (0-1)"
    )
    vulnerability_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Overall vulnerability confidence (0-1)"
    )
    successful_prompts: list[dict[str, Any]] = Field(
        default_factory=list, description="Prompts that successfully triggered leakage"
    )
    key_findings: list[str] = Field(
        default_factory=list, description="Key insights from this attack"
    )

    # Metadata
    attack_trace_path: str = Field(
        ..., description="Path to full attack trace markdown file"
    )
    discovered_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of discovery (UTC)",
    )
    iteration_number: int = Field(
        ..., ge=1, description="Iteration number when vulnerability was found"
    )

    @field_validator("discovered_at")
    @classmethod
    def validate_timezone_aware(cls, v: datetime) -> datetime:
        """Ensure datetime is timezone-aware (UTC)."""
        if v.tzinfo is None:
            raise ValueError("discovered_at must be timezone-aware")
        return v

    def to_markdown(self) -> str:
        """
        Generate markdown representation of attack memory card.

        Returns:
            Markdown-formatted attack memory card with structured sections
        """
        # Format discovered_at
        discovered_str = self.discovered_at.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Format successful prompts
        prompts_section = ""
        if self.successful_prompts:
            prompts_section = "**Successful Prompts**:\n"
            for prompt_info in self.successful_prompts:
                prompt_text = prompt_info.get("prompt", "unknown")
                rate = prompt_info.get("detection_rate", 0.0)
                prompts_section += f"- \"{prompt_text}\" → {rate:.1%} detection\n"
        else:
            prompts_section = "**Successful Prompts**: None specifically identified\n"

        # Format key findings
        findings_section = ""
        if self.key_findings:
            for i, finding in enumerate(self.key_findings, 1):
                findings_section += f"{i}. {finding}\n"
        else:
            findings_section = "No specific lessons documented.\n"

        markdown = f"""# Attack Memory: {self.attack_id}

## [METADATA]
- **Task Type**: {self.task_type}
- **Target Concept**: {self.unlearned_target}
- **Unlearning Method**: {self.unlearning_method or "Unknown"}
- **Model**: {self.model_name or "Unknown"}
- **Discovered**: {discovered_str}
- **Iteration**: {self.iteration_number}
- **Detection Rate**: {self.detection_rate:.1%}
- **Max Confidence**: {self.max_confidence:.2f}
- **Overall Confidence**: {self.vulnerability_confidence:.2f}
- **Attack Trace**: {self.attack_trace_path}

## [HYPOTHESIS]
**Attack Type**: {self.hypothesis_attack_type}

**Summary**: {self.hypothesis_summary}

**Reasoning**: {self.hypothesis_reasoning}

## [EXPERIMENT_PARAMETERS]
```json
{self._format_json(self.experiment_parameters)}
```

## [RESULTS]
**Detection Rate**: {self.detection_rate:.1%}

**Max Confidence**: {self.max_confidence:.2f}

{prompts_section}

## [LESSONS_LEARNED]
{findings_section}
"""
        return markdown

    def _format_json(self, obj: Any, indent: int = 2) -> str:
        """Format JSON with proper indentation."""
        import json

        return json.dumps(obj, indent=indent, default=str)

    @classmethod
    def from_markdown(cls, markdown_content: str, attack_id: str) -> "AttackMemoryCard":
        """
        Parse attack memory card from markdown.

        Args:
            markdown_content: Markdown content to parse
            attack_id: Attack ID from filename

        Returns:
            AttackMemoryCard instance

        Note:
            This is a simplified parser. For production, consider using
            more robust markdown parsing libraries.
        """
        import json
        import re

        # Extract metadata section
        metadata_match = re.search(
            r"## \[METADATA\]\n(.*?)(?=\n## \[|$)", markdown_content, re.DOTALL
        )
        metadata_text = metadata_match.group(1) if metadata_match else ""

        # Parse metadata fields
        def extract_field(text: str, field: str) -> Optional[str]:
            match = re.search(rf"- \*\*{field}\*\*:\s*(.+)", text)
            return match.group(1).strip() if match else None

        task_type = extract_field(metadata_text, "Task Type") or "unknown"
        unlearned_target = extract_field(metadata_text, "Target Concept") or "unknown"
        unlearning_method = extract_field(metadata_text, "Unlearning Method")
        model_name = extract_field(metadata_text, "Model")
        discovered_str = extract_field(metadata_text, "Discovered")
        detection_rate_str = extract_field(metadata_text, "Detection Rate")
        max_confidence_str = extract_field(metadata_text, "Max Confidence")
        overall_confidence_str = extract_field(metadata_text, "Overall Confidence")
        iteration_str = extract_field(metadata_text, "Iteration")
        attack_trace_path = extract_field(metadata_text, "Attack Trace") or ""

        # Parse discovered timestamp
        discovered_at = datetime.now(timezone.utc)
        if discovered_str:
            try:
                discovered_at = datetime.strptime(discovered_str, "%Y-%m-%dT%H:%M:%SZ")
                discovered_at = discovered_at.replace(tzinfo=timezone.utc)
            except ValueError:
                pass

        # Parse hypothesis section
        hypothesis_match = re.search(
            r"## \[HYPOTHESIS\]\n(.*?)(?=\n## \[|$)", markdown_content, re.DOTALL
        )
        hypothesis_text = hypothesis_match.group(1) if hypothesis_match else ""

        attack_type_match = re.search(r"\*\*Attack Type\*\*:\s*(.+)", hypothesis_text)
        hypothesis_attack_type = (
            attack_type_match.group(1).strip() if attack_type_match else "Unknown"
        )

        summary_match = re.search(r"\*\*Summary\*\*:\s*(.+)", hypothesis_text)
        hypothesis_summary = (
            summary_match.group(1).strip() if summary_match else "No summary"
        )

        reasoning_match = re.search(r"\*\*Reasoning\*\*:\s*(.+)", hypothesis_text, re.DOTALL)
        hypothesis_reasoning = (
            reasoning_match.group(1).strip() if reasoning_match else "No reasoning"
        )

        # Parse experiment parameters
        params_match = re.search(
            r"## \[EXPERIMENT_PARAMETERS\]\n```json\n(.*?)\n```",
            markdown_content,
            re.DOTALL,
        )
        experiment_parameters = {}
        if params_match:
            try:
                experiment_parameters = json.loads(params_match.group(1))
            except json.JSONDecodeError:
                pass

        # Parse numeric fields
        detection_rate = 0.0
        if detection_rate_str:
            detection_rate = float(detection_rate_str.rstrip("%")) / 100.0

        max_confidence = 0.0
        if max_confidence_str:
            max_confidence = float(max_confidence_str)

        vulnerability_confidence = 0.0
        if overall_confidence_str:
            vulnerability_confidence = float(overall_confidence_str)

        iteration_number = 1
        if iteration_str:
            iteration_number = int(iteration_str)

        return cls(
            attack_id=attack_id,
            task_type=task_type,
            unlearned_target=unlearned_target,
            unlearning_method=unlearning_method,
            model_name=model_name,
            hypothesis_attack_type=hypothesis_attack_type,
            hypothesis_summary=hypothesis_summary,
            hypothesis_reasoning=hypothesis_reasoning,
            hypothesis_full={},  # Not stored in markdown
            experiment_parameters=experiment_parameters,
            detection_rate=detection_rate,
            max_confidence=max_confidence,
            vulnerability_confidence=vulnerability_confidence,
            successful_prompts=[],  # Could parse from RESULTS section if needed
            key_findings=[],  # Could parse from LESSONS_LEARNED if needed
            attack_trace_path=attack_trace_path,
            discovered_at=discovered_at,
            iteration_number=iteration_number,
        )


__all__ = ["AttackMemoryCard"]
