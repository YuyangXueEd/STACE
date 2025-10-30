"""
Task parser data models for building structured research specifications.

The `TaskSpec` model centralizes validation and serialization of CLI inputs
combined with TaskParserAgent outputs so downstream agents receive a consistent
view of the research task configuration.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field, ValidationError, field_validator
from uuid import uuid4

if TYPE_CHECKING:  # pragma: no cover - import is just for typing
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
    base_model_path: Optional[str] = Field(
        default=None, description="Path to base/original model (optional)"
    )
    unlearned_model_path: str = Field(..., description="Path to unlearned/erased model")
    unlearned_target: str = Field(
        ..., description="Primary concept, dataset, or capability that was unlearned"
    )
    unlearning_method: Optional[str] = Field(
        default=None, description="Method used for unlearning (optional)"
    )
    user_prompt: str = Field(..., description="Raw user prompt provided at the CLI")
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
        base_model_path: Optional[str],
        unlearned_model_path: str,
        user_prompt: str,
        parser_result: Optional["TaskParserResult"],
        overrides: Optional[Mapping[str, Optional[str]]] = None,
    ) -> "TaskSpec":
        """
        Combine CLI overrides and task parser output into a validated TaskSpec.

        Args:
            task_type: Task type (concept_erasure or data_based_unlearning).
            base_model_path: Optional path to the base/original model.
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
            for key in (
                "model_name",
                "model_version",
                "unlearned_target",
                "unlearning_method",
            )
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


__all__ = ["TaskSpec"]
