"""
LLM-backed task parser that converts free-form prompts into structured TaskSpec fields.

This implementation relies on a dedicated prompt configuration and CAMEL's structured
output support to avoid heuristic parsing entirely.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from dotenv import load_dotenv

from aust.src.utils.logging_config import get_logger
from aust.src.utils.model_config import load_model_settings

logger = get_logger(__name__)
load_dotenv()

_TASK_PARSER_MODEL_FALLBACK = {
    "model_name": "openai/gpt-5-nano",
    "config": {
        "temperature": 0.0,
        "max_tokens": 800,
        "top_p": 0.9,
    },
}

_TASK_PARSER_SETTINGS = load_model_settings(
    "task_parser", _TASK_PARSER_MODEL_FALLBACK
)

_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "configs" / "prompts"
_TASK_PARSER_PROMPT_FILE = _PROMPTS_DIR / "task_parser.yaml"


@dataclass
class TaskParserResult:
    """
    Container for extracted TaskSpec-aligned fields and raw LLM response.
    """

    fields: dict[str, Optional[str]]
    raw_response: str

    def normalized_dict(self) -> dict[str, Optional[str]]:
        """Return lowercase-key dict with trimmed string values."""
        data: dict[str, Optional[str]] = {}
        for field_name in ("model_name", "model_version", "unlearned_target", "unlearning_method"):
            value = self.fields.get(field_name) if self.fields else None
            if value is None:
                data[field_name] = None
            else:
                value_str = str(value).strip()
                data[field_name] = value_str or None
        return data


class TaskParserAgent:
    """
    LLM agent that emits structured task metadata without heuristic parsing.
    """

    DEFAULT_MODEL = _TASK_PARSER_SETTINGS["model_name"]
    _PROMPT_CACHE: Optional[dict[str, str]] = None

    @classmethod
    def _load_prompt_config(cls) -> dict[str, str]:
        if cls._PROMPT_CACHE is not None:
            return cls._PROMPT_CACHE

        if not _TASK_PARSER_PROMPT_FILE.exists():
            raise FileNotFoundError(
                f"Task parser prompt file not found at {_TASK_PARSER_PROMPT_FILE}"
            )

        with _TASK_PARSER_PROMPT_FILE.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        if not isinstance(data, dict):
            raise ValueError(
                f"Task parser prompt file {_TASK_PARSER_PROMPT_FILE} must contain a YAML mapping"
            )

        system_prompt = data.get("system_prompt")
        user_template = data.get("user_prompt_template")

        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError(
                f"system_prompt missing or empty in {_TASK_PARSER_PROMPT_FILE}"
            )
        if not isinstance(user_template, str) or not user_template.strip():
            raise ValueError(
                f"user_prompt_template missing or empty in {_TASK_PARSER_PROMPT_FILE}"
            )

        cls._PROMPT_CACHE = {
            "system_prompt": system_prompt.strip(),
            "user_prompt_template": user_template,
        }
        return cls._PROMPT_CACHE

    def __init__(self, model_name: Optional[str] = None, temperature: Optional[float] = None):
        model_config = deepcopy(_TASK_PARSER_SETTINGS.get("config", {}))
        if temperature is not None:
            model_config["temperature"] = temperature

        config = ChatGPTConfig(**model_config)

        prompt_config = self._load_prompt_config()
        system_prompt = prompt_config["system_prompt"]
        self._user_prompt_template = prompt_config["user_prompt_template"]

        backend = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=model_name or self.DEFAULT_MODEL,
            url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model_config_dict=config.as_dict(),
        )

        self._agent = ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name="TaskParser",
                content=system_prompt,
            ),
            model=backend,
        )

    def parse_prompt(self, prompt: str, task_type: Optional[str] = None) -> TaskParserResult:
        """
        Parse a natural language prompt into structured fields.

        Args:
            prompt: Raw user prompt describing the unlearning scenario.

        Returns:
            TaskParserResult containing normalized fields and diagnostics.

        Raises:
            ValueError: If the agent returns malformed content or cannot be parsed.
        """
        prompt_text = prompt.strip()
        safe_prompt = prompt_text.replace("{", "{{").replace("}", "}}").strip()
        format_args = {
            "user_prompt": safe_prompt,
            "task_type": (task_type or "unknown").strip(),
        }
        user_payload = self._user_prompt_template.format(**format_args)

        user_message = BaseMessage.make_user_message(
            role_name="User",
            content=user_payload,
        )

        try:
            response = self._agent.step(user_message)
            message = self._select_last_message(response)
            structured = self._extract_structured_payload(message)
            return TaskParserResult(fields=structured, raw_response=message.content)
        finally:
            self._agent.reset()

    def _select_last_message(self, response) -> Any:
        messages = getattr(response, "msgs", None) or []
        if not messages:
            raise ValueError("TaskParserAgent returned no messages.")
        return messages[-1]

    def _extract_structured_payload(self, message) -> dict[str, Optional[str]]:
        """
        Convert a CAMEL message into a dict of TaskSpec-compatible fields.
        """
        parsed = getattr(message, "parsed", None)
        if parsed and isinstance(parsed, dict):
            return self._normalize_fields(parsed)

        content = message.content or ""
        try:
            payload = self._load_json_block(content)
            if not isinstance(payload, dict):
                raise ValueError("Parsed payload is not a JSON object.")
            return self._normalize_fields(payload)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("Unable to parse TaskParserAgent response: %s", content)
            raise ValueError(f"TaskParserAgent returned invalid JSON: {exc}") from exc

    @staticmethod
    def _normalize_fields(payload: dict[str, Any]) -> dict[str, Optional[str]]:
        allowed_keys = {"model_name", "model_version", "unlearned_target", "unlearning_method"}
        # Backward compatibility: accept 'unlearned_concept' and map to 'unlearned_target'
        if "unlearned_target" not in payload and "unlearned_concept" in payload:
            payload = dict(payload)
            payload["unlearned_target"] = payload.get("unlearned_concept")
        normalized: dict[str, Optional[str]] = {}
        for key in allowed_keys:
            value = payload.get(key)
            if value is None:
                normalized[key] = None
            else:
                normalized[key] = str(value).strip() or None
        return normalized

    @staticmethod
    def _load_json_block(content: str) -> Any:
        """
        Load JSON from content, handling optional Markdown fences without regex.
        """
        text = content.strip()
        if text.startswith("```"):
            # Drop the leading fence (``` or ```json)
            fence_trimmed = text[3:]
            if "\n" in fence_trimmed:
                fence_trimmed = fence_trimmed.split("\n", 1)[1]
            text = fence_trimmed
            if text.endswith("```"):
                text = text[:-3]
        return json.loads(text.strip())


__all__ = ["TaskParserAgent", "TaskParserResult"]
