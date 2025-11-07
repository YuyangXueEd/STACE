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

_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "configs" / "prompts"
_TASK_PARSER_PROMPT_FILE = _PROMPTS_DIR / "task_parser.yaml"
_HYPOTHESIS_TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "configs" / "hypothesis"


_STYLE_KEYWORDS = {
    "style",
    "painting",
    "illustration",
    "van gogh",
    "anime",
    "renaissance",
    "pixel art",
    "watercolor",
    "impressionism",
    "oil painting",
    "sketch",
    "photography style",
    "comic",
}

_ABSTRACT_KEYWORDS = {
    "emotion",
    "concept",
    "behavior",
    "abstract",
    "nudity",
    "violence",
    "politics",
    "religion",
    "censorship",
    "safety",
    "ethics",
    "morality",
}

_OBJECT_HINTS = {
    "object",
    "entity",
    "character",
    "person",
    "animal",
    "plant",
    "food",
    "vehicle",
    "place",
}


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
        for field_name in ("model_name", "model_version", "unlearned_target", "unlearning_method", "target_type"):
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
            if not structured.get("target_type"):
                target = structured.get("unlearned_target")
                inferred_type = self.classify_target_type(target)
                if inferred_type:
                    structured["target_type"] = inferred_type
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
        allowed_keys = {"model_name", "model_version", "unlearned_target", "unlearning_method", "target_type"}
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

    @staticmethod
    def classify_target_type(unlearned_target: Optional[str]) -> Optional[str]:
        """
        Heuristically classify an unlearned target when the LLM leaves target_type empty.

        Args:
            unlearned_target: Target text extracted from the prompt

        Returns:
            One of "object", "abstract", "style", or None if classification fails.
        """
        if not unlearned_target:
            return None

        value = str(unlearned_target).strip().lower()
        if not value:
            return None

        tokenized = {token.strip() for token in value.replace("-", " ").split()}

        # Check explicit keywords for style concepts
        if any(keyword in value for keyword in _STYLE_KEYWORDS):
            return "style"
        if any(token in _STYLE_KEYWORDS for token in tokenized):
            return "style"

        # Check abstract concept cues
        if any(keyword in value for keyword in _ABSTRACT_KEYWORDS):
            return "abstract"
        if any(token in _ABSTRACT_KEYWORDS for token in tokenized):
            return "abstract"

        # Fallback object hints
        if any(token in _OBJECT_HINTS for token in tokenized):
            return "object"

        # If the target is a single capitalized word (e.g., proper noun), treat as style
        if unlearned_target.strip().istitle() and " " in unlearned_target.strip():
            return "style"

        # Default to object for concrete single-word nouns
        if len(tokenized) == 1:
            return "object"

        return None

    @staticmethod
    def load_hypothesis_templates(target_type: Optional[str]) -> list[dict[str, Any]]:
        """
        Load hypothesis seed templates for a given target type.

        Args:
            target_type: Classification of target ('object', 'abstract', 'style', or None)

        Returns:
            List of template dictionaries with metadata and hypothesis content.
            Returns empty list if no templates found or target_type is None.
        """
        if not target_type:
            logger.info("No target_type provided, falling back to starter_template.yaml")
            starter_path = _HYPOTHESIS_TEMPLATES_DIR / "starter_template.yaml"
            if starter_path.exists():
                try:
                    with starter_path.open("r", encoding="utf-8") as f:
                        template_data = yaml.safe_load(f)
                    seed = TaskParserAgent._extract_seed_template(template_data, starter_path)
                    return [seed] if seed else []
                except Exception as exc:
                    logger.error(f"Failed to load starter template: {exc}")
                    return []
            return []

        target_dir = _HYPOTHESIS_TEMPLATES_DIR / target_type
        if not target_dir.exists() or not target_dir.is_dir():
            logger.warning(
                f"No hypothesis template directory found for target_type='{target_type}', "
                f"falling back to starter_template.yaml"
            )
            starter_path = _HYPOTHESIS_TEMPLATES_DIR / "starter_template.yaml"
            if starter_path.exists():
                try:
                    with starter_path.open("r", encoding="utf-8") as f:
                        template_data = yaml.safe_load(f)
                    seed = TaskParserAgent._extract_seed_template(template_data, starter_path)
                    return [seed] if seed else []
                except Exception as exc:
                    logger.error(f"Failed to load starter template: {exc}")
                    return []
            return []

        templates = []
        yaml_files = sorted(target_dir.glob("*.yaml")) + sorted(target_dir.glob("*.yml"))

        if not yaml_files:
            logger.warning(
                f"No YAML templates found in {target_dir}, falling back to starter_template.yaml"
            )
            starter_path = _HYPOTHESIS_TEMPLATES_DIR / "starter_template.yaml"
            if starter_path.exists():
                try:
                    with starter_path.open("r", encoding="utf-8") as f:
                        template_data = yaml.safe_load(f)
                    seed = TaskParserAgent._extract_seed_template(template_data, starter_path)
                    return [seed] if seed else []
                except Exception as exc:
                    logger.error(f"Failed to load starter template: {exc}")
                    return []
            return []

        for yaml_file in yaml_files:
            try:
                with yaml_file.open("r", encoding="utf-8") as f:
                    template_data = yaml.safe_load(f)
                seed_template = TaskParserAgent._extract_seed_template(template_data, yaml_file)
                if seed_template:
                    seed_template.setdefault("_source_path", str(yaml_file))
                    templates.append(seed_template)
                    logger.debug(f"Loaded template from {yaml_file.name}")
            except Exception as exc:
                logger.warning(f"Failed to load template {yaml_file.name}: {exc}")
                continue

        logger.info(f"Loaded {len(templates)} template(s) for target_type='{target_type}'")
        return templates

    @staticmethod
    def _extract_seed_template(raw_data: Any, source_path: Path) -> Optional[dict[str, Any]]:
        """
        Normalize raw YAML template data into a single seed_template dict.
        """
        if not raw_data:
            return None

        if isinstance(raw_data, dict) and "seed_template" in raw_data:
            seed = raw_data.get("seed_template")
        else:
            seed = raw_data

        if not isinstance(seed, dict):
            logger.warning("Template at %s does not contain a valid seed_template section", source_path)
            return None

        normalized = dict(seed)
        normalized.setdefault("id", source_path.stem)
        normalized.setdefault("source_path", str(source_path))
        return normalized


__all__ = ["TaskParserAgent", "TaskParserResult"]
