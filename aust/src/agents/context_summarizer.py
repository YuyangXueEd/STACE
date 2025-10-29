"""
Lightweight context summarizer agent for keeping prompts within token budgets.

This agent delegates summarisation to a compact OpenRouter-compatible model
and preserves markdown structure (section headings, bullet lists) whenever
possible. It is designed as a last resort when heuristic truncation cannot
keep prompts below the configured character budget.
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Optional
from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType

from aust.src.utils.logging_config import get_logger
from aust.src.utils.model_config import load_model_settings

logger = get_logger(__name__)

_CONTEXT_SUMMARIZER_MODEL_FALLBACK = {
    "model_name": "openai/gpt-5-nano",
    "config": {
        "temperature": 0.0,
        "max_tokens": 600,
        "top_p": 0.9,
    },
}
_CONTEXT_SUMMARIZER_SETTINGS = load_model_settings(
    "context_summarizer", _CONTEXT_SUMMARIZER_MODEL_FALLBACK
)


class ContextSummarizerAgent:
    """LLM-backed summariser that condenses long prompts before model calls."""

    DEFAULT_MODEL = _CONTEXT_SUMMARIZER_SETTINGS["model_name"]

    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Initialize summariser agent.

        Args:
            model_name: Optional override for the summariser model
            temperature: Sampling temperature for the summariser
            max_tokens: Maximum tokens for the summariser response
        """
        config_dict = deepcopy(_CONTEXT_SUMMARIZER_SETTINGS.get("config", {}))
        if temperature is not None:
            config_dict["temperature"] = temperature
        if max_tokens is not None:
            config_dict["max_tokens"] = max_tokens

        config = ChatGPTConfig(**config_dict)

        backend = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=model_name or self.DEFAULT_MODEL,
            url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model_config_dict=config.as_dict(),
        )

        system_prompt = (
            "You are a concise research analyst. Compress the provided context so "
            "that it stays within a strict character budget while preserving:\n"
            "- Key facts and numerical details relevant to machine unlearning tasks\n"
            "- Markdown headings (###, ####) and bullet structure when helpful\n"
            "- Critical differences between past iterations and feedback\n\n"
            "Remove redundant wording, verbose rationale, or repeated examples."
        )

        self._agent = ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name="ContextSummarizer",
                content=system_prompt,
            ),
            model=backend,
        )

    def summarize(self, text: str, *, target_chars: int) -> str:
        """
        Summarise `text` to stay within `target_chars`.

        Args:
            text: Raw context string
            target_chars: Character budget to respect

        Returns:
            Summarised text (best effort). Falls back to trimmed original if
            the model output exceeds the budget or is empty.
        """
        if not text or len(text) <= target_chars:
            return text

        user_prompt = (
            f"Summarise the context below to <= {target_chars} characters. "
            "Keep the most decision-critical details for planning requests to "
            "machine unlearning research models. Retain headings where meaningful. "
            "If you must drop information, prefer removing redundant or low-impact "
            "details.\n\n"
            "--- BEGIN CONTEXT ---\n"
            f"{text}\n"
            "--- END CONTEXT ---"
        )

        message = BaseMessage.make_user_message(role_name="Orchestrator", content=user_prompt)

        try:
            response = self._agent.step(message)
            content = response.msgs[-1].content.strip() if response and response.msgs else ""
        finally:
            self._agent.reset()

        if not content:
            logger.warning("Context summariser returned empty output; using fallback trim.")
            return self._fallback_trim(text, target_chars)

        if len(content) > target_chars:
            logger.debug(
                "Summariser output exceeds target (%s > %s); applying hard trim.",
                len(content),
                target_chars,
            )
            return self._fallback_trim(content, target_chars)

        return content

    @staticmethod
    def _fallback_trim(text: str, target_chars: int) -> str:
        """Hard truncate text when no better summarisation is available."""
        if len(text) <= target_chars:
            return text
        if target_chars <= 3:
            return text[:target_chars]
        return text[: target_chars - 3] + "..."


__all__ = ["ContextSummarizerAgent"]
