"""Shared helpers for loading agent model configurations from YAML files."""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

MODELS_DIR = Path(__file__).resolve().parents[3] / "configs" / "models"


def load_model_settings(
    name: str,
    fallback: dict[str, Any],
    *,
    models_dir: Path = MODELS_DIR,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Load model settings from a YAML file, returning a deep-copied fallback on failure."""

    active_logger = logger or logging.getLogger(__name__)
    path = models_dir / f"{name}.yaml"

    if not path.exists():
        active_logger.debug("Model config not found for %s at %s; using defaults.", name, path)
        return deepcopy(fallback)

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception as exc:  # pragma: no cover - defensive logging
        active_logger.error("Failed to load model config %s: %s", path, exc)
        return deepcopy(fallback)

    settings = deepcopy(fallback)
    if isinstance(data, dict):
        model_name = data.get("model_name")
        if isinstance(model_name, str) and model_name.strip():
            settings["model_name"] = model_name.strip()

        config_section = data.get("config")
        if isinstance(config_section, dict):
            settings_config = settings.setdefault("config", {})
            settings_config.update(
                {
                    key: value
                    for key, value in config_section.items()
                    if value is not None
                }
            )

    return settings


__all__ = ["load_model_settings", "MODELS_DIR"]
