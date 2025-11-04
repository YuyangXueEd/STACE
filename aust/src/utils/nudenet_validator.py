"""Helper utilities for NudeNet-based nudity validation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from aust.src.utils.logging_config import get_logger

logger = get_logger(__name__)

_NUDITY_KEYWORDS: Sequence[str] = ("nudity", "nude", "nsfw", "explicit", "adult")


def is_nudity_concept(concept: str) -> bool:
    """Return True when the supplied concept implies a nudity-related task."""

    if not concept:
        return False
    lowered = concept.lower()
    return any(keyword in lowered for keyword in _NUDITY_KEYWORDS)


def run_nudenet_validation(
    image_paths: Iterable[str | Path],
    *,
    max_images: int | None = None,
) -> dict[str, object]:
    """Execute NudeNet nudity detection for the provided images.

    Args:
        image_paths: Paths to images that should be analyzed.
        max_images: Optional cap on how many images to analyze (taken in order).

    Returns:
        A result dictionary from NudeNetToolkit or an error payload when NudeNet is unavailable.
    """

    try:
        from aust.src.toolkits.nudenet_toolkit import NudeNetToolkit
    except ImportError:
        logger.warning("NudeNet not installed. Install with: pip install nudenet>=3.4.2")
        return {"error": "NudeNet not available"}

    path_list: list[str] = []
    for path in image_paths:
        if path is None:
            continue
        if isinstance(path, Path):
            resolved = str(path)
        else:
            resolved = str(path)
        path_list.append(resolved)
        if max_images is not None and len(path_list) >= max_images:
            break

    if not path_list:
        return {"error": "No images supplied for NudeNet validation"}

    try:
        toolkit = NudeNetToolkit()
        return toolkit.detect_nudity_batch(path_list)
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        logger.error("NudeNet validation failed: %s", exc)
        return {"error": str(exc)}
