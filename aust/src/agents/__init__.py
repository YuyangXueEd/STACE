"""Agent entry points for CAUST."""

# Lazy imports to avoid circular dependencies and loading heavy dependencies
__all__ = [
    "DeepUnlearnEvaluationAgent",
    "DeepUnlearnOrchestrator",
    "PaperCardAgent",
    "PaperCardPdfAgent",
]


def __getattr__(name):
    """Lazy import of agents to avoid loading heavy dependencies unnecessarily."""
    if name == "DeepUnlearnEvaluationAgent":
        from .data_based.deepunlearn_evaluator import DeepUnlearnEvaluationAgent

        return DeepUnlearnEvaluationAgent
    elif name == "DeepUnlearnOrchestrator":
        from .data_based.deepunlearn_orchestrator import DeepUnlearnOrchestrator

        return DeepUnlearnOrchestrator
    elif name == "PaperCardAgent":
        from .paper_card_agent import PaperCardAgent

        return PaperCardAgent
    elif name == "PaperCardPdfAgent":
        from .paper_card_pdf_agent import PaperCardPdfAgent

        return PaperCardPdfAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
