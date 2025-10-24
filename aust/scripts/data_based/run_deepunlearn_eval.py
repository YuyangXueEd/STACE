"""Utility script to drive the DeepUnlearn evaluation agent.

This helper mirrors :mod:`run_deepunlearn_pipeline` but targets the
``deepunlearn_evaluator`` agent to automate post-training evaluation of an
original and unlearned checkpoint.  It prepares a structured prompt, forwards
the necessary artifact paths, and records the evaluation transcript.

Example usage:
python -m aust.scripts.run_deepunlearn_eval \
  --dataset cifar10 \
  --model-arch resnet18 \
  --splits-dir data/cifar10/splits \
  --original-model data/cifar10/10_resnet18_original.pth \
  --unlearned-model data/cifar10/10_resnet18_unlearned_finetune.pth

"""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent
from typing import List, Optional, Sequence

from aust.src.agents import deepunlearn_evaluator


def _render_context_block(
    *,
    dataset: str,
    model_arch: str,
    splits_dir: Path,
    original_model: Path,
    unlearned_model: Path,
    batch_size: int,
    random_state: int,
    device: str,
    split_index: Optional[int],
    forget_index: Optional[int],
) -> str:
    """Build the context section appended to the agent prompt."""
    lines = [
        f"- dataset: {dataset}",
        f"- model_type: {model_arch}",
        f"- splits_directory: {splits_dir}",
        f"- original_model_checkpoint: {original_model}",
        f"- unlearned_model_checkpoint: {unlearned_model}",
        f"- evaluation_batch_size: {batch_size}",
        f"- random_state: {random_state}",
        f"- evaluation_device: {device}",
    ]
    if split_index is not None and forget_index is not None:
        lines.append(
            f"- lira_indices: split_index={split_index}, forget_index={forget_index}"
        )
    return "Context:\n" + "\n".join(lines)


def _render_evaluation_prompt(
    *,
    dataset: str,
    model_arch: str,
    splits_dir: Path,
    original_model: Path,
    unlearned_model: Path,
    batch_size: int,
    random_state: int,
    device: str,
    split_index: Optional[int],
    forget_index: Optional[int],
    metrics: Optional[Sequence[str]],
) -> str:
    """Create the instruction prompt delivered to the evaluation agent."""
    metrics_list = [metric.strip() for metric in metrics or [] if metric and metric.strip()]
    if metrics_list:
        metrics_clause = (
            "the following evaluation metric(s): " + ", ".join(metrics_list)
        )
        step_two = (
            "Invoke only the relevant toolkit functions to obtain those metrics "
            "and compare the original and unlearned models."
        )
    else:
        metrics_clause = (
            "all available evaluation metrics (retain/forget accuracy, validation/test "
            "accuracy, membership inference scores, indiscernibility, retention ratio, "
            "SAPE, and model weight distance)"
        )
        step_two = (
            "Collect the full suite of metrics, including indiscernibility, accuracy "
            "retention, SAPE, and model weight distance, in addition "
            "to the per-split accuracies and MIA scores."
        )

    lira_clause = ""
    if split_index is not None and forget_index is not None:
        lira_clause = (
            f", split_index={split_index}, forget_index={forget_index}"
        )

    context = _render_context_block(
        dataset=dataset,
        model_arch=model_arch,
        splits_dir=splits_dir,
        original_model=original_model,
        unlearned_model=unlearned_model,
        batch_size=batch_size,
        random_state=random_state,
        device=device,
        split_index=split_index,
        forget_index=forget_index,
    )

    return dedent(
        f"""\
        Test {metrics_clause} for the provided DeepUnlearn artifacts.

        Use the DeepUnlearnEvaluationToolkit to:
        1. Call `evaluate_model_checkpoint` for both checkpoints with dataset="{dataset}", model_type="{model_arch}", splits_dir="{splits_dir}", batch_size={batch_size}, random_state={random_state}, device="{device}"{lira_clause}.
        2. {step_two}
        3. Summarise the findings in a concise report that cites the tool outputs and highlights any missing information.

        {context}
        """
    ).strip()


def _extend_args(target: List[str], flag: str, value: Optional[str]) -> None:
    """Append CLI flag/value pairs if the value is provided."""
    if value is None:
        return
    target.extend([flag, value])


def _extend_flag(target: List[str], flag: str, enabled: bool) -> None:
    """Append CLI flag when the corresponding boolean is true."""
    if enabled:
        target.append(flag)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the evaluation runner."""
    parser = argparse.ArgumentParser(
        description="Run the DeepUnlearn evaluation agent with a structured prompt."
    )
    parser.add_argument(
        "--dataset",
        default=deepunlearn_evaluator.DEFAULT_DATASET,
        help="Dataset name associated with the checkpoints (default: %(default)s).",
    )
    parser.add_argument(
        "--model-arch",
        default=deepunlearn_evaluator.DEFAULT_MODEL_ARCH,
        help="Model architecture (forwarded as model_type) (default: %(default)s).",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        required=True,
        help="Directory containing retain/forget/val/test split files (e.g., data/splits/<dataset>/123).",
    )
    parser.add_argument(
        "--original-model",
        type=Path,
        required=True,
        help="Checkpoint path of the original (pre-unlearning) model.",
    )
    parser.add_argument(
        "--unlearned-model",
        type=Path,
        required=True,
        help="Checkpoint path of the unlearned model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=deepunlearn_evaluator.DEFAULT_BATCH_SIZE,
        help="Batch size forwarded to evaluation loaders (default: %(default)s).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=deepunlearn_evaluator.DEFAULT_RANDOM_STATE,
        help="Random state used during evaluation (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Device identifier for evaluation (default: "auto").',
    )
    parser.add_argument(
        "--split-index",
        type=int,
        default=None,
        help="Optional LIRA split index (requires --forget-index).",
    )
    parser.add_argument(
        "--forget-index",
        type=int,
        default=None,
        help="Optional LIRA forget index (requires --split-index).",
    )
    parser.add_argument(
        "--metric",
        dest="metrics",
        action="append",
        default=None,
        help="Specific metric to emphasise in the evaluation (repeatable).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=deepunlearn_evaluator.DEFAULT_MAX_TURNS,
        help="Maximum conversation turns with the agent (default: %(default)s).",
    )
    parser.add_argument(
        "--model-platform",
        default=None,
        help="Override the model platform (falls back to evaluator defaults).",
    )
    parser.add_argument(
        "--model-type",
        default=None,
        help="Override the LLM model type (falls back to evaluator defaults).",
    )
    parser.add_argument(
        "--model-url",
        default=None,
        help="Override the base URL for OpenAI-compatible endpoints.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Model API key; if omitted, environment variables are used.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature override for the model backend.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where evaluation transcripts are persisted.",
    )
    parser.add_argument(
        "--deepunlearn-dir",
        type=Path,
        default=None,
        help="Path to the DeepUnlearn repository (default handled by evaluator).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root passed into the toolkit (default handled by evaluator).",
    )
    parser.add_argument(
        "--toolkit-timeout",
        type=int,
        default=None,
        help="Optional timeout (seconds) for toolkit commands.",
    )
    parser.add_argument(
        "--toolkit-verbose",
        action="store_true",
        help="Enable verbose DeepUnlearn subprocess output.",
    )
    parser.add_argument(
        "--correlation-id",
        default=None,
        help="Correlation identifier forwarded to the evaluator.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Log level for the evaluator run (default handled by evaluator).",
    )
    parser.add_argument(
        "--console-format",
        choices=("json", "text"),
        default=None,
        help="Console log format when file logging is disabled.",
    )
    parser.add_argument(
        "--disable-file-logging",
        action="store_true",
        help="Disable file logging for this run (console output only).",
    )
    parser.add_argument(
        "--extra-arg",
        dest="extra_args",
        action="append",
        default=None,
        help="Additional raw arguments forwarded directly to the evaluator (repeatable).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for scripted automation of the evaluation agent."""
    args = _parse_args(argv)

    splits_dir = args.splits_dir.resolve()
    original_model = args.original_model.resolve()
    unlearned_model = args.unlearned_model.resolve()

    for path, label in [
        (splits_dir, "splits directory"),
        (original_model, "original model"),
        (unlearned_model, "unlearned model"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    if not splits_dir.is_dir():
        raise ValueError(f"Splits path is not a directory: {splits_dir}")

    if (args.split_index is None) ^ (args.forget_index is None):
        raise ValueError(
            "Both --split-index and --forget-index must be provided together for LIRA evaluation."
        )

    prompt = _render_evaluation_prompt(
        dataset=args.dataset,
        model_arch=args.model_arch,
        splits_dir=splits_dir,
        original_model=original_model,
        unlearned_model=unlearned_model,
        batch_size=args.batch_size,
        random_state=args.random_state,
        device=args.device,
        split_index=args.split_index,
        forget_index=args.forget_index,
        metrics=args.metrics,
    )

    evaluator_args: List[str] = [
        "--dataset",
        args.dataset,
        "--model-arch",
        args.model_arch,
        "--splits-dir",
        str(splits_dir),
        "--original-model",
        str(original_model),
        "--unlearned-model",
        str(unlearned_model),
        "--batch-size",
        str(args.batch_size),
        "--random-state",
        str(args.random_state),
        "--device",
        args.device,
        "--max-turns",
        str(args.max_turns),
        "--prompt",
        prompt,
    ]

    _extend_args(
        evaluator_args,
        "--split-index",
        str(args.split_index) if args.split_index is not None else None,
    )
    _extend_args(
        evaluator_args,
        "--forget-index",
        str(args.forget_index) if args.forget_index is not None else None,
    )

    _extend_args(evaluator_args, "--model-platform", args.model_platform)
    _extend_args(evaluator_args, "--model-type", args.model_type)
    _extend_args(evaluator_args, "--model-url", args.model_url)
    _extend_args(evaluator_args, "--api-key", args.api_key)
    _extend_args(
        evaluator_args,
        "--temperature",
        str(args.temperature) if args.temperature is not None else None,
    )
    _extend_args(
        evaluator_args,
        "--output-dir",
        str(args.output_dir.resolve()) if args.output_dir else None,
    )
    _extend_args(
        evaluator_args,
        "--deepunlearn-dir",
        str(args.deepunlearn_dir.resolve()) if args.deepunlearn_dir else None,
    )
    _extend_args(
        evaluator_args,
        "--project-root",
        str(args.project_root.resolve()) if args.project_root else None,
    )
    _extend_args(
        evaluator_args,
        "--toolkit-timeout",
        str(args.toolkit_timeout) if args.toolkit_timeout is not None else None,
    )
    _extend_args(evaluator_args, "--correlation-id", args.correlation_id)
    _extend_args(evaluator_args, "--log-level", args.log_level)
    _extend_args(evaluator_args, "--console-format", args.console_format)

    _extend_flag(evaluator_args, "--toolkit-verbose", args.toolkit_verbose)
    _extend_flag(evaluator_args, "--disable-file-logging", args.disable_file_logging)

    if args.extra_args:
        evaluator_args.extend(args.extra_args)

    return deepunlearn_evaluator.main(evaluator_args)


if __name__ == "__main__":
    raise SystemExit(main())
