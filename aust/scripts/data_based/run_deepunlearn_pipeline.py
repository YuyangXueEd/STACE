"""Utility script to drive the DeepUnlearn orchestration agent.

This helper crafts a prompt that nudges the agent from dataset preparation
to executing a single unlearning method, producing a ready-to-use checkpoint.
It forwards configuration flags to `deepunlearn_orchestrator` and relies on
the registered toolkit functions for execution.

Usage example:
python -m aust.scripts.run_deepunlearn_pipeline \
  --toolkit-verbose \
  --train-override unlearner.cfg.num_epochs=20 \
  --unlearn-override unlearner.cfg.num_epochs=5 \
  --materialize-dir data
"""

from __future__ import annotations

import argparse
from textwrap import dedent
from typing import List, Optional, Sequence

from aust.src.agents import deepunlearn_orchestrator


def _normalise_num_classes(raw: Sequence[int] | None) -> List[int]:
    """Normalise CLI num_classes values while preserving orchestrator defaults."""
    if not raw:
        return list(deepunlearn_orchestrator.DEFAULT_NUM_CLASSES)
    return list(raw)


def _render_pipeline_prompt(
    dataset: str,
    model_arch: str,
    num_classes: Sequence[int],
    model_seeds: int,
    *,
    train_method: str,
    train_model_seed: int,
    train_random_state: Optional[int],
    train_overrides: Optional[Sequence[str]],
    train_save_path: Optional[str],
    unlearn_method: str,
    unlearn_model_seed: int,
    unlearn_random_state: Optional[int],
    unlearn_overrides: Optional[Sequence[str]],
    unlearn_save_path: Optional[str],
    data_dir: str,
) -> str:
    """Create the agent prompt that drives the DeepUnlearn pipeline."""
    num_classes_literal = ", ".join(str(value) for value in num_classes)
    train_random_clause = (
        f", random_state={train_random_state}"
        if train_random_state is not None
        else ""
    )
    train_save_clause = (
        f', save_path="{train_save_path}"'
        if train_save_path
        else ""
    )
    train_overrides_clause = ""
    if train_overrides:
        train_overrides_literal = ", ".join(f'"{item}"' for item in train_overrides)
        train_overrides_clause = f", overrides=[{train_overrides_literal}]"

    unlearn_random_clause = (
        f", random_state={unlearn_random_state}"
        if unlearn_random_state is not None
        else ""
    )
    unlearn_save_clause = (
        f', save_path="{unlearn_save_path}"' if unlearn_save_path else ""
    )
    unlearn_overrides_clause = ""
    if unlearn_overrides:
        unlearn_overrides_literal = ", ".join(
            f'"{item}"' for item in unlearn_overrides
        )
        unlearn_overrides_clause = f", overrides=[{unlearn_overrides_literal}]"

    train_call = (
        f'train_model(dataset="{dataset}", model="{model_arch}", method="{train_method}", '
        f"model_seed={train_model_seed}"
        f"{train_random_clause}{train_save_clause}{train_overrides_clause})"
    )

    unlearn_call = (
        f'run_unlearning_method(method="{unlearn_method}", dataset="{dataset}", '
        f'model="{model_arch}", model_seed={unlearn_model_seed}'
        f"{unlearn_random_clause}{unlearn_save_clause}{unlearn_overrides_clause})"
    )

    return dedent(
        f"""\
        Execute the DeepUnlearn pipeline for dataset "{dataset}" and architecture "{model_arch}" to produce a prepared unlearned model.

        Follow this exact sequence:
        1. Call `prepare_data_splits(dataset="{dataset}")`.
        2. Call `generate_initial_models(model="{model_arch}", num_classes=[{num_classes_literal}], model_seeds={model_seeds})`.
        3. Call `link_model_initializations()`.
        4. Call `{train_call}`.
        5. Call `{unlearn_call}`.
        6. Call `materialize_artifacts(dataset="{dataset}", model="{model_arch}", train_method="{train_method}", unlearn_method="{unlearn_method}", model_seed={unlearn_model_seed}, data_dir="{data_dir}")`.
        7. Inspect the tool responses, confirm where the trained, unlearned, and copied artifacts were saved (look for fields such as `artifact_path`, `data_dir`, or `metadata_file`), highlight those locations, and summarise the work completed. Mention any remaining manual follow-ups.

        After each tool call, report the key outputs (paths, counts, or errors) in no more than two concise sentences.
        Stop immediately if any step fails and explain what needs to be resolved. Finish with a short summary once all steps succeed.
        """
    )


def _extend_args(target: List[str], flag: str, value: str | None) -> None:
    """Append CLI flag/value pairs if the value is provided."""
    if value is None:
        return
    target.extend([flag, value])


def _extend_flag(target: List[str], flag: str, enabled: bool) -> None:
    """Append CLI flag when the corresponding boolean is true."""
    if enabled:
        target.append(flag)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the DeepUnlearn orchestration agent through the full pipeline."
    )
    parser.add_argument(
        "--dataset",
        default=deepunlearn_orchestrator.DEFAULT_DATASET,
        help="Dataset name to process (default: %(default)s).",
    )
    parser.add_argument(
        "--model-arch",
        default=deepunlearn_orchestrator.DEFAULT_MODEL_ARCH,
        help="Model architecture for initialisation (default: %(default)s).",
    )
    parser.add_argument(
        "--num-classes",
        nargs="+",
        type=int,
        default=None,
        help="List of class counts for model initialisation.",
    )
    parser.add_argument(
        "--model-seeds",
        type=int,
        default=deepunlearn_orchestrator.DEFAULT_MODEL_SEEDS,
        help="Number of random seeds for initial model generation (default: %(default)s).",
    )
    parser.add_argument(
        "--train-method",
        default="original",
        help="Training method to execute prior to unlearning (default: %(default)s).",
    )
    parser.add_argument(
        "--train-model-seed",
        type=int,
        default=0,
        help="Model seed to target for the training run (default: %(default)s).",
    )
    parser.add_argument(
        "--train-random-state",
        type=int,
        default=None,
        help="Optional random state forwarded to the training command.",
    )
    parser.add_argument(
        "--train-save-path",
        default=None,
        help="Optional relative save_path override for the training output.",
    )
    parser.add_argument(
        "--train-override",
        dest="train_overrides",
        action="append",
        default=None,
        help="Additional Hydra override forwarded to the training command (repeatable).",
    )
    parser.add_argument(
        "--unlearn-method",
        default="finetune",
        help="Unlearning method to execute after training (default: %(default)s).",
    )
    parser.add_argument(
        "--unlearn-model-seed",
        type=int,
        default=0,
        help="Model seed to target for the unlearning run (default: %(default)s).",
    )
    parser.add_argument(
        "--unlearn-random-state",
        type=int,
        default=None,
        help="Optional random state forwarded to the unlearning command.",
    )
    parser.add_argument(
        "--unlearn-save-path",
        default=None,
        help="Optional relative save_path override for the unlearning output.",
    )
    parser.add_argument(
        "--unlearn-override",
        dest="unlearn_overrides",
        action="append",
        default=None,
        help="Additional Hydra override forwarded to the unlearning command (repeatable).",
    )
    parser.add_argument(
        "--materialize-dir",
        default="data",
        help="Destination directory for copied models and splits (default: %(default)s).",
    )
    parser.add_argument(
        "--model-platform",
        default=None,
        help="Optional override of the model platform (falls back to orchestrator defaults).",
    )
    parser.add_argument(
        "--model-type",
        default=None,
        help="Optional override of the model type (falls back to orchestrator defaults).",
    )
    parser.add_argument(
        "--model-url",
        default=None,
        help="Optional override for the model base URL when using OpenAI-compatible APIs.",
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
        "--max-turns",
        type=int,
        default=deepunlearn_orchestrator.DEFAULT_MAX_TURNS,
        help="Maximum conversation turns with the agent (default: %(default)s).",
    )
    parser.add_argument(
        "--deepunlearn-dir",
        default=None,
        help="Path to the DeepUnlearn repository (default handled by orchestrator).",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Project root path passed into the toolkit (default handled by orchestrator).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where orchestration transcripts are persisted.",
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
        "--log-level",
        default=None,
        help="Log level for the orchestrator run (default handled by orchestrator).",
    )
    parser.add_argument(
        "--disable-file-logging",
        action="store_true",
        help="Disable file logging for this run (console output only).",
    )
    parser.add_argument(
        "--console-format",
        choices=("json", "text"),
        default=None,
        help="Console log format when file logging is disabled.",
    )
    parser.add_argument(
        "--extra-arg",
        dest="extra_args",
        action="append",
        default=None,
        help="Additional raw arguments forwarded directly to the orchestrator (repeatable).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    train_overrides = [
        override.strip()
        for override in (args.train_overrides or [])
        if isinstance(override, str) and override.strip()
    ]
    unlearn_overrides = [
        override.strip()
        for override in (args.unlearn_overrides or [])
        if isinstance(override, str) and override.strip()
    ]

    default_train_save = f"unlearn/model_{args.model_arch}-unlearner_{args.train_method}"
    train_save_path = args.train_save_path or default_train_save
    materialize_dir = args.materialize_dir

    init_dir_override_prefix = "unlearner.cfg.model_initializations_dir="
    expected_init_dir = (
        f"unlearn/model_{args.model_arch}-unlearner_{args.train_method}/{args.train_method}"
    )
    if not any(override.startswith(init_dir_override_prefix) for override in unlearn_overrides):
        unlearn_overrides.append(f"{init_dir_override_prefix}{expected_init_dir}")

    num_classes = _normalise_num_classes(args.num_classes)

    pipeline_prompt = _render_pipeline_prompt(
        dataset=args.dataset,
        model_arch=args.model_arch,
        num_classes=num_classes,
        model_seeds=args.model_seeds,
        train_method=args.train_method,
        train_model_seed=args.train_model_seed,
        train_random_state=args.train_random_state,
        train_overrides=train_overrides,
        train_save_path=train_save_path,
        unlearn_method=args.unlearn_method,
        unlearn_model_seed=args.unlearn_model_seed,
        unlearn_random_state=args.unlearn_random_state,
        unlearn_overrides=unlearn_overrides,
        unlearn_save_path=args.unlearn_save_path,
        data_dir=materialize_dir,
    )

    orchestrator_args: List[str] = [
        "--dataset",
        args.dataset,
        "--model-arch",
        args.model_arch,
        "--model-seeds",
        str(args.model_seeds),
        "--train-method",
        args.train_method,
        "--train-model-seed",
        str(args.train_model_seed),
        "--train-save-path",
        train_save_path,
        "--data-root",
        materialize_dir,
        "--unlearn-method",
        args.unlearn_method,
        "--unlearn-model-seed",
        str(args.unlearn_model_seed),
        "--max-turns",
        str(args.max_turns),
    ]

    if args.num_classes:
        num_literal = ",".join(str(value) for value in num_classes)
        orchestrator_args.extend(["--num-classes", num_literal])

    _extend_args(orchestrator_args, "--model-platform", args.model_platform)
    _extend_args(orchestrator_args, "--model-type", args.model_type)
    _extend_args(orchestrator_args, "--model-url", args.model_url)
    _extend_args(orchestrator_args, "--api-key", args.api_key)
    _extend_args(
        orchestrator_args,
        "--temperature",
        str(args.temperature) if args.temperature is not None else None,
    )
    _extend_args(orchestrator_args, "--deepunlearn-dir", args.deepunlearn_dir)
    _extend_args(orchestrator_args, "--project-root", args.project_root)
    _extend_args(orchestrator_args, "--output-dir", args.output_dir)
    _extend_args(
        orchestrator_args,
        "--toolkit-timeout",
        str(args.toolkit_timeout) if args.toolkit_timeout is not None else None,
    )
    _extend_args(orchestrator_args, "--log-level", args.log_level)
    _extend_args(orchestrator_args, "--console-format", args.console_format)
    _extend_flag(orchestrator_args, "--toolkit-verbose", args.toolkit_verbose)
    _extend_flag(orchestrator_args, "--disable-file-logging", args.disable_file_logging)

    _extend_args(
        orchestrator_args,
        "--train-random-state",
        str(args.train_random_state) if args.train_random_state is not None else None,
    )
    _extend_args(
        orchestrator_args,
        "--unlearn-random-state",
        str(args.unlearn_random_state) if args.unlearn_random_state is not None else None,
    )
    _extend_args(orchestrator_args, "--unlearn-save-path", args.unlearn_save_path)
    for override in train_overrides:
        orchestrator_args.extend(["--train-override", override])
    for override in unlearn_overrides:
        orchestrator_args.extend(["--unlearn-override", override])

    orchestrator_args.extend(["--prompt", pipeline_prompt])

    if args.extra_args:
        orchestrator_args.extend(args.extra_args)

    return deepunlearn_orchestrator.main(orchestrator_args)


if __name__ == "__main__":
    raise SystemExit(main())
