#!/usr/bin/env python3
"""
Entry point for running the CAUST inner loop orchestrator.

This script wires together the TaskParser, TaskSpec assembly, and the
`InnerLoopOrchestrator` so that researchers can kick off (or resume) an inner
loop experiment from the CLI.  Typical usage::

    python aust/scripts/main.py \
        --task-type concept_erasure \
        --prompt "Attack Stable Diffusion 1.4 unlearned with concept Cat [with ESD]" \
        --unlearned-model-path data/unlearned_models/esd/stable-diffusion/Cat/esd-Cat-from-Cat-esdx-pipeline \
        --model-name "Stable Diffusion" \
        --model-version "1.4" \
        --unlearned-target "Cat" \
        --unlearning-method "ESD" \
        --max-iterations 2 \
        --max-debate-rounds 2 \
        [--base-model-path data/unlearned_models/esd/stable-diffusion/Cat/base_pipeline]

The script can also resume from a saved `loop_state.json` via `--resume-state`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence, TYPE_CHECKING
import agentops

from dotenv import load_dotenv

agentops.init(os.environ.get("AGENTOPS_API_KEY", ""))

# Ensure the project root is on the import path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aust.src.utils.logging_config import get_logger, setup_logging

if TYPE_CHECKING:  # pragma: no cover - import hints only
    from aust.src.agents.inner_loop_orchestrator import InnerLoopOrchestrator
    from aust.src.data_models.task_spec import TaskSpec

load_dotenv()
logger = get_logger(__name__)

DEFAULT_TASK_DESCRIPTION = "Evaluate vulnerabilities in machine unlearning"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_RAG_STORAGE = PROJECT_ROOT / "aust" / "rag_paper_db"
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "aust" / "configs"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Define and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run or resume the CAUST inner loop orchestrator."
    )

    spec_group = parser.add_argument_group("Task specification inputs")
    spec_group.add_argument(
        "--task-spec-json",
        type=Path,
        help="Path to an existing TaskSpec JSON file. Overrides individual spec arguments.",
    )
    spec_group.add_argument(
        "--task-type",
        default="concept_erasure",
        choices=["concept_erasure", "data_based_unlearning"],
        help="Task type supplied to TaskParser/TaskSpec (default: %(default)s).",
    )
    spec_group.add_argument(
        "--prompt",
        help="User prompt describing the vulnerability test scenario.",
    )
    spec_group.add_argument(
        "--prompt-file",
        type=Path,
        help="Optional file containing an additional prompt segment.",
    )
    spec_group.add_argument(
        "--base-model-path",
        type=Path,
        help="Path to the base/original model (pipeline directory or single-file checkpoint).",
    )
    spec_group.add_argument(
        "--unlearned-model-path",
        type=Path,
        help="Path to the unlearned/concept-erased model (pipeline directory or single-file checkpoint).",
    )
    spec_group.add_argument(
        "--model-name",
        help="Override for model_name in the TaskSpec.",
    )
    spec_group.add_argument(
        "--model-version",
        help="Override for model_version in the TaskSpec.",
    )
    spec_group.add_argument(
        "--unlearned-target",
        help="Override for unlearned_target in the TaskSpec.",
    )
    spec_group.add_argument(
        "--unlearning-method",
        help="Override for unlearning_method in the TaskSpec.",
    )
    spec_group.add_argument(
        "--skip-task-parser",
        action="store_true",
        help="Skip TaskParserAgent and rely solely on the provided overrides.",
    )
    spec_group.add_argument(
        "--task-parser-model",
        help="Optional override for the TaskParserAgent model name.",
    )

    loop_group = parser.add_argument_group("Inner loop configuration")
    loop_group.add_argument("--task-id", help="Optional explicit task identifier.")
    loop_group.add_argument(
        "--task-description",
        help=f"High-level task description (default: '{DEFAULT_TASK_DESCRIPTION}').",
    )
    loop_group.add_argument(
        "--max-iterations",
        type=int,
        default=2,
        help="Maximum number of inner loop iterations (default: %(default)s).",
    )
    loop_group.add_argument(
        "--disable-debate",
        action="store_true",
        help="Disable hypothesis/critic debate.",
    )
    loop_group.add_argument(
        "--quality-threshold",
        type=float,
        default=0.85,
        help="Quality score threshold to stop the debate early (default: %(default)s).",
    )
    loop_group.add_argument(
        "--max-debate-rounds",
        type=int,
        default=3,
        help="Maximum number of debate rounds per iteration (default: %(default)s).",
    )
    loop_group.add_argument(
        "--rag-top-k",
        type=int,
        default=3,
        help="Default retrieval depth hint for the query generator (default: %(default)s).",
    )
    loop_group.add_argument(
        "--generator-model",
        help="Override hypothesis generator model identifier.",
    )
    loop_group.add_argument(
        "--critic-model",
        help="Override critic model identifier.",
    )
    loop_group.add_argument(
        "--query-generator-model",
        help="Override query generator model identifier.",
    )
    loop_group.add_argument(
        "--query-max-queries",
        type=int,
        default=3,
        help="Maximum number of RAG queries per iteration (default: %(default)s).",
    )

    path_group = parser.add_argument_group("Paths and I/O")
    path_group.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Base directory for outputs (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    path_group.add_argument(
        "--rag-storage-path",
        type=Path,
        default=DEFAULT_RAG_STORAGE,
        help=f"Path to the paper RAG vector store (default: {DEFAULT_RAG_STORAGE}).",
    )
    path_group.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help=f"Configuration directory (default: {DEFAULT_CONFIG_DIR}).",
    )
    path_group.add_argument(
        "--resume-state",
        type=Path,
        help="Path to a saved loop_state.json to resume from.",
    )

    log_group = parser.add_argument_group("Logging")
    log_group.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level for the run (default: %(default)s).",
    )
    log_group.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help=f"Directory where CAUST log files will be stored (default: {DEFAULT_LOG_DIR}).",
    )
    log_group.add_argument(
        "--console-style",
        default="rich",
        choices=["rich", "json", "text"],
        help="Console logging style (default: %(default)s).",
    )
    log_group.add_argument(
        "--no-console-log",
        action="store_true",
        help="Disable console logging output.",
    )
    log_group.add_argument(
        "--no-file-log",
        action="store_true",
        help="Disable file logging output.",
    )
    log_group.add_argument(
        "--enable-agentops",
        action="store_true",
        help="Keep AgentOps instrumentation enabled (default: disable by unsetting AGENTOPS_API_KEY for this run).",
    )

    return parser.parse_args(argv)

def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    setup_logging(
        log_level=args.log_level,
        log_dir=args.log_dir,
        enable_console=not args.no_console_log,
        enable_file=not args.no_file_log,
        console_style=args.console_style,
    )

    _configure_agentops(args.enable_agentops)

    try:
        state = run_inner_loop(args)
    except Exception as exc:  # pragma: no cover - CLI error path
        logger.exception("Inner loop execution failed: %s", exc)
        return 1

    logger.info(
        "Inner loop complete. Final state saved to %s",
        state.output_dir / "loop_state.json",
    )
    logger.info("Attack trace written to %s", state.attack_trace_file)
    logger.info(
        "Exit condition: %s (vulnerability_found=%s, highest_confidence=%.2f)",
        state.exit_condition.value if state.exit_condition else "unknown",
        state.vulnerability_found,
        state.highest_vulnerability_confidence,
    )

    return 0


def run_inner_loop(args: argparse.Namespace):
    """
    Build or resume an InnerLoopOrchestrator and execute it.
    """
    from aust.src.agents.inner_loop_orchestrator import InnerLoopOrchestrator

    if args.resume_state:
        orchestrator = InnerLoopOrchestrator.resume_from_state(
            state_file=_assert_path(args.resume_state, "resume-state"),
            rag_storage_path=args.rag_storage_path,
            config_dir=args.config_dir,
            rag_top_k=args.rag_top_k,
            generator_model=args.generator_model,
            critic_model=args.critic_model,
            quality_threshold=args.quality_threshold,
            max_debate_iterations=args.max_debate_rounds,
            query_generator_model=args.query_generator_model,
            query_max_queries=args.query_max_queries,
        )
    else:
        task_spec = _load_task_spec(args)
        task_description = (
            args.task_description or task_spec.user_prompt or DEFAULT_TASK_DESCRIPTION
        )

        orchestrator = InnerLoopOrchestrator(
            task_id=args.task_id,
            task_type=task_spec.task_type,
            task_description=task_description,
            task_spec=task_spec,
            max_iterations=args.max_iterations,
            enable_debate=not args.disable_debate,
            output_dir=args.output_root,
            rag_storage_path=args.rag_storage_path,
            config_dir=args.config_dir,
            rag_top_k=args.rag_top_k,
            generator_model=args.generator_model,
            critic_model=args.critic_model,
            quality_threshold=args.quality_threshold,
            max_debate_iterations=args.max_debate_rounds,
            query_generator_model=args.query_generator_model,
            query_max_queries=args.query_max_queries,
        )

    final_state = orchestrator.run()
    return final_state


def _load_task_spec(args: argparse.Namespace) -> TaskSpec:
    """Load or assemble a TaskSpec based on CLI arguments."""
    from aust.src.data_models.task_spec import TaskSpec
    from aust.src.agents.task_parser_agent import TaskParserAgent

    if args.task_spec_json:
        path = _assert_path(args.task_spec_json, "task-spec-json")
        logger.info("Loading TaskSpec from %s", path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return TaskSpec(**data)

    prompt = _read_prompt(args)
    if not prompt:
        raise ValueError(
            "A prompt (via --prompt or --prompt-file) is required when --task-spec-json is not supplied."
        )

    base_model_path = None
    if args.base_model_path is not None:
        base_model_path = _assert_path(
            args.base_model_path, "--base-model-path"
        ).as_posix()
    unlearned_model_path = _assert_path(
        args.unlearned_model_path, "--unlearned-model-path"
    ).as_posix()

    parser_result = None
    if not args.skip_task_parser:
        logger.info("Running TaskParserAgent to extract structured fields")
        parser = TaskParserAgent(model_name=args.task_parser_model)
        parser_result = parser.parse_prompt(prompt, task_type=args.task_type)

    overrides = {
        "model_name": args.model_name,
        "model_version": args.model_version,
        "unlearned_target": args.unlearned_target,
        "unlearning_method": args.unlearning_method,
    }

    return TaskSpec.assemble(
        task_type=args.task_type,
        base_model_path=base_model_path,
        unlearned_model_path=unlearned_model_path,
        user_prompt=prompt,
        parser_result=parser_result,
        overrides=overrides,
    )


def _read_prompt(args: argparse.Namespace) -> str:
    """Combine prompt text from CLI and optional file."""
    segments: list[str] = []

    if args.prompt_file:
        path = _assert_path(args.prompt_file, "--prompt-file")
        file_text = path.read_text(encoding="utf-8").strip()
        if file_text:
            segments.append(file_text)

    if args.prompt:
        prompt_text = str(args.prompt).strip()
        if prompt_text:
            segments.append(prompt_text)

    return "\n\n".join(segment for segment in segments if segment)


def _assert_path(path: Optional[Path], flag_name: str) -> Path:
    """Ensure a path argument exists and points to a file."""
    if path is None:
        raise ValueError(f"{flag_name} is required.")
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"{flag_name} not found: {resolved}")
    return resolved


def _configure_agentops(enable_agentops: bool) -> None:
    """Disable AgentOps instrumentation unless explicitly enabled."""
    if enable_agentops:
        return

    removed_key = os.environ.pop("AGENTOPS_API_KEY", None)
    if removed_key is not None:
        logger.info("AgentOps disabled for this run (use --enable-agentops to keep it enabled).")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
