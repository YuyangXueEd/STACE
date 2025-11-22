#!/usr/bin/env python3
"""
Entry point for running the CAUST inner loop orchestrator.

This script wires together the TaskParser, TaskSpec assembly, and the
`InnerLoopOrchestrator` so that researchers can kick off (or resume) an inner
loop experiment from the CLI.  Typical usage::

    python aust/scripts/main.py \
        --task-type concept_erasure \
        --prompt "Attack Stable Diffusion 1.4 unlearned with Van Gogh" \
        --unlearned-model-path data/unlearned_models/esd/100/stable-diffusion/Van_Gogh/esd-Van_Gogh-from-Van_Gogh-esdx-pipeline \
        --model-name "Stable Diffusion" \
        --model-version "1.4" \
        --unlearned-target "Van Gogh" \
        --max-iterations 10 \
        --max-debate-rounds 2 \
        --skip-judge \
        --stop-on-vulnerability 
        


The script can also resume from a saved `loop_state.json` via `--resume-state`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence, TYPE_CHECKING
from uuid import uuid4

from dotenv import load_dotenv

# Ensure the project root is on the import path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aust.src.utils.logging_config import get_logger, setup_logging

from aust.src.agents.judge import JudgeAgent
from aust.src.agents.reporter import ReporterAgent

if TYPE_CHECKING:  # pragma: no cover - import hints only
    from aust.src.agents.inner_loop_orchestrator import InnerLoopOrchestrator
    from aust.src.data_models.loop_state import InnerLoopState
    from aust.src.data_models.report import AcademicReport
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
        "--target-type",
        help="Optional override for target_type in the TaskSpec (e.g., style, object, attribute, general).",
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
    loop_group.add_argument(
        "--outer-iterations",
        type=int,
        default=1,
        help="Number of outer loop cycles (inner loop → report → judge) to execute (default: %(default)s).",
    )
    loop_group.add_argument(
        "--stop-on-vulnerability",
        dest="stop_on_vulnerability",
        action="store_true",
        default=False,
        help="Stop the loop early when a high-confidence vulnerability is detected (default: disabled).",
    )
    loop_group.add_argument(
        "--no-stop-on-vulnerability",
        dest="stop_on_vulnerability",
        action="store_false",
        help="Disable early stopping when a vulnerability is detected (default).",
    )
    loop_group.add_argument(
        "--vulnerability-stop-threshold",
        type=float,
        default=0.9,
        help="Confidence threshold for early stopping when enabled (default: %(default)s).",
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

    judge_group = parser.add_argument_group("Judge evaluation")
    judge_group.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip judge workforce evaluation (default: run committee if configuration permits).",
    )
    judge_group.add_argument(
        "--judge-personas",
        type=Path,
        help="Optional path to judge persona YAML configuration (defaults to built-in personas).",
    )
    judge_group.add_argument(
        "--judge-model-config",
        default="judge",
        help="Model config name to use when initialising JudgeAgent (default: %(default)s).",
    )
    judge_group.add_argument(
        "--judge-output-dir",
        type=Path,
        help="Directory where judge evaluations will be written (default: <task_output>/judgments).",
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

    try:
        states = run_outer_loop(args)
    except Exception as exc:  # pragma: no cover - CLI error path
        logger.exception("Outer loop execution failed: %s", exc)
        return 1

    if not states:
        logger.warning("No inner loop executions were completed.")
        return 0

    final_state = states[-1]
    logger.info(
        "Outer loop complete. Latest inner loop state saved to %s",
        final_state.output_dir / "loop_state.json",
    )
    logger.info("Attack trace written to %s", final_state.attack_trace_file)
    logger.info(
        "Final exit condition: %s (vulnerability_found=%s, highest_confidence=%.2f)",
        final_state.exit_condition.value if final_state.exit_condition else "unknown",
        final_state.vulnerability_found,
        final_state.highest_vulnerability_confidence,
    )

    return 0


def run_inner_loop(
    args: argparse.Namespace,
    *,
    task_spec: Optional["TaskSpec"] = None,
    task_id_override: Optional[str] = None,
    output_root_override: Optional[Path] = None,
):
    """
    Build or resume an InnerLoopOrchestrator and execute it.
    """
    from aust.src.agents.inner_loop_orchestrator import InnerLoopOrchestrator

    output_root = Path(output_root_override) if output_root_override else Path(args.output_root)

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
            stop_on_vulnerability=args.stop_on_vulnerability,
            vulnerability_confidence_threshold=args.vulnerability_stop_threshold,
        )
    else:
        spec = task_spec or _load_task_spec(args)
        task_description = (
            args.task_description or spec.user_prompt or DEFAULT_TASK_DESCRIPTION
        )
        task_id_value = task_id_override or args.task_id

        orchestrator = InnerLoopOrchestrator(
            task_id=task_id_value,
            task_type=spec.task_type,
            task_description=task_description,
            task_spec=spec,
            max_iterations=args.max_iterations,
            enable_debate=not args.disable_debate,
            output_dir=output_root,
            rag_storage_path=args.rag_storage_path,
            config_dir=args.config_dir,
            rag_top_k=args.rag_top_k,
            generator_model=args.generator_model,
            critic_model=args.critic_model,
            quality_threshold=args.quality_threshold,
            max_debate_iterations=args.max_debate_rounds,
            query_generator_model=args.query_generator_model,
            query_max_queries=args.query_max_queries,
            stop_on_vulnerability=args.stop_on_vulnerability,
            vulnerability_confidence_threshold=args.vulnerability_stop_threshold,
        )

    final_state = orchestrator.run()
    return final_state


def run_outer_loop(args: argparse.Namespace) -> list["InnerLoopState"]:
    """
    Execute the outer loop workflow (inner loop → report → judge) for one or more iterations.
    """
    states: list["InnerLoopState"] = []
    outer_iterations = max(1, getattr(args, "outer_iterations", 1))

    # Resume mode bypasses outer-loop iteration handling
    if args.resume_state:
        state = run_inner_loop(args)
        states.append(state)
        _process_inner_loop_outputs(args, state)
        return states

    # Build TaskSpec once and reuse across outer iterations
    task_spec = _load_task_spec(args)
    if hasattr(task_spec, "metadata"):
        pending_judge_feedback = task_spec.metadata.get("outer_loop", {}).get("judge_feedback")
    else:
        pending_judge_feedback = None


    if hasattr(task_spec, "metadata"):
        outer_meta = task_spec.metadata.setdefault("outer_loop", {})
    elif isinstance(task_spec, dict):
        metadata = task_spec.setdefault("metadata", {})
        outer_meta = metadata.setdefault("outer_loop", {})
    else:
        outer_meta = None

    pending_judge_feedback = (
        outer_meta.get("judge_feedback") if outer_meta else None
    )

    for outer_index in range(outer_iterations):
        if outer_meta is not None:
            if pending_judge_feedback:
                outer_meta["judge_feedback"] = pending_judge_feedback
            else:
                outer_meta.pop("judge_feedback", None)

        if outer_index == 0:
            task_id_override = args.task_id
        else:
            base_id = states[0].task_id if states else args.task_id
            if not base_id:
                base_id = f"task_{uuid4().hex[:8]}"
            task_id_override = f"{base_id}_outer{outer_index + 1:02d}"

        logger.info(
            "===== Outer iteration %d/%d (task_id=%s) =====",
            outer_index + 1,
            outer_iterations,
            task_id_override or "auto",
        )

        state = run_inner_loop(
            args,
            task_spec=task_spec,
            task_id_override=task_id_override,
            output_root_override=args.output_root,
        )
        states.append(state)

        report, report_path = _process_inner_loop_outputs(args, state)

        summary_text: Optional[str] = None
        if report_path:
            summary_path = state.output_dir / "judgments" / f"summary_{state.task_id}.md"
            if summary_path.exists():
                summary_text = summary_path.read_text(encoding="utf-8").strip()

        if summary_text:
            pending_judge_feedback = summary_text
        else:
            pending_judge_feedback = None

    return states


def generate_report_for_state(state: "InnerLoopState") -> Optional[tuple["AcademicReport", Path]]:
    """Generate and save a report for the completed inner loop state.

    Returns a tuple of (report, path) on success, or None on failure.
    """
    from aust.src.data_models.loop_state import InnerLoopState as _InnerLoopState

    if not isinstance(state, _InnerLoopState):
        logger.warning("generate_report_for_state called with unexpected state type: %s", type(state))
        return None

    reporter = ReporterAgent(output_dir=state.output_dir)

    attack_trace_dir = state.output_dir / "attack_traces"
    attack_trace_json_path = attack_trace_dir / f"trace_{state.task_id}.json"

    attack_trace_md_path = state.attack_trace_file
    if not attack_trace_md_path:
        attack_trace_md_path = attack_trace_dir / f"trace_{state.task_id}.md"

    try:
        report = reporter.generate_report(
            inner_loop_state=state,
            attack_trace_json_path=attack_trace_json_path,
            attack_trace_md_path=attack_trace_md_path,
            retrieved_papers=None,
        )
        report_path = reporter.save_report(report, state.task_id)
        return report, report_path
    except Exception as exc:  # pragma: no cover - safeguard around final reporting
        logger.error("Failed to generate report: %s", exc, exc_info=True)
        return None


def _process_inner_loop_outputs(
    args: argparse.Namespace,
    state: "InnerLoopState",
) -> tuple[Optional["AcademicReport"], Optional[Path]]:
    """
    Generate report and optionally run judge evaluation for a completed inner loop state.
    """
    report_result = generate_report_for_state(state)
    if not report_result:
        logger.info("Report generation skipped or failed; see logs above for details")
        return None, None

    report, report_path = report_result
    logger.info("Report generated at %s", report_path)

    if args.skip_judge:
        logger.info("Judge evaluation skipped via --skip-judge flag")
    else:
        _run_judge_committee(args, state, report, report_path)

    reporter = ReporterAgent(output_dir=state.output_dir)
    attack_trace_dir = state.output_dir / "attack_traces"

    # AC2: Persist successful attacks as reusable seed templates
    saved_templates = reporter.save_successful_templates_from_traces(
        state.task_id,
        traces_dir=attack_trace_dir,
        task_context=getattr(state, "task_spec", None),
    )
    if saved_templates:
        logger.info(
            "Successful attacks captured as templates: %s",
            ", ".join(str(path) for path in saved_templates),
        )

    # AC4: Generate long-term report via LongTermMemoryAgent
    try:
        from aust.src.agents.long_term_memory_agent import LongTermMemoryAgent

        memory_agent = LongTermMemoryAgent()
        long_report_path = memory_agent.generate_long_term_report(
            task_id=state.task_id,
            traces_dir=attack_trace_dir,
            output_dir=state.output_dir,
        )
        if long_report_path:
            logger.info("Long-term memory report saved to %s", long_report_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to generate long-term memory report: %s", exc)

    return report, report_path


def _run_judge_committee(
    args: argparse.Namespace,
    state: "InnerLoopState",
    report: "AcademicReport",
    report_path: Path | None,
) -> None:
    """Run the judge workforce and log aggregate results."""
    try:
        from aust.src.data_models.loop_state import InnerLoopState as _InnerLoopState
    except ImportError:  # pragma: no cover - defensive
        logger.error("Unable to import InnerLoopState; skipping judge evaluation")
        return

    if not isinstance(state, _InnerLoopState):
        logger.warning("run_judge_committee received unexpected state type: %s", type(state))
        return

    persona_path: Path | None = None
    if args.judge_personas:
        try:
            persona_path = _assert_path(args.judge_personas, "--judge-personas")
        except (ValueError, FileNotFoundError) as exc:
            logger.error("Invalid judge persona configuration: %s", exc)
            return

    judge_output_dir = Path(args.judge_output_dir) if args.judge_output_dir else state.output_dir / "judgments"

    attack_trace_json_path = state.output_dir / "attack_traces" / f"trace_{state.task_id}.json"
    attack_trace_payload: Path | None = attack_trace_json_path if attack_trace_json_path.exists() else None

    experiment_summary = _build_experiment_summary(state)
    if report_path:
        experiment_summary.setdefault("report_path", str(report_path))

    judge_kwargs: dict[str, Any] = {
        "output_dir": judge_output_dir,
        "model_config_name": args.judge_model_config,
    }
    if persona_path is not None:
        judge_kwargs["persona_config_path"] = persona_path

    try:
        judge_agent = JudgeAgent(**judge_kwargs)
    except EnvironmentError as exc:
        logger.warning("Judge evaluation skipped: %s", exc)
        return
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to initialise JudgeAgent: %s", exc, exc_info=True)
        return

    try:
        aggregate = judge_agent.run_committee(
            report=report,
            attack_trace=attack_trace_payload,
            experiment_results=experiment_summary,
            run_id=state.task_id,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Judge committee execution failed: %s", exc, exc_info=True)
        return

    summary_path = judge_output_dir / f"summary_{state.task_id}.md"
    logger.info(
        "Judge committee completed with %d persona evaluations. Summary written to %s",
        len(aggregate.persona_evaluations),
        summary_path,
    )
    if aggregate.average_overall_rating is not None:
        logger.info(
            "Judge average overall rating: %.2f (0-5 scale)",
            aggregate.average_overall_rating,
        )
    if aggregate.dimension_averages:
        dimension_line = ", ".join(
            f"{dimension}: {score:.2f}"
            for dimension, score in sorted(aggregate.dimension_averages.items())
        )
        logger.debug("Judge dimension averages: %s", dimension_line)


def _build_experiment_summary(state: "InnerLoopState") -> dict[str, Any]:
    """Construct a lightweight JSON-serialisable summary of experiment outcomes."""
    iterations_summary: list[dict[str, Any]] = []

    for iteration in state.iterations:
        hypothesis = getattr(iteration, "hypothesis", None)
        iterations_summary.append(
            {
                "iteration_number": iteration.iteration_number,
                "attack_type": getattr(hypothesis, "attack_type", None),
                "hypothesis_description": getattr(hypothesis, "description", None),
                "experiment_executed": iteration.experiment_executed,
                "vulnerability_detected": iteration.vulnerability_detected,
                "vulnerability_confidence": iteration.vulnerability_confidence,
                "key_learning": iteration.key_learning,
                "started_at": iteration.started_at.isoformat() if iteration.started_at else None,
                "completed_at": iteration.completed_at.isoformat() if iteration.completed_at else None,
            }
        )

    return {
        "task_id": state.task_id,
        "task_type": state.task_type,
        "task_description": state.task_description,
        "vulnerability_found": state.vulnerability_found,
        "highest_confidence": state.highest_vulnerability_confidence,
        "exit_condition": state.exit_condition.value if state.exit_condition else None,
        "total_iterations": len(state.iterations),
        "started_at": state.started_at.isoformat() if state.started_at else None,
        "completed_at": state.completed_at.isoformat() if state.completed_at else None,
        "duration_seconds": state.total_duration_seconds,
        "iterations": iterations_summary,
    }


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
        "target_type": args.target_type,
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


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
