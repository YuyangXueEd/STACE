"""DeepUnlearn orchestration agent using CAMEL-AI toolkits.

This module wires the DeepUnlearn pipeline toolkit into a CAMEL ChatAgent so
that pipeline operations can be triggered via natural language prompts. The
agent executes MCP-style function calls exposed by ``DeepUnlearnToolkit`` and
persists the interaction (including tool call inputs/outputs) to ``data/`` for
subsequent analysis.

Typical usage from the repository root::

    python -m aust.src.agents.deepunlearn_orchestrator

The script requires an inference endpoint that supports tool calling. By
default it targets OpenRouter; set ``OPENROUTER_API_KEY`` or override the
platform/model via the CLI or environment variables documented below.

If no prompt is supplied, the agent automatically executes DeepUnlearn
pipeline steps 1–6 for the CIFAR10 dataset using sensible defaults (resnet18,
two seeds, standard hyperparameter spec) and summarises each step concisely.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, Iterable, List, Optional, Sequence

from dotenv import load_dotenv

from camel.agents import ChatAgent
from camel.logger import get_logger as camel_get_logger
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.types.agents import ToolCallingRecord

from aust.src.logging_config import (
    get_logger,
    set_correlation_id,
    setup_logging,
)
from aust.src.toolkits import DeepUnlearnToolkit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_MESSAGE = dedent(
    """\
    You are the DeepUnlearn Orchestrator for CAUST.

    Manage the DeepUnlearn benchmark pipeline by calling the registered MCP
    functions when appropriate. Follow this workflow:
    1. Confirm the objective and outline the steps you will execute.
    2. Call the DeepUnlearn functions with explicit arguments to complete each
       step. Execute one function per tool call.
    3. After every tool call, interpret the JSON response and explain the key
       outputs, paying attention to file paths and counts.
    4. When objectives are complete, summarize the work performed and the
       resulting artifacts. Mention any follow-up actions needed.

    Never fabricate results—only report details that were returned by the
    tools or provided by the user.
    """
)

DEFAULT_MODEL_PLATFORM_ENV = "DEEPUNLEARN_AGENT_MODEL_PLATFORM"
DEFAULT_MODEL_TYPE_ENV = "DEEPUNLEARN_AGENT_MODEL_TYPE"
DEFAULT_API_KEY_ENV = "DEEPUNLEARN_AGENT_API_KEY"
DEFAULT_MODEL_URL_ENV = "DEEPUNLEARN_AGENT_MODEL_URL"
DEFAULT_TEMPERATURE_ENV = "DEEPUNLEARN_AGENT_TEMPERATURE"
DEFAULT_CORRELATION_ENV = "CAUST_CORRELATION_ID"

DEFAULT_MODEL_PLATFORM = ModelPlatformType.OPENAI_COMPATIBLE_MODEL.value
DEFAULT_MODEL_TYPE = "openai/gpt-5-nano"
DEFAULT_MODEL_URL = "https://openrouter.ai/api/v1"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TURNS = 1
DEFAULT_DATASET = "cifar10"
DEFAULT_MODEL_ARCH = "resnet18"
DEFAULT_NUM_CLASSES = [10]
DEFAULT_MODEL_SEEDS = 1
DEFAULT_TRAIN_METHOD = "original"
DEFAULT_UNLEARN_METHOD = "finetune"
DEFAULT_TRAIN_MODEL_SEED = 0
DEFAULT_UNLEARN_MODEL_SEED = 0
DEFAULT_CONSOLE_FORMAT = "text"


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run the DeepUnlearn orchestration agent."
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="User prompt(s) to feed into the agent (can be specified multiple times).",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Path to a text file containing an additional prompt.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Dataset to target when auto-running pipeline steps (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--model-arch",
        default=DEFAULT_MODEL_ARCH,
        help=f"Model architecture to initialise during step 2 (default: {DEFAULT_MODEL_ARCH}).",
    )
    parser.add_argument(
        "--num-classes",
        default=None,
        help=(
            "Comma-separated list of class counts for model initialisations "
            f"(default: {DEFAULT_NUM_CLASSES})."
        ),
    )
    parser.add_argument(
        "--model-seeds",
        type=int,
        default=DEFAULT_MODEL_SEEDS,
        help=f"Number of seeds for model initialisation step (default: {DEFAULT_MODEL_SEEDS}).",
    )
    parser.add_argument(
        "--train-method",
        default=DEFAULT_TRAIN_METHOD,
        help="Baseline training method to run before unlearning (default: %(default)s).",
    )
    parser.add_argument(
        "--train-model-seed",
        type=int,
        default=DEFAULT_TRAIN_MODEL_SEED,
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
        default=DEFAULT_UNLEARN_METHOD,
        help="Unlearning method to execute after training (default: %(default)s).",
    )
    parser.add_argument(
        "--unlearn-model-seed",
        type=int,
        default=DEFAULT_UNLEARN_MODEL_SEED,
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
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Destination directory for copied models and splits.",
    )
    parser.add_argument(
        "--model-platform",
        default=os.getenv(DEFAULT_MODEL_PLATFORM_ENV, DEFAULT_MODEL_PLATFORM),
        help=(
            "Model platform identifier understood by CAMEL "
            "(default: OpenAI-compatible via OpenRouter)."
        ),
    )
    parser.add_argument(
        "--model-type",
        default=os.getenv(DEFAULT_MODEL_TYPE_ENV, DEFAULT_MODEL_TYPE),
        help="Model type to request from the selected platform.",
    )
    parser.add_argument(
        "--model-url",
        default=os.getenv(DEFAULT_MODEL_URL_ENV, DEFAULT_MODEL_URL),
        help="Base URL for OpenAI-compatible endpoints (if required).",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv(DEFAULT_API_KEY_ENV),
        help="API key for the selected model platform.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv(DEFAULT_TEMPERATURE_ENV, DEFAULT_TEMPERATURE)),
        help="Sampling temperature for the model (default: 0.2).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=DEFAULT_MAX_TURNS,
        help="Maximum number of prompts to send to the agent (default: 1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory where orchestration results will be saved.",
    )
    parser.add_argument(
        "--deepunlearn-dir",
        type=Path,
        default=Path("external/DeepUnlearn"),
        help="Path to the DeepUnlearn repository.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root used when instantiating the toolkit.",
    )
    parser.add_argument(
        "--toolkit-timeout",
        type=int,
        default=None,
        help="Optional timeout (in seconds) for toolkit commands.",
    )
    parser.add_argument(
        "--toolkit-verbose",
        action="store_true",
        help="Enable verbose output from toolkit subprocesses.",
    )
    parser.add_argument(
        "--correlation-id",
        default=os.getenv(DEFAULT_CORRELATION_ENV),
        help="Optional correlation identifier for logging context.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level for the orchestrator (default: INFO).",
    )
    parser.add_argument(
        "--console-format",
        choices=("json", "text"),
        default=DEFAULT_CONSOLE_FORMAT,
        help=(
            "Console log format (default: text). "
            "JSON output remains available in the log file if enabled."
        ),
    )
    parser.add_argument(
        "--disable-file-logging",
        action="store_true",
        help="Disable writing orchestrator logs to disk (console only).",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _parse_num_classes_arg(raw_value: Optional[str]) -> List[int]:
    """Normalise the --num-classes argument into a list of integers."""
    if not raw_value:
        return list(DEFAULT_NUM_CLASSES)

    try:
        values = [
            int(item.strip())
            for item in raw_value.split(",")
            if item.strip()
        ]
    except ValueError as exc:
        raise ValueError(
            "--num-classes must be a comma-separated list of integers."
        ) from exc

    if not values:
        raise ValueError(
            "At least one value must be provided to --num-classes."
        )

    return values


def build_default_prompt(args: argparse.Namespace, num_classes: List[int]) -> str:
    """Construct the default prompt covering steps 1-6."""
    num_classes_literal = ", ".join(str(val) for val in num_classes)
    data_root = str(args.data_root)

    def fmt_overrides(values: Optional[Sequence[str]]) -> str:
        if not values:
            return ""
        literal = ", ".join(f'"{item}"' for item in values)
        return f", overrides=[{literal}]"

    train_random_clause = (
        f", random_state={args.train_random_state}"
        if args.train_random_state is not None
        else ""
    )
    train_save_clause = (
        f', save_path="{args.train_save_path}"'
        if args.train_save_path
        else ""
    )
    train_overrides_clause = fmt_overrides(args.train_overrides)

    unlearn_random_clause = (
        f", random_state={args.unlearn_random_state}"
        if args.unlearn_random_state is not None
        else ""
    )
    unlearn_save_clause = (
        f', save_path="{args.unlearn_save_path}"'
        if args.unlearn_save_path
        else ""
    )
    unlearn_overrides_clause = fmt_overrides(args.unlearn_overrides)

    train_call = (
        f'train_model(dataset="{args.dataset}", model="{args.model_arch}", '
        f'method="{args.train_method}", model_seed={args.train_model_seed}'
        f"{train_random_clause}{train_save_clause}{train_overrides_clause})"
    )
    unlearn_call = (
        f'run_unlearning_method(method="{args.unlearn_method}", dataset="{args.dataset}", '
        f'model="{args.model_arch}", model_seed={args.unlearn_model_seed}'
        f"{unlearn_random_clause}{unlearn_save_clause}{unlearn_overrides_clause})"
    )
    materialize_call = (
        f'materialize_artifacts(dataset="{args.dataset}", model="{args.model_arch}", '
        f'train_method="{args.train_method}", unlearn_method="{args.unlearn_method}", '
        f'model_seed={args.unlearn_model_seed}, data_dir="{data_root}")'
    )

    return dedent(
        f"""\
        Execute DeepUnlearn pipeline steps 1 through 6 in order for the dataset "{args.dataset}".

        Use this plan:
        - Step 1: Call `prepare_data_splits(dataset="{args.dataset}")`.
        - Step 2: Call `generate_initial_models(model="{args.model_arch}", num_classes=[{num_classes_literal}], model_seeds={args.model_seeds})`.
        - Step 3: Call `link_model_initializations()`.
        - Step 4: Call `{train_call}`.
        - Step 5: Call `{unlearn_call}`.
        - Step 6: Call `{materialize_call}`.
        - Step 7: Inspect tool responses, confirm copied artifact locations, and summarise completed work with any remaining manual follow ups.

        After each tool call, report the key outputs, especially artifact paths or counts, using no more than two concise sentences.
        If any tool fails, explain the failure, stop further steps, and summarise the completed work.
        """
    )


def ensure_prompts(
    args: argparse.Namespace,
    auto_num_classes: Optional[List[int]] = None,
) -> List[str]:
    """Derive the list of prompts from CLI arguments and defaults."""
    prompts: List[str] = []

    if args.prompts:
        prompts.extend(args.prompts)

    if args.prompt_file:
        prompts.append(args.prompt_file.read_text(encoding="utf-8"))

    if not prompts:
        if auto_num_classes is not None:
            num_classes = auto_num_classes
        else:
            try:
                num_classes = _parse_num_classes_arg(args.num_classes)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
        prompts.append(build_default_prompt(args, num_classes))

    # Respect max_turns while preserving order.
    return prompts[: max(args.max_turns, 1)]


def resolve_api_key(args: argparse.Namespace) -> Optional[str]:
    """Determine the API key in priority order."""
    if args.api_key:
        return args.api_key

    # Fallback to common environment variables when using OpenRouter-like APIs.
    for env_var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        if env_key := os.getenv(env_var):
            return env_key

    return None


def should_require_api_key(model_platform: str) -> bool:
    """Heuristic to decide if an API key is mandatory."""
    try:
        platform_enum = ModelPlatformType(model_platform)
    except ValueError:
        # Treat unknown strings as requiring manual handling.
        return True

    return platform_enum not in {
        ModelPlatformType.VLLM,
        ModelPlatformType.OLLAMA,
        ModelPlatformType.SGLANG,
        ModelPlatformType.LMSTUDIO,
        ModelPlatformType.STUB,
    }


def safe_serialize(value: Any) -> Any:
    """Recursively convert objects to JSON-serializable structures."""
    if isinstance(value, dict):
        return {key: safe_serialize(val) for key, val in value.items()}

    if isinstance(value, (list, tuple)):
        return [safe_serialize(item) for item in value]

    if hasattr(value, "model_dump"):
        return value.model_dump()

    if hasattr(value, "dict"):
        return value.dict()

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def serialise_tool_calls(tool_calls: Optional[Iterable[ToolCallingRecord]]) -> List[Dict[str, Any]]:
    """Convert tool call records into plain dictionaries."""
    if not tool_calls:
        return []
    return [record.as_dict() for record in tool_calls]


def persist_results(output_dir: Path, payload: Dict[str, Any]) -> Path:
    """Persist orchestration results to the data directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    file_path = output_dir / f"deepunlearn_orchestration_{timestamp}.json"
    file_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return file_path


# ---------------------------------------------------------------------------
# Orchestrator implementation
# ---------------------------------------------------------------------------


class DeepUnlearnOrchestrator:
    """Encapsulates orchestration workflow around a CAMEL ChatAgent."""

    def __init__(
        self,
        *,
        system_message: str,
        model_platform: str,
        model_type: str,
        model_url: Optional[str],
        api_key: Optional[str],
        temperature: float,
        deepunlearn_dir: Path,
        project_root: Path,
        toolkit_timeout: Optional[int],
        toolkit_verbose: bool,
        logger_name: str = __name__,
    ) -> None:
        self.logger = get_logger(logger_name)
        self._configure_camel_logging()

        # Initialize toolkit and register tools with the agent.
        self.logger.debug(
            "Initializing DeepUnlearn toolkit at %s", deepunlearn_dir
        )
        toolkit = DeepUnlearnToolkit(
            deepunlearn_dir=str(deepunlearn_dir),
            project_root=str(project_root),
            verbose=toolkit_verbose,
            timeout=toolkit_timeout,
        )
        tools = toolkit.get_tools()
        self.logger.info("Registered %d DeepUnlearn tools.", len(tools))

        model_config = {"temperature": temperature}
        model = ModelFactory.create(
            model_platform=model_platform,
            model_type=model_type,
            url=model_url,
            api_key=api_key,
            model_config_dict=model_config,
        )
        self.logger.info(
            "Model backend ready (platform=%s, model=%s).",
            model_platform,
            model_type,
        )

        self.agent = ChatAgent(
            system_message=system_message,
            model=model,
            tools=tools,
            mask_tool_output=False,
            prune_tool_calls_from_memory=False,
        )

    @staticmethod
    def _configure_camel_logging() -> None:
        """Ensure CAMEL internals reuse CAUST loggers."""
        # CAMEL's internal loggers default to WARNING; align with CAUST config.
        camel_logger = camel_get_logger("camel")
        camel_logger.setLevel(get_logger(__name__).level)

    def run(self, prompts: Sequence[str]) -> Dict[str, Any]:
        """Execute the orchestration prompts."""
        conversation: List[Dict[str, Any]] = []
        for turn_index, prompt in enumerate(prompts, start=1):
            self.logger.info("Turn %d: submitting prompt to agent.", turn_index)
            print(f"[Step {turn_index}] input: {prompt}")
            response = self.agent.step(prompt)
            tool_calls = serialise_tool_calls(response.info.get("tool_calls"))
            assistant_contents = [
                msg.content
                for msg in response.msgs
                if msg.role_type.value == "assistant" and msg.content
            ]
            if assistant_contents:
                print(f"[Step {turn_index}] output: {assistant_contents[-1]}")
            else:
                print(f"[Step {turn_index}] output: (no assistant response)")

            turn_payload = {
                "turn": turn_index,
                "user_prompt": prompt,
                "assistant_messages": [
                    {
                        "role": msg.role_type.value,
                        "role_name": msg.role_name,
                        "content": msg.content,
                    }
                    for msg in response.msgs
                ],
                "tool_calls": tool_calls,
                "terminated": response.terminated,
                "info": safe_serialize(
                    {
                        key: value
                        for key, value in response.info.items()
                        if key != "tool_calls"
                    }
                ),
            }

            if tool_calls:
                tool_names = ", ".join(
                    record["tool_name"] for record in tool_calls
                )
                self.logger.info(
                    "Turn %d: executed tool(s): %s.", turn_index, tool_names
                )

            conversation.append(turn_payload)

            if response.terminated:
                self.logger.info(
                    "Agent terminated conversation after turn %d.", turn_index
                )
                break

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompts": list(prompts),
            "conversation": conversation,
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    load_dotenv()
    args = parse_args(argv)

    default_train_save = Path(
        f"unlearn/model_{args.model_arch}-unlearner_{args.train_method}"
    )
    if args.train_save_path is None:
        args.train_save_path = default_train_save

    expected_init_dir = (
        f"unlearn/model_{args.model_arch}-unlearner_{args.train_method}/{args.train_method}"
    )
    init_override_prefix = "unlearner.cfg.model_initializations_dir="
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
    if not any(
        override.startswith(init_override_prefix) for override in unlearn_overrides
    ):
        unlearn_overrides.append(f"{init_override_prefix}{expected_init_dir}")
    args.train_overrides = train_overrides
    args.unlearn_overrides = unlearn_overrides

    logger = setup_logging(
        log_level=args.log_level,
        enable_console=False,
        enable_file=not args.disable_file_logging,
        log_dir=Path("logs"),
    )

    if args.console_format == "text":
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setFormatter(
                    logging.Formatter("%(levelname)s | %(message)s")
                )

    if args.correlation_id:
        set_correlation_id(args.correlation_id)
        logger.info("Correlation ID set to %s.", args.correlation_id)

    auto_pipeline = not (args.prompts or args.prompt_file)
    precomputed_num_classes: Optional[List[int]] = None
    if auto_pipeline:
        try:
            precomputed_num_classes = _parse_num_classes_arg(args.num_classes)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    prompts = ensure_prompts(args, precomputed_num_classes)
    api_key = resolve_api_key(args)

    if should_require_api_key(args.model_platform) and api_key is None:
        raise RuntimeError(
            "No API key detected for the requested model platform. "
            "Provide --api-key or set OPENROUTER_API_KEY / OPENAI_API_KEY."
        )

    orchestrator = DeepUnlearnOrchestrator(
        system_message=DEFAULT_SYSTEM_MESSAGE,
        model_platform=args.model_platform,
        model_type=args.model_type,
        model_url=args.model_url,
        api_key=api_key,
        temperature=args.temperature,
        deepunlearn_dir=args.deepunlearn_dir,
        project_root=args.project_root,
        toolkit_timeout=args.toolkit_timeout,
        toolkit_verbose=args.toolkit_verbose,
    )

    run_payload = orchestrator.run(prompts)
    metadata: Dict[str, Any] = {
        "auto_pipeline": auto_pipeline,
        "dataset": args.dataset,
        "model_arch": args.model_arch,
        "model_seeds": args.model_seeds,
        "model_platform": args.model_platform,
        "model_type": args.model_type,
        "train_method": args.train_method,
        "train_model_seed": args.train_model_seed,
        "unlearn_method": args.unlearn_method,
        "unlearn_model_seed": args.unlearn_model_seed,
    }
    if args.train_random_state is not None:
        metadata["train_random_state"] = args.train_random_state
    if args.unlearn_random_state is not None:
        metadata["unlearn_random_state"] = args.unlearn_random_state
    if args.train_save_path:
        metadata["train_save_path"] = str(args.train_save_path)
    if args.unlearn_save_path:
        metadata["unlearn_save_path"] = args.unlearn_save_path
    if args.data_root:
        metadata["data_root"] = str(args.data_root)
    if args.train_overrides:
        metadata["train_overrides"] = args.train_overrides
    if args.unlearn_overrides:
        metadata["unlearn_overrides"] = args.unlearn_overrides
    if precomputed_num_classes is not None:
        metadata["num_classes"] = precomputed_num_classes
    elif args.num_classes:
        try:
            metadata["num_classes"] = _parse_num_classes_arg(args.num_classes)
        except ValueError:
            metadata["num_classes"] = args.num_classes
    run_payload["metadata"] = metadata

    output_path = persist_results(args.output_dir, run_payload)

    logger.info(
        "Orchestration complete. Saved transcript to %s (%d turn(s)).",
        output_path,
        len(run_payload["conversation"]),
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
