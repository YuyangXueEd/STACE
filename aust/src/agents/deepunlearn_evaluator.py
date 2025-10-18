"""DeepUnlearn evaluation agent using CAMEL-AI toolkits.

This module wires :class:`DeepUnlearnEvaluationToolkit` into a CAMEL agent so
that post-unlearning evaluation metrics can be produced from natural language
instructions.  The agent expects checkpoints and dataset splits to reside in
the ``data`` directory (or user-provided paths) and supports prompts such as
``"test all the evaluation metrics"`` or ``"test a specific evaluation metric"``.

Typical usage from the repository root::

    python -m aust.src.agents.deepunlearn_evaluator \\
        --splits-dir data/splits/cifar10/123 \\
        --original-model data/models/resnet18_cifar10_seed000_original.pth \\
        --unlearned-model data/models/resnet18_cifar10_seed000_finetune.pth

The script targets an OpenAI-compatible endpoint by default.  Override the
model platform, type, or API key with CLI flags or environment variables
(``DEEPUNLEARN_AGENT_*``) mirroring :mod:`deepunlearn_orchestrator`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
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
from aust.src.toolkits import DeepUnlearnEvaluationToolkit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_MESSAGE = dedent(
    """\
    You are the DeepUnlearn Evaluation Orchestrator for CAUST.

    Evaluate original and unlearned checkpoints by invoking the registered
    DeepUnlearn evaluation tools. Follow this workflow:
    1. Confirm which metrics the user wants to analyse (all metrics by default).
    2. Call the evaluation toolkit functions with explicit arguments. Evaluate
       each supplied checkpoint before computing comparisons.
    3. Use derived tools (e.g. retention, indiscernibility, weight distance) to
       compare original and unlearned models when appropriate.
    4. Summarise the computed metrics in a concise report that cites the
       underlying tool outputs. Highlight remaining follow-ups if data is
       missing.

    Never fabricate numbers—only report results returned by the tools or
    provided in the context instructions.
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
DEFAULT_OUTPUT_DIR = Path("data")
DEFAULT_DATASET = "cifar10"
DEFAULT_MODEL_ARCH = "resnet18"
DEFAULT_BATCH_SIZE = 128
DEFAULT_RANDOM_STATE = 0
DEFAULT_CONSOLE_FORMAT = "text"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EvaluationInputs:
    """Resolved paths and settings required for evaluation."""

    dataset: str
    checkpoint_model_type: str
    splits_dir: Path
    original_model: Path
    unlearned_model: Path
    batch_size: int
    random_state: int
    device: str
    split_index: Optional[int]
    forget_index: Optional[int]


# ---------------------------------------------------------------------------
# CLI parsing helpers
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the evaluation agent."""
    parser = argparse.ArgumentParser(
        description="Run the DeepUnlearn evaluation agent."
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="User prompt(s) to feed into the agent (repeatable).",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Path to a text file containing an additional prompt.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Dataset name for evaluation (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--model-arch",
        default=DEFAULT_MODEL_ARCH,
        help=f"Model architecture (passed as model_type) for evaluation calls (default: {DEFAULT_MODEL_ARCH}).",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        required=True,
        help="Directory containing retain/forget/val/test split indices (e.g., data/splits/<dataset>/123).",
    )
    parser.add_argument(
        "--original-model",
        type=Path,
        required=True,
        help="Checkpoint path for the original (pre-unlearning) model.",
    )
    parser.add_argument(
        "--unlearned-model",
        type=Path,
        required=True,
        help="Checkpoint path for the unlearned model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size used during evaluation (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Random seed forwarded to evaluation helper functions (default: {DEFAULT_RANDOM_STATE}).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Device string forwarded to evaluation (default: "auto").',
    )
    parser.add_argument(
        "--split-index",
        type=int,
        default=None,
        help="Optional LIRA split index to evaluate specific forget partitions.",
    )
    parser.add_argument(
        "--forget-index",
        type=int,
        default=None,
        help="Optional LIRA forget index used alongside --split-index.",
    )
    parser.add_argument(
        "--model-platform",
        default=os.getenv(DEFAULT_MODEL_PLATFORM_ENV, DEFAULT_MODEL_PLATFORM),
        help="Model platform identifier understood by CAMEL.",
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
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where evaluation transcripts will be saved (default: data/).",
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
        help="Optional timeout (seconds) for toolkit commands.",
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
        help="Logging level for the evaluator (default: INFO).",
    )
    parser.add_argument(
        "--console-format",
        choices=("json", "text"),
        default=DEFAULT_CONSOLE_FORMAT,
        help="Console log format (default: text).",
    )
    parser.add_argument(
        "--disable-file-logging",
        action="store_true",
        help="Disable writing evaluator logs to disk (console only).",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def build_context_block(inputs: EvaluationInputs) -> str:
    """Prepare a textual context block describing available resources."""
    context_lines = [
        f"- dataset: {inputs.dataset}",
        f"- model_type: {inputs.checkpoint_model_type}",
        f"- splits_directory: {inputs.splits_dir}",
        f"- original_model_checkpoint: {inputs.original_model}",
        f"- unlearned_model_checkpoint: {inputs.unlearned_model}",
        f"- evaluation_batch_size: {inputs.batch_size}",
        f"- random_state: {inputs.random_state}",
        f"- evaluation_device: {inputs.device}",
    ]
    if inputs.split_index is not None and inputs.forget_index is not None:
        context_lines.append(
            f"- lira_indices: split_index={inputs.split_index}, forget_index={inputs.forget_index}"
        )
    return "Context:\n" + "\n".join(context_lines)


def build_default_prompt(inputs: EvaluationInputs) -> str:
    """Construct a default prompt instructing the agent to evaluate all metrics."""
    context = build_context_block(inputs)
    return dedent(
        f"""\
        Test all available evaluation metrics for the supplied DeepUnlearn artifacts.

        Use the DeepUnlearnEvaluationToolkit to:
        1. Call `evaluate_model_checkpoint` for both checkpoints with dataset="{inputs.dataset}", model_type="{inputs.checkpoint_model_type}", splits_dir="{inputs.splits_dir}", batch_size={inputs.batch_size}, random_state={inputs.random_state}, device="{inputs.device}"{", split_index=%d, forget_index=%d" % (inputs.split_index, inputs.forget_index) if inputs.split_index is not None else ""}.
        2. Compute indiscernibility, accuracy retention, and model weight distance based on the returned metrics.
        3. Summarise the evaluated metrics in a concise report comparing the two models.

        {context}
        """
    ).strip()


def attach_context(prompt: str, context: str) -> str:
    """Append the resource context to a user prompt."""
    prompt = prompt.rstrip()
    if context not in prompt:
        prompt = f"{prompt}\n\n{context}"
    return prompt


def gather_prompts(args: argparse.Namespace, inputs: EvaluationInputs) -> List[str]:
    """Collect prompts from CLI arguments or fallback to the default."""
    prompts: List[str] = []
    if args.prompts:
        prompts.extend(args.prompts)

    if args.prompt_file:
        prompts.append(args.prompt_file.read_text(encoding="utf-8"))

    context_block = build_context_block(inputs)
    if not prompts:
        prompts.append(build_default_prompt(inputs))
    else:
        prompts = [attach_context(prompt, context_block) for prompt in prompts]

    return prompts[: max(args.max_turns, 1)]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def resolve_api_key(args: argparse.Namespace) -> Optional[str]:
    """Determine the API key in priority order."""
    if args.api_key:
        return args.api_key

    for env_var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        if env_key := os.getenv(env_var):
            return env_key

    return None


def should_require_api_key(model_platform: str) -> bool:
    """Decide whether an API key is required for the selected platform."""
    try:
        platform_enum = ModelPlatformType(model_platform)
    except ValueError:
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
    """Convert tool call records into serializable dictionaries."""
    if not tool_calls:
        return []
    return [record.as_dict() for record in tool_calls]


def persist_results(output_dir: Path, payload: Dict[str, Any]) -> Path:
    """Write evaluation transcripts to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    file_path = output_dir / f"deepunlearn_evaluation_{timestamp}.json"
    file_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return file_path


def resolve_inputs(args: argparse.Namespace) -> EvaluationInputs:
    """Validate and bundle evaluation resources into a dataclass."""
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

    return EvaluationInputs(
        dataset=args.dataset,
        checkpoint_model_type=args.model_arch,
        splits_dir=splits_dir,
        original_model=original_model,
        unlearned_model=unlearned_model,
        batch_size=args.batch_size,
        random_state=args.random_state,
        device=args.device,
        split_index=args.split_index,
        forget_index=args.forget_index,
    )


# ---------------------------------------------------------------------------
# Agent implementation
# ---------------------------------------------------------------------------


class DeepUnlearnEvaluationAgent:
    """Driver for the evaluation workflow around a CAMEL ChatAgent."""

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

        self.logger.debug(
            "Initializing DeepUnlearn evaluation toolkit at %s",
            deepunlearn_dir,
        )
        toolkit = DeepUnlearnEvaluationToolkit(
            deepunlearn_dir=str(deepunlearn_dir),
            project_root=str(project_root),
            verbose=toolkit_verbose,
            timeout=toolkit_timeout,
        )
        tools = toolkit.get_tools()
        self.logger.info("Registered %d evaluation tool(s).", len(tools))

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
        """Synchronise CAMEL logging with CAUST configuration."""
        camel_logger = camel_get_logger("camel")
        camel_logger.setLevel(get_logger(__name__).level)

    def run(self, prompts: Sequence[str]) -> Dict[str, Any]:
        """Execute evaluation prompts and capture the transcript."""
        conversation: List[Dict[str, Any]] = []
        for turn_index, prompt in enumerate(prompts, start=1):
            self.logger.info("Turn %d: submitting prompt to evaluation agent.", turn_index)
            print(f"[Turn {turn_index}] input: {prompt}")
            response = self.agent.step(prompt)
            tool_calls = serialise_tool_calls(response.info.get("tool_calls"))
            assistant_contents = [
                msg.content
                for msg in response.msgs
                if msg.role_type.value == "assistant" and msg.content
            ]
            if assistant_contents:
                print(f"[Turn {turn_index}] output: {assistant_contents[-1]}")
            else:
                print(f"[Turn {turn_index}] output: (no assistant response)")

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
                    "Turn %d: executed tool(s): %s.",
                    turn_index,
                    tool_names,
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

    inputs = resolve_inputs(args)
    prompts = gather_prompts(args, inputs)
    api_key = resolve_api_key(args)

    if should_require_api_key(args.model_platform) and api_key is None:
        raise RuntimeError(
            "No API key detected for the requested model platform. "
            "Provide --api-key or set OPENROUTER_API_KEY / OPENAI_API_KEY."
        )

    evaluator = DeepUnlearnEvaluationAgent(
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

    run_payload = evaluator.run(prompts)
    run_payload["metadata"] = {
        "dataset": inputs.dataset,
        "checkpoint_model_type": inputs.checkpoint_model_type,
        "splits_dir": str(inputs.splits_dir),
        "original_model": str(inputs.original_model),
        "unlearned_model": str(inputs.unlearned_model),
        "batch_size": inputs.batch_size,
        "random_state": inputs.random_state,
        "device": inputs.device,
        "split_index": inputs.split_index,
        "forget_index": inputs.forget_index,
        "model_platform": args.model_platform,
        "llm_model_type": args.model_type,
    }

    output_path = persist_results(args.output_dir, run_payload)

    logger.info(
        "Evaluation complete. Saved transcript to %s (%d turn(s)).",
        output_path,
        len(run_payload["conversation"]),
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
