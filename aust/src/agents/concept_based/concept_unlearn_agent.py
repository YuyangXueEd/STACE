"""Concept Unlearning Agent using CAMEL-AI and ESD toolkit.

This module wires the ConceptUnlearnToolkit into a CAMEL ChatAgent, allowing
concept erasure operations to be triggered via natural language prompts. The
agent parses user requests to extract the concept, model, and method variant,
then executes the appropriate ESD unlearning method.

Typical usage from the repository root::

    python -m aust.src.agents.concept_based.concept_unlearn_agent \
        --concept "Van Gogh" --model stable-diffusion

Or with a custom natural language prompt::

    python -m aust.src.agents.concept_based.concept_unlearn_agent \
        --prompt "using ESD to erase nudity from SDXL with noxattn"

The script requires an LLM endpoint that supports tool calling. By default it
targets OpenRouter; set OPENROUTER_API_KEY in your environment.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Optional, Sequence

from dotenv import load_dotenv

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType

from aust.src.utils.logging_config import get_logger, setup_logging
from aust.src.toolkits import ConceptUnlearnToolkit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_MESSAGE = dedent(
    """\
    You are a Concept Unlearning Orchestrator for text-to-image models.

    Your role is to parse user prompts and extract the following parameters:
    - unlearning_method: Currently only 'ESD' is supported
    - concept: The concept to erase (e.g., 'Van Gogh', 'nudity', 'monster')
    - model: T2I model to modify (optional, default: 'stable-diffusion')
      Supported models: stable-diffusion, sdxl, flux
    - method_variant: ESD variant (optional, default: 'xattn')
      For SD: xattn, noxattn, full, selfattn
      For SDXL/FLUX: esd-x, esd-x-strict (but xattn/noxattn will be mapped automatically)

    Parse natural language prompts like:
    - "using ESD to erase Van Gogh" -> erase_concept_esd(concept="Van Gogh", model="stable-diffusion", method_variant="xattn")
    - "using ESD to erase nudity from SDXL with noxattn" -> erase_concept_esd(concept="nudity", model="sdxl", method_variant="noxattn")
    - "erase monster from Flux using ESD" -> erase_concept_esd(concept="monster", model="flux", method_variant="xattn")

    When a user provides a prompt:
    1. Extract the concept, model, and method_variant
    2. Use defaults when parameters are not specified
    3. Call the erase_concept_esd tool with the extracted parameters
    4. Report the results clearly, including the checkpoint path

    After tool execution, explain:
    - What concept was erased
    - Which model was modified
    - Where the checkpoint was saved
    - Training parameters used

    Once it failed, suggest checking logs for details. And stop further actions.
    """
)

DEFAULT_MODEL_PLATFORM = ModelPlatformType.OPENAI_COMPATIBLE_MODEL.value
DEFAULT_MODEL_TYPE = "openai/gpt-5-nano"
DEFAULT_MODEL_URL = "https://openrouter.ai/api/v1"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_CONCEPT = "Van Gogh"
DEFAULT_T2I_MODEL = "stable-diffusion"
DEFAULT_METHOD = "ESD"
DEFAULT_VARIANT = "xattn"

# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Concept Unlearning agent for T2I models."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Natural language prompt (e.g., 'using ESD to erase Van Gogh from SDXL')",
    )
    parser.add_argument(
        "--concept",
        default=DEFAULT_CONCEPT,
        help=f"Concept to erase (default: {DEFAULT_CONCEPT})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_T2I_MODEL,
        help=f"T2I model to modify (default: {DEFAULT_T2I_MODEL}). "
             "Options: stable-diffusion, sdxl, flux",
    )
    parser.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        help=f"Unlearning method (default: {DEFAULT_METHOD}). Currently only ESD is supported.",
    )
    parser.add_argument(
        "--variant",
        default=DEFAULT_VARIANT,
        help=f"Method variant (default: {DEFAULT_VARIANT}). "
             "For SD: xattn, noxattn, full, selfattn. For SDXL/FLUX: esd-x, esd-x-strict",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Number of training iterations (default: 200)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output from ESD training",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory for log files (default: logs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/concept_unlearn_interactions"),
        help="Output directory for interaction logs (default: data/concept_unlearn_interactions)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Timeout in seconds for ESD training subprocesses (default: 7200)",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main agent execution
# ---------------------------------------------------------------------------


def run_concept_unlearn_agent(
    prompts: list[str],
    toolkit: ConceptUnlearnToolkit,
    model_platform: str = DEFAULT_MODEL_PLATFORM,
    model_type: str = DEFAULT_MODEL_TYPE,
    model_url: str = DEFAULT_MODEL_URL,
    api_key: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    system_message: str = DEFAULT_SYSTEM_MESSAGE,
    logger=None,
) -> dict:
    """Run the concept unlearning agent with the given prompts.

    Args:
        prompts: List of user prompts to execute
        toolkit: ConceptUnlearnToolkit instance
        model_platform: LLM platform (default: OPENAI_COMPATIBLE_MODEL)
        model_type: LLM model type (default: openai/gpt-5)
        model_url: API URL (default: OpenRouter)
        api_key: API key for the LLM (default: from OPENROUTER_API_KEY env)
        temperature: LLM temperature (default: 0.2)
        system_message: System message for the agent
        logger: Logger instance

    Returns:
        Dict containing interaction history and results
    """
    if logger is None:
        logger = get_logger(__name__)

    # Create LLM model
    logger.info(f"Creating model: {model_type}")
    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found in environment. "
                "Please set it or pass --api-key"
            )

    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type=model_type,
        url=model_url,
        api_key=api_key,
        model_config_dict={"temperature": temperature},
    )

    # Create ChatAgent with toolkit
    logger.info("Creating ConceptUnlearnAgent with toolkit")
    agent = ChatAgent(
        system_message=system_message,
        model=model,
        tools=toolkit.get_tools(),
    )

    # Execute prompts
    interaction_history = []
    for i, prompt in enumerate(prompts):
        logger.info(f"Processing prompt {i + 1}/{len(prompts)}: {prompt[:100]}...")

        response = agent.step(prompt)
        response_content = response.msgs[0].content if response.msgs else ""

        logger.info(f"Agent response received ({len(response_content)} chars)")
        logger.debug(f"Full response: {response_content}")

        # Log tool calls if any
        if hasattr(response, "info") and response.info and "tool_calls" in response.info:
            tool_calls = response.info["tool_calls"] or []
            logger.info(f"Tool calls executed: {len(tool_calls)}")
            for tc in tool_calls:
                tool_name = getattr(tc, "tool_name", getattr(tc, "func_name", "unknown"))
                tool_args = getattr(tc, "args", {})
                logger.debug(f"Tool: {tool_name}, Args: {tool_args}")

        interaction_history.append({
            'prompt': prompt,
            'response': response_content,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'tool_calls': (
                [
                    {
                        'function': getattr(tc, "tool_name", getattr(tc, "func_name", None)),
                        'arguments': getattr(tc, "args", {}),
                        'result': getattr(tc, "result", None),
                    }
                    for tc in response.info.get('tool_calls', [])
                ]
                if hasattr(response, "info") and response.info else []
            ),
        })

        print("\n" + "=" * 80)
        print(f"PROMPT {i + 1}:")
        print("=" * 80)
        print(prompt)
        print("\n" + "-" * 80)
        print("RESPONSE:")
        print("-" * 80)
        print(response_content)
        print("=" * 80 + "\n")

    return {
        'interactions': interaction_history,
        'model_type': model_type,
        'temperature': temperature,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main entry point for the concept unlearning agent."""
    args = parse_args(argv)

    # Load environment
    load_dotenv()

    # Setup logging
    setup_logging(
        log_level=args.log_level,
        log_dir=args.log_dir,
        enable_console=True,
        enable_file=True,
    )
    logger = get_logger(__name__)

    logger.info("Starting ConceptUnlearnAgent")
    logger.info(f"Arguments: {vars(args)}")

    # Initialize toolkit
    toolkit = ConceptUnlearnToolkit(
        esd_dir="./external/esd",
        project_root=".",
        verbose=args.verbose,
        timeout=args.timeout,
    )

    # Build prompts
    prompts = []
    if args.prompt:
        # Use custom natural language prompt
        prompts.append(args.prompt)
    else:
        # Build structured prompt from CLI args
        prompt = (
            f"using {args.method} to erase {args.concept} from {args.model} "
            f"with {args.variant}"
        )
        if args.iterations != 200:
            prompt += f" ({args.iterations} iterations)"
        prompts.append(prompt)

    logger.info(f"Executing {len(prompts)} prompt(s)")

    # Run agent
    try:
        result = run_concept_unlearn_agent(
            prompts=prompts,
            toolkit=toolkit,
            logger=logger,
        )

        # Save interaction log
        output_dir = args.output_dir or Path("data/concept_unlearn_interactions")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"interaction_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

        logger.info(f"Interaction log saved to: {output_file}")
        logger.info("ConceptUnlearnAgent completed successfully")

        return 0

    except Exception as e:
        logger.error(f"Agent execution failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
