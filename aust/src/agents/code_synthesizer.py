"""
Code Synthesizer Agent for concept-erasure attack code generation and repair.

This agent translates refined hypotheses into executable Python code for
concept-erasure attacks, using a code-repair-execute loop to automatically
fix failures up to a configurable retry budget (default 5, max 5).

The agent uses Ollama's qwen3-coder:latest model (256k context) to maintain full
synthesis → execute → repair history in a single conversation, improving repair
quality through context continuity.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml
from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig, OllamaConfig
from camel.interpreters.subprocess_interpreter import SubprocessInterpreter
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from dotenv import load_dotenv

from aust.src.data_models import (
    CodeArtifact,
    CodeArtifactStatus,
    CodeRepairHistory,
    ExecutionStatus,
    Hypothesis,
    RunResult,
)
from aust.src.utils.logging_config import get_logger

load_dotenv()
logger = get_logger(__name__)

AUST_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = AUST_ROOT / "configs" / "prompts"
MODELS_DIR = AUST_ROOT / "configs" / "models"
CODE_SYNTHESIZER_PROMPT = "code_synthesizer_concept_erasure.yaml"
CODE_SYNTHESIZER_MODEL = "code_synthesizer.yaml"
DIFFUSERS_LIB_PATH = AUST_ROOT.parent / "external" / "diffusers"
REPO_ROOT = AUST_ROOT.parent


def _load_model_settings_strict(name: str) -> dict[str, Any]:
    """Load model settings and raise if the YAML file is missing or invalid."""
    path = MODELS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Model settings not found for '{name}' at {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    model_name = data.get("model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError(f"Model settings '{name}' must define a non-empty model_name")

    config = data.get("config") or {}
    if not isinstance(config, dict):
        raise ValueError(f"Model settings '{name}' must provide a dict under 'config'")

    model_platform_value = data.get("model_platform")
    if model_platform_value is None:
        model_platform = ModelPlatformType.OPENAI_COMPATIBLE_MODEL
    else:
        try:
            model_platform = ModelPlatformType(str(model_platform_value).lower())
        except ValueError as exc:
            raise ValueError(
                f"Invalid model_platform '{model_platform_value}' in settings '{name}'"
            ) from exc

    requires_api_key = data.get("requires_api_key")
    if requires_api_key is None:
        requires_api_key = model_platform not in {
            ModelPlatformType.OLLAMA,
            ModelPlatformType.STUB,
        }
    elif not isinstance(requires_api_key, bool):
        raise ValueError(
            f"Model settings '{name}' must define requires_api_key as a boolean"
        )

    api_key_env_var = data.get("api_key_env_var")
    if api_key_env_var is not None and (
        not isinstance(api_key_env_var, str) or not api_key_env_var.strip()
    ):
        raise ValueError(
            f"Model settings '{name}' must define api_key_env_var as a non-empty string"
        )

    base_url = data.get("base_url")
    if base_url is not None and (not isinstance(base_url, str) or not base_url.strip()):
        raise ValueError(
            f"Model settings '{name}' must define base_url as a non-empty string"
        )

    return {
        "model_name": model_name.strip(),
        "config": config,
        "model_platform": model_platform,
        "requires_api_key": requires_api_key,
        "api_key_env_var": api_key_env_var.strip() if api_key_env_var else None,
        "base_url": base_url.strip() if base_url else None,
    }


_CODE_SYNTHESIZER_SETTINGS = _load_model_settings_strict("code_synthesizer")


class CodeSynthesizerAgent:
    """
    Agent that synthesizes concept-erasure attack code from hypotheses and repairs failures.

    This agent forms the complete Step 3 of the inner loop:
    1. Takes final hypothesis from Step 2 (hypothesis refinement)
    2. Generates executable Python code for concept-erasure attack
    3. Executes code in sandboxed environment
    4. Repairs code on failure (retry budget: default 5, max 5)
    5. Reports results to Inner Loop Orchestrator for Step 4 (evaluation)

    Features:
    - Single agent handles both synthesis and repair (leveraging 256k context window)
    - References external/diffusers library for implementation patterns
    - Uses CAMEL's SubprocessInterpreter for sandboxed execution
    - Tracks repair history to avoid oscillations
    - Focused diff repairs (not full rewrites)
    """

    DEFAULT_MODEL = _CODE_SYNTHESIZER_SETTINGS["model_name"]
    DEFAULT_PLATFORM = _CODE_SYNTHESIZER_SETTINGS["model_platform"]
    DEFAULT_TIMEOUT_SECONDS = 15 * 60  # 15 minutes
    DEFAULT_MAX_RETRIES = 5
    MAX_RETRIES_LIMIT = 5

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        execution_timeout: int = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        output_dir: Optional[Path] = None,
        require_confirm: bool = False,
    ) -> None:
        """
        Initialize CodeSynthesizer agent.

        Args:
            model_name: Override model for code generation (default: qwen3-coder:latest)
            execution_timeout: Timeout for code execution in seconds (default: 900)
            max_retries: Maximum repair attempts (default: 5, max: 5)
            output_dir: Base directory for code and logs (default: ./outputs)
            require_confirm: Require user confirmation before execution (default: False)
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.model_platform = self.DEFAULT_PLATFORM
        self.execution_timeout = execution_timeout
        self.max_retries = min(max_retries, self.MAX_RETRIES_LIMIT)
        self.output_dir = output_dir or (REPO_ROOT / "outputs")
        self.require_confirm = require_confirm

        self._requires_api_key = _CODE_SYNTHESIZER_SETTINGS["requires_api_key"]
        self._api_key_env_var = _CODE_SYNTHESIZER_SETTINGS.get("api_key_env_var")
        self._base_url = _CODE_SYNTHESIZER_SETTINGS.get("base_url")

        api_key: Optional[str] = None
        if self._requires_api_key:
            env_var = self._api_key_env_var or "OPENROUTER_API_KEY"
            api_key = os.getenv(env_var)
            if not api_key:
                raise EnvironmentError(
                    f"{env_var} is required to initialize CodeSynthesizerAgent"
                )
        elif self._api_key_env_var:
            api_key = os.getenv(self._api_key_env_var)

        model_settings = _CODE_SYNTHESIZER_SETTINGS["config"].copy()
        self._model_backend = self._create_backend(
            model_name=self.model_name,
            config=model_settings,
            platform=self.model_platform,
            api_key=api_key,
            base_url=self._base_url,
        )

        self._prompt_config = self._load_prompt_config()
        self._interpreter = SubprocessInterpreter(
            require_confirm=self.require_confirm,
            print_stdout=False,
            print_stderr=False,
            execution_timeout=self.execution_timeout,
        )

        logger.info(
            "CodeSynthesizerAgent initialized (model=%s, platform=%s, timeout=%ds, max_retries=%d)",
            self.model_name,
            self.model_platform.value,
            self.execution_timeout,
            self.max_retries,
        )

    def synthesize_and_execute(
        self,
        hypothesis: Hypothesis,
        task_spec: dict[str, Any],
        task_id: str,
        iteration_number: int,
    ) -> tuple[CodeRepairHistory, Optional[CodeArtifact], Optional[RunResult]]:
        """
        Complete code-repair-execute loop for a hypothesis.

        This is the main entry point for Step 3 of the inner loop.

        Args:
            hypothesis: Refined hypothesis from Step 2
            task_spec: Structured TaskSpec dictionary
            task_id: Unique task identifier
            iteration_number: Current iteration number

        Returns:
            Tuple of (repair_history, final_artifact, final_run_result)
        """
        logger.info(
            "Starting code-repair-execute loop for hypothesis %s (iteration %d)",
            hypothesis.hypothesis_id,
            iteration_number,
        )

        # Initialize repair history
        repair_history = CodeRepairHistory(
            hypothesis_id=hypothesis.hypothesis_id,
            task_id=task_id,
            iteration_number=iteration_number,
            max_attempts=self.max_retries,
        )

        # Create output directory for this hypothesis
        run_output_dir = (
            self.output_dir
            / "runs"
            / f"iter_{iteration_number:02d}_{hypothesis.hypothesis_id[:8]}"
        )
        run_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Run output directory: %s", run_output_dir)

        # Initialize chat agent for synthesis and repair
        agent = self._create_chat_agent()

        try:
            # Attempt 1: Initial synthesis
            artifact = self._synthesize_code(
                agent=agent,
                hypothesis=hypothesis,
                task_spec=task_spec,
                task_id=task_id,
                iteration_number=iteration_number,
                repair_attempt=0,
                previous_error=None,
                run_output_dir=run_output_dir,
            )

            while repair_history.should_continue:
                # Execute code
                run_result = self._execute_code(
                    artifact=artifact,
                    run_output_dir=run_output_dir,
                    attempt_number=repair_history.current_attempt + 1,
                )

                # Record attempt
                repair_history.add_attempt(artifact, run_result)

                # Check if successful
                if run_result.is_success:
                    logger.info(
                        "Code execution succeeded on attempt %d/%d",
                        repair_history.current_attempt,
                        self.max_retries,
                    )
                    break

                # Check if repairable
                if not run_result.is_repairable:
                    logger.warning(
                        "Code execution failed with non-repairable error: %s",
                        run_result.status.value,
                    )
                    break

                # Check if retries exhausted
                if not repair_history.should_continue:
                    logger.warning(
                        "Max retry attempts exhausted (%d/%d)",
                        repair_history.current_attempt,
                        self.max_retries,
                    )
                    break

                # Repair code
                logger.info(
                    "Attempting repair %d/%d for artifact %s",
                    repair_history.current_attempt + 1,
                    self.max_retries,
                    artifact.artifact_id[:8],
                )

                error_feedback = run_result.extract_error_for_repair()
                artifact = self._repair_code(
                    agent=agent,
                    previous_artifact=artifact,
                    error_feedback=error_feedback,
                    repair_attempt=repair_history.current_attempt + 1,
                )

        finally:
            agent.reset()

        # Save repair history
        history_file = run_output_dir / "repair_history.json"
        repair_history_data = repair_history.model_dump(mode="json")
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(repair_history_data, f, indent=2, default=str)

        logger.info(
            "Code-repair-execute loop complete: %s (attempts=%d, success=%s)",
            hypothesis.hypothesis_id[:8],
            repair_history.current_attempt,
            repair_history.is_success,
        )

        return (
            repair_history,
            repair_history.final_artifact,
            repair_history.final_run_result,
        )

    def _synthesize_code(
        self,
        *,
        agent: ChatAgent,
        hypothesis: Hypothesis,
        task_spec: dict[str, Any],
        task_id: str,
        iteration_number: int,
        repair_attempt: int,
        previous_error: Optional[str],
        run_output_dir: Optional[Path] = None,
    ) -> CodeArtifact:
        """
        Generate code artifact from hypothesis.

        Args:
            agent: Chat agent for code generation
            hypothesis: Source hypothesis
            task_spec: Structured TaskSpec
            task_id: Task ID
            iteration_number: Iteration number
            repair_attempt: Repair attempt number (0 for initial)
            previous_error: Error feedback from previous attempt (if repair)
            run_output_dir: Output directory for this run (where to save artifacts)

        Returns:
            CodeArtifact with generated code
        """
        user_prompt = self._build_synthesis_prompt(
            hypothesis=hypothesis,
            task_spec=task_spec,
            is_repair=repair_attempt > 0,
            error_feedback=previous_error,
            run_output_dir=run_output_dir,
        )

        logger.debug("Sending synthesis prompt to model (repair_attempt=%d)", repair_attempt)

        response = agent.step(BaseMessage.make_user_message("Orchestrator", user_prompt))
        content = self._extract_last_content(response)

        # Extract code from response
        code = self._extract_code_from_response(content)

        artifact = CodeArtifact(
            code=code,
            hypothesis_id=hypothesis.hypothesis_id,
            task_id=task_id,
            iteration_number=iteration_number,
            repair_attempt=repair_attempt,
            model_used=self.model_name,
        )

        if repair_attempt > 0:
            artifact.repair_feedback = previous_error
            artifact.status = CodeArtifactStatus.REPAIRED

        logger.info(
            "Code synthesized: artifact_id=%s (repair_attempt=%d, code_length=%d)",
            artifact.artifact_id[:8],
            repair_attempt,
            len(code),
        )

        return artifact

    def _repair_code(
        self,
        *,
        agent: ChatAgent,
        previous_artifact: CodeArtifact,
        error_feedback: str,
        repair_attempt: int,
    ) -> CodeArtifact:
        """
        Repair code based on error feedback.

        This uses the same agent/conversation, leveraging the 256k context window
        to maintain full synthesis → execute → repair history.

        Args:
            agent: Chat agent (same conversation)
            previous_artifact: Failed artifact
            error_feedback: Error information for repair
            repair_attempt: Repair attempt number

        Returns:
            Repaired CodeArtifact
        """
        repair_prompt = self._build_repair_prompt(
            previous_code=previous_artifact.code,
            error_feedback=error_feedback,
            repair_attempt=repair_attempt,
        )

        logger.debug("Sending repair prompt to model (attempt=%d)", repair_attempt)

        response = agent.step(BaseMessage.make_user_message("Orchestrator", repair_prompt))
        content = self._extract_last_content(response)

        # Extract repaired code
        repaired_code = self._extract_code_from_response(content)

        repaired_artifact = CodeArtifact(
            code=repaired_code,
            hypothesis_id=previous_artifact.hypothesis_id,
            task_id=previous_artifact.task_id,
            iteration_number=previous_artifact.iteration_number,
            parent_artifact_id=previous_artifact.artifact_id,
            repair_attempt=repair_attempt,
            repair_feedback=error_feedback,
            status=CodeArtifactStatus.REPAIRED,
            model_used=self.model_name,
        )

        logger.info(
            "Code repaired: artifact_id=%s (parent=%s, attempt=%d)",
            repaired_artifact.artifact_id[:8],
            previous_artifact.artifact_id[:8],
            repair_attempt,
        )

        return repaired_artifact

    def _execute_code(
        self,
        *,
        artifact: CodeArtifact,
        run_output_dir: Path,
        attempt_number: int,
    ) -> RunResult:
        """
        Execute code artifact in sandboxed environment.

        Args:
            artifact: Code artifact to execute
            run_output_dir: Directory for execution outputs
            attempt_number: Attempt number for logging

        Returns:
            RunResult with execution details
        """
        logger.info(
            "Executing code artifact %s (attempt %d)",
            artifact.artifact_id[:8],
            attempt_number,
        )

        # Mark artifact as executing
        artifact.mark_status(CodeArtifactStatus.EXECUTING)

        # Save code to file
        code_file = run_output_dir / f"attempt_{attempt_number:02d}_{artifact.artifact_id[:8]}.py"
        artifact.save_to_file(code_file)

        # Create run result
        run_result = RunResult(
            artifact_id=artifact.artifact_id,
            status=ExecutionStatus.FAILURE,  # Default to failure
            started_at=datetime.now(timezone.utc),
            timeout_seconds=self.execution_timeout,
            output_dir=run_output_dir,
        )

        # Execute code
        try:
            exec_output = self._interpreter.run_file(file=code_file, code_type="python")

            # Parse execution result
            run_result.completed_at = datetime.now(timezone.utc)

            # Parse execution output into stdout, stderr, and exit code.
            stdout_text = exec_output.strip()
            stderr_text: Optional[str] = None
            exit_code = 0

            stderr_pattern = r"\(stderr:\s*(.*?)(?=\)\s*(?:\(Execution failed with return code \d+\)|$))"
            stderr_match = re.search(stderr_pattern, exec_output, re.DOTALL)
            if stderr_match:
                stdout_text = exec_output[: stderr_match.start()].strip()
                stderr_text = stderr_match.group(1).strip()
            elif "(stderr:" in exec_output:
                prefix, _, suffix = exec_output.partition("(stderr:")
                stdout_text = prefix.strip()
                end_idx = suffix.rfind(")")
                if end_idx != -1:
                    stderr_text = suffix[:end_idx].strip()
                else:
                    stderr_text = suffix.strip()

            exit_code_match = re.search(
                r"\(Execution failed with return code (\d+)\)", exec_output
            )
            if exit_code_match:
                exit_code = int(exit_code_match.group(1))

            combined_output = "\n".join(
                part for part in (stdout_text, stderr_text) if part
            )
            exception_detected = self._contains_exception_signature(combined_output)

            run_result.stdout = stdout_text
            if stderr_text:
                run_result.stderr = stderr_text

            execution_failed = exit_code != 0 or exception_detected
            if execution_failed and exit_code == 0 and exception_detected:
                logger.warning(
                    "Detected exception signature in output despite zero exit code; "
                    "marking execution as failure."
                )
                exit_code = 1

            if execution_failed:
                run_result.exit_code = exit_code
                run_result.status = ExecutionStatus.FAILURE
                artifact.mark_status(CodeArtifactStatus.FAILED)

                if not run_result.stderr:
                    # Fall back to any output we captured so the repair agent has context.
                    run_result.stderr = combined_output or "Execution failed with no stderr output."

                run_result.error_summary = self._extract_error_summary(run_result.stderr)
                run_result.error_snippet = self._extract_error_snippet(run_result.stderr)
                run_result.traceback = self._extract_traceback(run_result.stderr)
            else:
                run_result.exit_code = 0
                run_result.status = ExecutionStatus.SUCCESS
                artifact.mark_status(CodeArtifactStatus.SUCCESS)

        except Exception as exc:  # pylint: disable=broad-except
            run_result.completed_at = datetime.now(timezone.utc)
            run_result.status = ExecutionStatus.FAILURE
            run_result.stderr = str(exc)
            run_result.exit_code = 1
            run_result.error_summary = f"Execution exception: {type(exc).__name__}"
            artifact.mark_status(CodeArtifactStatus.FAILED)
            logger.error(
                "Code execution raised exception: %s",
                exc,
                exc_info=True,
            )

        # Save run result metadata
        log_file = run_output_dir / f"run_{attempt_number:02d}_{run_result.run_id[:8]}.json"
        run_result.log_file = log_file
        run_result.save_metadata(log_file)

        duration = run_result.duration_seconds or 0.0
        logger.info(
            "Execution complete: status=%s, exit_code=%d, duration=%.1fs",
            run_result.status.value,
            run_result.exit_code,
            duration,
        )

        return run_result

    def _build_synthesis_prompt(
        self,
        *,
        hypothesis: Hypothesis,
        task_spec: dict[str, Any],
        is_repair: bool,
        error_feedback: Optional[str],
        run_output_dir: Optional[Path] = None,
    ) -> str:
        """Build user prompt for code synthesis."""
        template = self._prompt_config.get("synthesis_prompt_template", "")
        if not template:
            raise ValueError("synthesis_prompt_template missing in prompt config")

        # Format task spec details
        task_spec_summary = self._format_task_spec(task_spec)

        # Format hypothesis details
        hypothesis_summary = textwrap.dedent(
            f"""
            **Attack Type**: {hypothesis.attack_type}
            **Target Type**: {hypothesis.target_type}
            **Description**: {hypothesis.description}
            **Experiment Design**:
            {textwrap.indent(hypothesis.experiment_design, "  ")}
            **Confidence**: {hypothesis.confidence_score:.2f}
            **Novelty**: {hypothesis.novelty_score:.2f}
            """
        ).strip()

        # Build diffusers reference note
        diffusers_note = ""
        if DIFFUSERS_LIB_PATH.exists():
            diffusers_note = f"Reference implementation patterns from: {DIFFUSERS_LIB_PATH}"

        # Build output directory note
        output_dir_note = ""
        if run_output_dir:
            output_dir_note = textwrap.dedent(
                f"""
                **IMPORTANT**: Save all outputs (images, results, logs) to this directory:
                `{run_output_dir}`

                Use this exact path in your code for all file outputs.
                """
            ).strip()

        payload = {
            "task_spec": task_spec_summary,
            "hypothesis": hypothesis_summary,
            "diffusers_reference": diffusers_note,
            "output_directory": output_dir_note,
        }

        return template.format(**payload).strip()

    def _build_repair_prompt(
        self,
        *,
        previous_code: str,
        error_feedback: str,
        repair_attempt: int,
    ) -> str:
        """Build user prompt for code repair."""
        template = self._prompt_config.get("repair_prompt_template", "")
        if not template:
            raise ValueError("repair_prompt_template missing in prompt config")

        payload = {
            "error_feedback": error_feedback,
            "repair_attempt": repair_attempt,
        }

        return template.format(**payload).strip()

    def _create_chat_agent(self) -> ChatAgent:
        """Create chat agent for code synthesis and repair."""
        system_prompt = self._prompt_config.get("system_prompt", "")
        if not system_prompt:
            raise ValueError("system_prompt missing in prompt config")

        return ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name="CodeSynthesizer",
                content=system_prompt,
            ),
            model=self._model_backend,
        )

    def _create_backend(
        self,
        *,
        model_name: str,
        config: dict[str, Any],
        platform: ModelPlatformType,
        api_key: Optional[str],
        base_url: Optional[str],
    ):
        """Create model backend."""
        if isinstance(platform, str):
            model_platform = ModelPlatformType(platform.lower())
        else:
            model_platform = platform

        config = config or {}
        if model_platform == ModelPlatformType.OLLAMA:
            config_obj = OllamaConfig(**config)
        else:
            config_obj = ChatGPTConfig(**config)
        return ModelFactory.create(
            model_platform=model_platform,
            model_type=model_name,
            url=base_url,
            api_key=api_key,
            model_config_dict=config_obj.as_dict(),
        )

    def _load_prompt_config(self) -> dict[str, Any]:
        """Load prompt configuration."""
        path = PROMPTS_DIR / CODE_SYNTHESIZER_PROMPT
        if not path.exists():
            raise FileNotFoundError(f"Prompt config not found at {path}")

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return data

    @staticmethod
    def _extract_last_content(response) -> str:
        """Extract content from last message in response."""
        if not response or not getattr(response, "msgs", None):
            raise RuntimeError("LLM returned no messages")
        return response.msgs[-1].content or ""

    @staticmethod
    def _extract_code_from_response(content: str) -> str:
        """
        Extract Python code from LLM response.

        The response is expected to be JSON with a "code" field.
        Falls back to code fence extraction if JSON parsing fails.
        """
        # Try to parse as JSON first
        try:
            json_content = CodeSynthesizerAgent._strip_code_fences(content.strip())

            response_data = json.loads(json_content)

            # Extract code field
            if isinstance(response_data, dict) and "code" in response_data:
                return response_data["code"].strip()

            logger.warning("JSON response missing 'code' field, falling back to code fence extraction")

        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse JSON response: %s, falling back to code fence extraction", exc)

        # Fallback: Try to extract from Python code fence
        code_fence_pattern = r"```(?:python)?\s*(.*?)```"
        match = re.search(code_fence_pattern, content, flags=re.DOTALL)

        if match:
            return match.group(1).strip()

        # As a secondary fallback, strip any lingering fences from the content
        fence_stripped = CodeSynthesizerAgent._strip_code_fences(content)
        if fence_stripped != content:
            return fence_stripped.strip()

        # Last resort: treat entire content as code
        logger.warning("No code fence found, treating entire response as code")
        return content.strip()

    @staticmethod
    def _format_task_spec(task_spec: dict[str, Any]) -> str:
        """Format TaskSpec as readable summary."""
        lines = ["**TaskSpec Details:**"]
        for key in [
            "model_name",
            "model_version",
            "base_model_path",
            "unlearned_model_path",
            "unlearned_target",
            "unlearning_method",
        ]:
            value = task_spec.get(key)
            if value:
                if isinstance(value, str):
                    display_value = CodeSynthesizerAgent._normalize_display_value(value)
                else:
                    display_value = value
                lines.append(f"- {key}: {display_value}")
        return "\n".join(lines)

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove leading/trailing markdown code fences if present."""
        if not text:
            return text

        pattern = r"^```(?:json|python)?\s*(.*?)\s*```$"
        match = re.match(pattern, text.strip(), flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        return text

    @staticmethod
    def _normalize_display_value(value: str) -> str:
        """Normalize path-like values for display to the language model."""
        if not value or "://" in value:
            return value

        try:
            path = Path(value)
        except TypeError:
            return value

        # If the path is absolute and under the repository root, display it relative
        if path.is_absolute():
            try:
                repo_root = REPO_ROOT.resolve()
                relative = path.resolve().relative_to(repo_root)
                return relative.as_posix()
            except Exception:
                return path.as_posix()

        # For relative paths, ensure POSIX formatting
        return path.as_posix()

    @staticmethod
    def _contains_exception_signature(output: str) -> bool:
        """Detect if execution output contains signs of an unhandled exception."""
        if not output:
            return False

        lowered = output.lower()
        markers = (
            "traceback (most recent call last)",
            "error during execution",
            "fatal python error",
            "uncaught exception",
        )
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _extract_error_summary(stderr: str) -> str:
        """Extract brief error summary from stderr."""
        if not stderr:
            return "Unknown error"

        # Try to find last line with error
        lines = stderr.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if "Error:" in line or "Exception:" in line or "error" in line.lower():
                return line[:200]  # Truncate to 200 chars

        # Fallback: return last non-empty line
        for line in reversed(lines):
            line = line.strip()
            if line:
                return line[:200]

        return "Unknown error"

    @staticmethod
    def _extract_error_snippet(stderr: str) -> str:
        """Extract relevant error snippet (last 10 lines)."""
        if not stderr:
            return ""

        lines = stderr.strip().split("\n")
        snippet_lines = lines[-10:]  # Last 10 lines
        return "\n".join(snippet_lines)

    @staticmethod
    def _extract_traceback(stderr: str) -> Optional[str]:
        """Extract Python traceback if present."""
        if not stderr or "Traceback" not in stderr:
            return None

        # Find traceback section
        traceback_start = stderr.find("Traceback")
        if traceback_start == -1:
            return None

        return stderr[traceback_start:].strip()


__all__ = ["CodeSynthesizerAgent"]
