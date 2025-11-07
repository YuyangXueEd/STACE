"""
Long-Term Memory Agent for storing and retrieving successful attack discoveries.

This agent manages persistent memory of vulnerability discoveries across runs,
enabling the system to learn from past successes and avoid redundant exploration.
"""

import json
import re
from collections import Counter
from typing import Any, Optional
from datetime import datetime, timezone
from pathlib import Path

from aust.src.data_models import AttackMemoryCard, Hypothesis, IterationResult
from aust.src.utils.logging_config import get_logger
from aust.src.utils.markdown_parser import extract_markdown_sections
from aust.src.utils.model_config import load_model_settings

logger = get_logger(__name__)


class LongTermMemoryAgent:
    """
    Manages long-term memory of successful vulnerability discoveries.

    Stores attack memory cards as structured markdown files in:
    outputs/memory_store/attacks/{attack_id}.md
    """

    def __init__(self, memory_dir: Optional[Path] = None):
        """
        Initialize Long-Term Memory Agent.

        Args:
            memory_dir: Directory for storing attack memory cards.
                       Defaults to outputs/memory_store/attacks/
        """
        if memory_dir is None:
            # Default to outputs/memory_store/attacks/ at the project root
            project_root = Path(__file__).resolve().parents[3]
            memory_dir = project_root / "outputs" / "memory_store" / "attacks"

        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"LongTermMemoryAgent initialized with memory_dir: {self.memory_dir}")

    def create_attack_card(
        self,
        hypothesis: Hypothesis,
        iteration_result: IterationResult,
        task_spec: dict[str, Any],
        attack_trace_path: str,
    ) -> AttackMemoryCard:
        """
        Create an attack memory card from iteration results.

        Args:
            hypothesis: The successful hypothesis
            iteration_result: Results from the successful iteration
            task_spec: Task specification dict
            attack_trace_path: Path to attack trace markdown file

        Returns:
            AttackMemoryCard instance ready for storage
        """
        # Generate attack ID from task_id and iteration
        task_id = task_spec.get("task_id", "unknown")
        attack_id = f"{task_id}_iter{iteration_result.iteration_number}"

        # Extract experiment parameters
        experiment_results = iteration_result.experiment_results or {}
        experiment_parameters = {
            "prompts": experiment_results.get("prompts", []),
            "seeds": experiment_results.get("seeds", []),
            "num_images": experiment_results.get("num_images", 0),
            "guidance_scale": experiment_results.get("guidance_scale"),
            "num_steps": experiment_results.get("num_steps"),
        }

        # Parse successful prompts from evaluator feedback
        successful_prompts = self._extract_successful_prompts(
            iteration_result.evaluator_feedback or ""
        )

        # Extract key findings from evaluator feedback
        key_findings = self._extract_key_findings(
            iteration_result.evaluator_feedback or ""
        )

        card = AttackMemoryCard(
            attack_id=attack_id,
            task_type=task_spec.get("task_type", "unknown"),
            unlearned_target=task_spec.get("unlearned_target", "unknown"),
            unlearning_method=task_spec.get("unlearning_method"),
            model_name=task_spec.get("model_name"),
            hypothesis_attack_type=hypothesis.attack_type,
            hypothesis_summary=hypothesis.description,  # Use description field
            hypothesis_reasoning=hypothesis.experiment_design,  # Use experiment_design as reasoning
            hypothesis_full=hypothesis.model_dump(),
            experiment_parameters=experiment_parameters,
            detection_rate=iteration_result.vulnerability_confidence,  # Using as proxy
            max_confidence=iteration_result.vulnerability_confidence,
            vulnerability_confidence=iteration_result.vulnerability_confidence,
            successful_prompts=successful_prompts,
            key_findings=key_findings,
            attack_trace_path=attack_trace_path,
            iteration_number=iteration_result.iteration_number,
            discovered_at=datetime.now(timezone.utc),
        )

        logger.debug(f"Created attack memory card: {attack_id}")
        return card

    # ------------------------------------------------------------------ #
    # Long-term report generation (Story 5.2 AC4)
    # ------------------------------------------------------------------ #

    def generate_long_term_report(
        self,
        task_id: str,
        *,
        traces_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Generate a long-form academic report synthesising per-iteration traces.

        Args:
            task_id: Identifier for the task whose traces will be summarised.
            traces_dir: Directory containing ``attack_trace_iter_*.json`` files.
            output_dir: Destination directory for the generated report.

        Returns:
            Path to the generated markdown report, or ``None`` if generation fails.
        """
        project_root = Path(__file__).resolve().parents[3]

        if traces_dir is None:
            traces_dir = project_root / "outputs" / task_id / "attack_traces"

        iteration_traces = self._load_iteration_traces(traces_dir)
        if not iteration_traces:
            logger.warning("No iteration traces found in %s; skipping long-term report", traces_dir)
            return None

        prompts = self._load_long_report_prompts(project_root)

        attack_trace_str = json.dumps(iteration_traces, indent=2, ensure_ascii=False)
        system_prompt_template = prompts["system_prompt"]
        reports_agent = self._create_report_agent(system_prompt_template, attack_trace_str)

        sections = prompts.get("sections", {})
        report_parts = ["# Vulnerability Assessment Report\n"]

        if "introduction" in sections:
            intro_prompt = sections["introduction"]["prompt_template"]
            intro = self._generate_section_content(
                reports_agent,
                intro_prompt,
                section_name="introduction",
                iteration_traces=iteration_traces,
            )
            report_parts.append(f"\n## Introduction\n\n{intro}\n")

        if "generated attacking methods" in sections:
            methods_prompt = sections["generated attacking methods"]["prompt_template"]
            methods = self._generate_section_content(
                reports_agent,
                methods_prompt,
                section_name="generated attacking methods",
                iteration_traces=iteration_traces,
            )
            report_parts.append(f"\n## Generated Attacking Methods\n\n{methods}\n")

        if "summary" in sections:
            summary_prompt = sections["summary"]["prompt_template"]
            summary = self._generate_section_content(
                reports_agent,
                summary_prompt,
                section_name="summary",
                iteration_traces=iteration_traces,
            )
            report_parts.append(f"\n## Summary\n\n{summary}\n")

        if "discussion" in sections:
            discussion_prompt = sections["discussion"]["prompt_template"]
            discussion = self._generate_section_content(
                reports_agent,
                discussion_prompt,
                section_name="discussion",
                iteration_traces=iteration_traces,
            )
            report_parts.append(f"\n## Discussion\n\n{discussion}\n")

        full_report = "\n".join(report_parts)

        if output_dir is None:
            output_root = traces_dir.parent if traces_dir else project_root / "outputs"
        else:
            output_root = Path(output_dir)

        reports_dir = output_root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        report_path = reports_dir / f"long_term_report_{task_id}.md"
        report_path.write_text(full_report, encoding="utf-8")
        logger.info("Long-term report generated at %s", report_path)

        return report_path

    def _load_long_report_prompts(self, project_root: Path) -> dict[str, Any]:
        """Load report generation prompts from long_report_generator.yaml."""
        import yaml

        prompts_path = project_root / "aust" / "configs" / "prompts" / "long_report_generator.yaml"
        if not prompts_path.exists():
            raise FileNotFoundError(f"Report prompts not found: {prompts_path}")

        with prompts_path.open("r", encoding="utf-8") as handle:
            prompts = yaml.safe_load(handle) or {}

        if "system_prompt" not in prompts:
            raise ValueError("long_report_generator.yaml missing 'system_prompt'")

        logger.info("Loaded long-term report prompts from %s", prompts_path)
        return prompts

    def _load_iteration_traces(self, traces_dir: Path) -> list[dict[str, Any]]:
        """Read per-iteration trace JSON files from disk."""
        if not traces_dir.exists():
            return []

        traces: list[dict[str, Any]] = []
        for trace_file in sorted(traces_dir.glob("attack_trace_iter_*.json")):
            try:
                traces.append(json.loads(trace_file.read_text(encoding="utf-8")))
            except Exception as exc:
                logger.warning("Failed to load iteration trace %s: %s", trace_file, exc)
        return traces

    def _create_report_agent(self, system_prompt_template: str, attack_trace: str) -> Any:
        """Instantiate CAMEL ChatAgent for long-term report generation."""
        import os
        from camel.agents import ChatAgent
        from camel.configs import ChatGPTConfig
        from camel.messages import BaseMessage
        from camel.models import ModelFactory
        from camel.types import ModelPlatformType

        system_prompt = system_prompt_template.format(attack_trace=attack_trace)

        fallback = {
            "model_name": "openai/gpt-5-nano",
            "config": {
                "temperature": 0.7,
                "max_tokens": 128_000,
            },
        }
        settings = load_model_settings("long_term_memory_report", fallback)
        model_name = settings["model_name"]
        config_dict = settings.get("config", {})

        platform_value = settings.get("model_platform")
        if isinstance(platform_value, ModelPlatformType):
            platform = platform_value
        elif platform_value is not None:
            platform = ModelPlatformType(str(platform_value).lower())
        else:
            platform = ModelPlatformType.OPENAI_COMPATIBLE_MODEL

        api_key_env = settings.get("api_key_env_var") or "OPENROUTER_API_KEY"
        api_key = os.getenv(api_key_env)

        config = ChatGPTConfig(**config_dict)

        backend = ModelFactory.create(
            model_platform=platform,
            model_type=model_name,
            url=settings.get("base_url") or "https://openrouter.ai/api/v1",
            api_key=api_key,
            model_config_dict=config.as_dict(),
        )

        system_message = BaseMessage.make_assistant_message(
            role_name="LongTermMemoryReporter",
            content=system_prompt,
        )
        return ChatAgent(system_message=system_message, model=backend)

    def _generate_section_content(
        self,
        agent: Any,
        section_prompt: str,
        *,
        section_name: str,
        iteration_traces: list[dict[str, Any]],
    ) -> str:
        """Generate a single section of the long-term report."""
        from camel.messages import BaseMessage

        try:
            user_message = BaseMessage.make_user_message(
                role_name="User",
                content=section_prompt.strip(),
            )
            response = agent.step(user_message)
            if not response or not getattr(response, "msgs", None):
                info = getattr(response, "info", {})
                logger.warning(
                    "Empty response from LLM for section '%s' (info=%s); using fallback content",
                    section_name,
                    info or "<none>",
                )
                return self._build_fallback_section(section_name, iteration_traces)

            content = response.msgs[-1].content.strip()
            if not content:
                logger.warning(
                    "Received blank content from LLM for section '%s'; using fallback content",
                    section_name,
                )
                return self._build_fallback_section(section_name, iteration_traces)

            return content
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("Failed to generate section content: %s", exc, exc_info=True)
            fallback = self._build_fallback_section(section_name, iteration_traces)
            if fallback:
                logger.info(
                    "Generated fallback content for section '%s' after LLM error.",
                    section_name,
                )
                return fallback
            return f"Error generating content: {exc}"

    def _build_fallback_section(
        self,
        section_name: str,
        iteration_traces: list[dict[str, Any]],
    ) -> str:
        """Construct deterministic fallback content when LLM generation fails."""
        stats = self._collect_iteration_statistics(iteration_traces)
        key = section_name.lower()

        if key == "introduction":
            return self._fallback_introduction(stats)
        if key == "generated attacking methods":
            return self._fallback_methods(stats)
        if key == "summary":
            return self._fallback_summary(stats)
        if key == "discussion":
            return self._fallback_discussion(stats)

        return self._fallback_generic(stats)

    def _collect_iteration_statistics(
        self,
        iteration_traces: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Aggregate useful metadata from iteration traces for fallback content."""
        if not iteration_traces:
            return {
                "task": {},
                "iterations": [],
                "total_iterations": 0,
                "success_count": 0,
                "highest_confidence": 0.0,
                "average_debate_rounds": 0.0,
                "total_images": 0,
                "attack_type_counts": Counter(),
                "key_learnings": [],
                "vulnerability_found": False,
            }

        first = iteration_traces[0]
        task_info = {
            "task_id": first.get("task_id"),
            "task_type": first.get("task_type"),
            "task_description": first.get("task_description"),
        }
        spec = first.get("task_spec") or {}
        task_info.update(
            {
                "model_name": spec.get("model_name"),
                "model_version": spec.get("model_version"),
                "unlearning_method": spec.get("unlearning_method"),
                "unlearned_target": spec.get("unlearned_target"),
            }
        )

        iterations: list[dict[str, Any]] = []
        success_count = 0
        highest_confidence = 0.0
        total_rounds = 0.0
        rounds_count = 0
        total_images = 0
        attack_types: list[str] = []
        key_learnings: list[str] = []

        for trace in iteration_traces:
            iteration_data = trace.get("iteration", {})
            hypothesis = iteration_data.get("hypothesis") or {}
            attempts = iteration_data.get("attempts") or []

            confidence = self._safe_float(iteration_data.get("confidence"))
            highest_confidence = max(highest_confidence, confidence)

            vulnerability_detected = bool(iteration_data.get("vulnerability_detected"))
            if vulnerability_detected:
                success_count += 1

            debate_session = iteration_data.get("debate_session") or {}
            rounds = debate_session.get("total_rounds")
            if isinstance(rounds, (int, float)):
                total_rounds += float(rounds)
                rounds_count += 1

            debate_quality = debate_session.get("final_quality_score")
            debate_summary = iteration_data.get("debate_narrative")

            images_generated = 0
            evaluator_feedback_excerpt = None
            if attempts:
                # Sum images across attempts for completeness
                for attempt in attempts:
                    if not isinstance(attempt, dict):
                        continue
                    images_generated += int(attempt.get("images_generated") or 0)
                    feedback = attempt.get("evaluator_feedback")
                    if feedback and not evaluator_feedback_excerpt:
                        evaluator_feedback_excerpt = feedback

            total_images += images_generated

            key_learning = iteration_data.get("key_learning")
            if key_learning:
                key_learnings.append(key_learning)

            attack_type = hypothesis.get("attack_type")
            if attack_type:
                attack_types.append(attack_type)

            iterations.append(
                {
                    "iteration_number": iteration_data.get("iteration_number"),
                    "attack_type": attack_type,
                    "target_type": hypothesis.get("target_type"),
                    "description": hypothesis.get("description"),
                    "experiment_design": hypothesis.get("experiment_design"),
                    "vulnerability_detected": vulnerability_detected,
                    "confidence": confidence,
                    "attempts": attempts,
                    "images_generated": images_generated,
                    "evaluator_feedback": evaluator_feedback_excerpt,
                    "key_learning": key_learning,
                    "outcome": iteration_data.get("outcome_summary")
                    or iteration_data.get("final_status"),
                    "debate_rounds": rounds,
                    "debate_quality": debate_quality,
                    "debate_summary": debate_summary,
                }
            )

        attack_type_counts = Counter(attack_types)
        deduped_learnings = self._deduplicate_preserve_order(
            [self._truncate_text(text, limit=400) for text in key_learnings if text]
        )

        return {
            "task": task_info,
            "iterations": iterations,
            "total_iterations": len(iterations),
            "success_count": success_count,
            "highest_confidence": highest_confidence,
            "average_debate_rounds": (total_rounds / rounds_count) if rounds_count else 0.0,
            "total_images": total_images,
            "attack_type_counts": attack_type_counts,
            "key_learnings": deduped_learnings,
            "vulnerability_found": success_count > 0,
        }

    def _fallback_introduction(self, stats: dict[str, Any]) -> str:
        """Fallback introduction when LLM output is unavailable."""
        task = stats.get("task", {})
        target = task.get("unlearned_target") or "the specified concept"
        model_name = task.get("model_name") or "the target model"
        model_version = task.get("model_version")
        model_label = f"{model_name} v{model_version}" if model_version else model_name
        task_type = task.get("task_type") or "concept erasure"
        method = task.get("unlearning_method") or "an unspecified unlearning method"
        description = task.get("task_description") or "No task description provided."

        total_iterations = stats.get("total_iterations", 0)
        success_count = stats.get("success_count", 0)
        highest_confidence = stats.get("highest_confidence", 0.0)
        avg_rounds = stats.get("average_debate_rounds", 0.0)
        total_images = stats.get("total_images", 0)

        lines = [
            f"This assessment evaluates how well {model_label} retains or forgets the concept '{target}' under the {task_type} scenario using {method}.",
            f"The automated research loop generated hypotheses, debated refinements, and executed experiments across {total_iterations} iteration(s).",
            description,
        ]

        if success_count:
            lines.append(
                f"{success_count} iteration(s) surfaced a potential vulnerability, peaking at a confidence of {highest_confidence:.1%}."
            )
        else:
            lines.append(
                f"No high-confidence vulnerabilities were confirmed. The highest observed confidence score was {highest_confidence:.1%}."
            )

        if avg_rounds:
            lines.append(
                f"Multi-agent debate averaged {avg_rounds:.1f} round(s) per iteration, providing structured critique before experiments executed."
            )

        if total_images:
            lines.append(
                f"Experiment execution rendered {total_images} image(s) in total, supplying evaluators with qualitative evidence."
            )

        lines.append(
            "The following sections document the generated attack strategies, evaluation outcomes, and lessons for future red-teaming."
        )

        return "\n\n".join(lines)

    def _fallback_methods(self, stats: dict[str, Any]) -> str:
        """Fallback description of generated attacking methods."""
        if not stats.get("iterations"):
            return "No iteration data was captured; attacking methods cannot be summarised."

        sections: list[str] = []
        for entry in stats["iterations"]:
            header = f"### Iteration {entry.get('iteration_number')}: {entry.get('attack_type') or 'Unnamed attack'}"
            description = self._truncate_text(entry.get("description"), limit=600) or "No description recorded."
            experiment_design = self._truncate_text(entry.get("experiment_design"), limit=500)
            debate_summary = entry.get("debate_summary")
            debate_rounds = entry.get("debate_rounds")
            debate_quality = entry.get("debate_quality")
            evaluator_feedback = self._truncate_text(entry.get("evaluator_feedback"), limit=600)
            key_learning = self._truncate_text(entry.get("key_learning"), limit=300)

            execution_details = []
            if entry.get("images_generated"):
                execution_details.append(f"{entry['images_generated']} image(s) rendered")
            if entry.get("outcome"):
                execution_details.append(f"Outcome: {entry['outcome']}")
            execution_summary = "; ".join(execution_details) if execution_details else "Execution details were not recorded."

            result_line = (
                f"Result: {'Vulnerability detected' if entry.get('vulnerability_detected') else 'No vulnerability detected'} "
                f"(confidence {entry.get('confidence', 0.0):.1%})."
            )

            chunk_parts = [
                header,
                description,
            ]

            if experiment_design:
                chunk_parts.append(f"**Experiment Plan:** {experiment_design}")

            if debate_summary:
                chunk_parts.append("**Debate Highlights:**\n" + debate_summary)
            elif debate_rounds is not None:
                quality_text = (
                    f" (final quality score {debate_quality:.2f})"
                    if isinstance(debate_quality, (int, float))
                    else ""
                )
                chunk_parts.append(
                    f"**Debate Highlights:** {debate_rounds} round(s){quality_text} prior to execution."
                )

            chunk_parts.append(f"**Execution:** {execution_summary}")
            chunk_parts.append(result_line)

            if evaluator_feedback:
                chunk_parts.append("**Evaluator Feedback (excerpt):**\n" + evaluator_feedback)

            if key_learning:
                chunk_parts.append(f"**Key Learning:** {key_learning}")

            sections.append("\n\n".join(chunk_parts))

        return "\n\n".join(sections)

    def _fallback_summary(self, stats: dict[str, Any]) -> str:
        """Fallback summary section leveraging structured trace data."""
        iterations = stats.get("iterations", [])
        total_iterations = stats.get("total_iterations", 0)
        success_count = stats.get("success_count", 0)
        highest_confidence = stats.get("highest_confidence", 0.0)
        attack_type_counts: Counter = stats.get("attack_type_counts", Counter())
        key_learnings = stats.get("key_learnings", [])

        top_iteration = max(iterations, key=lambda item: item.get("confidence", 0.0), default=None)

        lines = [
            f"The assessment completed {total_iterations} iteration(s); {success_count} yielded a confirmed vulnerability signal.",
            f"The highest confidence recorded was {highest_confidence:.1%}.",
        ]

        if top_iteration:
            lines.append(
                "Peak confidence was observed in iteration "
                f"{top_iteration.get('iteration_number')} "
                f"({top_iteration.get('attack_type') or 'Unnamed attack'}), "
                f"which concluded with {top_iteration.get('confidence', 0.0):.1%} confidence."
            )

        if not success_count:
            lines.append(
                "Despite extensive probing, no attack breached the configured detection threshold, "
                "suggesting the unlearning procedure resisted the explored vectors."
            )

        if attack_type_counts:
            top_attacks = ", ".join(
                f"{attack} ({count})" for attack, count in attack_type_counts.most_common(5)
            )
            lines.append(f"Attack families explored included: {top_attacks}.")

        if key_learnings:
            lines.append("Key learnings captured during the campaign:")
            for learning in key_learnings[:5]:
                lines.append(f"- {learning}")

        return "\n\n".join(lines)

    def _fallback_discussion(self, stats: dict[str, Any]) -> str:
        """Fallback discussion section drawing on recorded trace metadata."""
        attack_type_counts: Counter = stats.get("attack_type_counts", Counter())
        key_learnings = stats.get("key_learnings", [])
        vulnerability_found = stats.get("vulnerability_found", False)
        avg_rounds = stats.get("average_debate_rounds", 0.0)

        discussion_points: list[str] = []

        if attack_type_counts:
            attack_catalogue = ", ".join(
                f"{attack} ({count})" for attack, count in attack_type_counts.most_common(3)
            )
            discussion_points.append(
                f"The exploration emphasised {attack_catalogue}, highlighting sustained interest in compositional prompt attacks and multilingual cues."
            )

        if vulnerability_found:
            discussion_points.append(
                "Detected vulnerabilities indicate that certain attack patterns still elicit leakage, warranting deeper ablation and defensive hardening."
            )
        else:
            discussion_points.append(
                "No high-confidence leakage was observed, but the absence of evidence does not prove complete erasure; further modalities (e.g., video or audio cues) should be considered."
            )

        if avg_rounds:
            discussion_points.append(
                f"Debate averaged {avg_rounds:.1f} round(s), which helped converge on reproducible prompts even when evaluators reported negative findings."
            )

        if key_learnings:
            discussion_points.append(
                "Representative insights captured during evaluation included:\n"
                + "\n".join(f"- {learning}" for learning in key_learnings[:3])
            )

        discussion_points.append(
            "Future work should broaden the search space with adaptive repair loops, richer perceptual metrics, and comparison against alternative unlearning baselines documented in recent literature."
        )

        return "\n\n".join(discussion_points)

    def _fallback_generic(self, stats: dict[str, Any]) -> str:
        """Generic fallback when section name is unrecognised."""
        total_iterations = stats.get("total_iterations", 0)
        return (
            f"The system executed {total_iterations} iteration(s), but no tailored narrative is available for this section."
        )

    @staticmethod
    def _truncate_text(text: Optional[str], limit: int = 600) -> Optional[str]:
        """Trim long text segments for fallback content."""
        if not text:
            return None
        stripped = text.strip()
        if len(stripped) <= limit:
            return stripped
        return stripped[:limit].rstrip() + "..."

    @staticmethod
    def _deduplicate_preserve_order(items: list[str]) -> list[str]:
        """Remove duplicates while preserving insertion order."""
        seen = set()
        result = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Convert value to float with graceful fallback."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _extract_successful_prompts(self, evaluator_feedback: str) -> list[dict[str, Any]]:
        """Extract successful prompts from evaluator feedback."""
        successful_prompts = []

        # Look for "Successful Prompts" section in feedback
        prompts_match = re.search(
            r"Successful Prompts:.*?\n(.*?)(?=\n\n|\Z)",
            evaluator_feedback,
            re.DOTALL,
        )

        if prompts_match:
            prompts_text = prompts_match.group(1)
            # Parse lines like: • 'kitten' (100.0%)
            for line in prompts_text.split("\n"):
                match = re.search(r"['\"](.*?)['\"].*?\((\d+\.?\d*)%\)", line)
                if match:
                    prompt = match.group(1)
                    rate_str = match.group(2)
                    successful_prompts.append(
                        {
                            "prompt": prompt,
                            "detection_rate": float(rate_str) / 100.0,
                        }
                    )

        return successful_prompts

    def _extract_key_findings(self, evaluator_feedback: str) -> list[str]:
        """Extract key findings from evaluator feedback."""
        findings = []

        # Extract vulnerability detected message as primary finding
        if "Vulnerability Detected:" in evaluator_feedback:
            # Extract the vulnerability message
            msg_match = re.search(
                r"⚠️ Vulnerability Detected: (.+?)(?=\n\n|\Z)",
                evaluator_feedback,
                re.DOTALL,
            )
            if msg_match:
                findings.append(msg_match.group(1).strip())

        # Add detection rate as finding
        rate_match = re.search(r"Concept Detected: \d+/\d+ \((\d+\.?\d*)%\)", evaluator_feedback)
        if rate_match:
            detection_rate = rate_match.group(1)
            findings.append(
                f"Detection rate: {detection_rate}% of generated images contained the concept"
            )

        # Add max confidence as finding
        conf_match = re.search(r"Max Confidence: (\d+\.?\d+)", evaluator_feedback)
        if conf_match:
            max_conf = conf_match.group(1)
            findings.append(
                f"Maximum detection confidence: {max_conf} (high confidence indicates clear leakage)"
            )

        return findings

    def store_attack(self, card: AttackMemoryCard) -> Path:
        """
        Store attack memory card to disk.

        Args:
            card: AttackMemoryCard to store

        Returns:
            Path to stored markdown file
        """
        filepath = self.memory_dir / f"{card.attack_id}.md"

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(card.to_markdown())

            logger.info(f"Stored attack memory card: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to store attack memory card {card.attack_id}: {e}")
            raise

    def get_attacks_by_task_type(
        self,
        task_type: str,
        min_confidence: float = 0.0,
        max_results: Optional[int] = None,
    ) -> list[AttackMemoryCard]:
        """
        Retrieve attack memory cards filtered by task type.

        Args:
            task_type: Filter by task type (concept_erasure, data_based_unlearning)
            min_confidence: Minimum vulnerability confidence (0.0-1.0)
            max_results: Maximum number of results to return (most recent first)

        Returns:
            List of AttackMemoryCard instances matching criteria
        """
        logger.debug(
            f"Querying memory: task_type={task_type}, min_confidence={min_confidence}"
        )

        attacks = []

        # Scan all markdown files in memory directory
        for filepath in self.memory_dir.glob("*.md"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    markdown_content = f.read()

                # Parse attack card
                attack_id = filepath.stem
                card = AttackMemoryCard.from_markdown(markdown_content, attack_id)

                # Apply filters
                if card.task_type == task_type and card.vulnerability_confidence >= min_confidence:
                    attacks.append(card)

            except Exception as e:
                logger.warning(f"Failed to parse attack memory card {filepath}: {e}")
                continue

        # Sort by discovery time (most recent first)
        attacks.sort(key=lambda x: x.discovered_at, reverse=True)

        # Limit results
        if max_results is not None:
            attacks = attacks[:max_results]

        logger.info(f"Retrieved {len(attacks)} attack memory cards")
        return attacks

    def get_all_attacks(self) -> list[AttackMemoryCard]:
        """
        Retrieve all stored attack memory cards.

        Returns:
            List of all AttackMemoryCard instances
        """
        return self.get_attacks_by_task_type(
            task_type="",  # Empty string matches nothing, but we override in loop
            min_confidence=0.0,
            max_results=None,
        )

    def get_memory_summary_for_context(
        self,
        task_type: str,
        min_confidence: float = 0.5,
        max_entries: int = 5,
    ) -> list[dict[str, str]]:
        """
        Get summarized memory entries for hypothesis context.

        Extracts only relevant sections (METADATA, HYPOTHESIS, RESULTS, LESSONS_LEARNED)
        to minimize token usage.

        Args:
            task_type: Filter by task type
            min_confidence: Minimum confidence threshold
            max_entries: Maximum number of entries to return

        Returns:
            List of dicts with attack_id and summary (markdown excerpt)
        """
        attacks = self.get_attacks_by_task_type(
            task_type=task_type,
            min_confidence=min_confidence,
            max_results=max_entries,
        )

        memory_entries = []

        for attack in attacks:
            # Generate full markdown
            markdown_content = attack.to_markdown()

            # Extract only relevant sections
            summary = extract_markdown_sections(
                markdown_content,
                sections_to_extract=["METADATA", "HYPOTHESIS", "RESULTS", "LESSONS_LEARNED"],
            )

            memory_entries.append(
                {
                    "attack_id": attack.attack_id,
                    "summary": summary,
                }
            )

        logger.debug(f"Generated {len(memory_entries)} memory summaries for context")
        return memory_entries


__all__ = ["LongTermMemoryAgent"]
