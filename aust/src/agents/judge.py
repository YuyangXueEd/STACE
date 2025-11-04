"""Judge agent workforce producing persona-driven evaluations."""

from __future__ import annotations

import json
import os
import re
import textwrap
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml
from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from dotenv import load_dotenv

from aust.src.data_models.judge import CommitteeAggregate, JudgeEvaluation, JudgeScore
from aust.src.data_models.report import AcademicReport
from aust.src.utils.logging_config import get_logger
from aust.src.utils.model_config import load_model_settings

load_dotenv()
logger = get_logger(__name__)

LLMRunner = Callable[[dict[str, Any], str, str], str]

_MODEL_FALLBACK = {
    "model_name": "openai/gpt-5-nano",
    "config": {
        "temperature": 0.2,
        "max_tokens": 900,
        "top_p": 0.9,
    },
}


class JudgeAgent:
    """Executes persona-specific report evaluations and aggregates committee output."""

    def __init__(
        self,
        *,
        persona_config_path: Path | None = None,
        output_dir: Path | None = None,
        model_config_name: str = "judge",
        llm_runner: LLMRunner | None = None,
    ) -> None:
        self.persona_config_path = persona_config_path or self._default_persona_path()
        self.personas = self._load_personas(self.persona_config_path)
        self.output_dir = output_dir or self._default_output_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_config_name = model_config_name
        self._llm_runner = llm_runner or self._build_llm_runner()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate(
        self,
        persona_id: str,
        *,
        report: AcademicReport | str,
        attack_trace: Mapping[str, Any] | str | Path | None = None,
        experiment_results: Mapping[str, Any] | None = None,
        run_id: str,
    ) -> JudgeEvaluation:
        """Produce a structured evaluation for the specified persona."""

        persona = self._get_persona(persona_id)
        system_prompt = self._build_system_prompt(persona)
        user_prompt = self._build_user_prompt(
            report=report,
            attack_trace=attack_trace,
            experiment_results=experiment_results,
        )
        try:
            raw_response = self._llm_runner(persona, system_prompt, user_prompt)
        except Exception as exc:  # pragma: no cover - network/model failure
            logger.error(
                "Judge persona %s failed to generate a response: %s",
                persona_id,
                exc,
                exc_info=True,
            )
            evaluation = self._build_failure_evaluation(
                persona=persona,
                run_id=run_id,
                error=exc,
                raw_response=None,
            )
        else:
            try:
                payload = self._parse_response(raw_response)
                evaluation = self._hydrate_evaluation(persona, payload, raw_response)
            except Exception as exc:
                logger.error(
                    "Judge persona %s returned an unparsable response: %s",
                    persona_id,
                    exc,
                    exc_info=True,
                )
                evaluation = self._build_failure_evaluation(
                    persona=persona,
                    run_id=run_id,
                    error=exc,
                    raw_response=raw_response,
                )

        report_path = self._write_persona_report(evaluation, persona, run_id)
        logger.info("Judge %s wrote evaluation to %s", persona["name"], report_path)
        return evaluation

    def run_committee(
        self,
        *,
        report: AcademicReport | str,
        attack_trace: Mapping[str, Any] | str | Path | None = None,
        experiment_results: Mapping[str, Any] | None = None,
        run_id: str,
    ) -> CommitteeAggregate:
        """Run the full judge workforce and produce an aggregate summary."""

        evaluations: list[JudgeEvaluation] = []
        for persona in self.personas.values():
            evaluation = self.evaluate(
                persona_id=persona["id"],
                report=report,
                attack_trace=attack_trace,
                experiment_results=experiment_results,
                run_id=run_id,
            )
            evaluations.append(evaluation)

        aggregate = CommitteeAggregate(run_id=run_id, persona_evaluations=evaluations)
        summary_path = self._write_committee_summary(aggregate)
        logger.info("Committee summary written to %s", summary_path)
        return aggregate

    # ------------------------------------------------------------------
    # Persona configuration helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _default_persona_path() -> Path:
        base_dir = Path(__file__).resolve().parents[2] / "configs"
        new_path = base_dir / "personas" / "judges.yaml"
        if new_path.exists():
            return new_path
        # Legacy fallback kept for backward compatibility with Story 4.4 drafts
        return base_dir / "judge_personas.yaml"

    @staticmethod
    def _default_output_dir() -> Path:
        return Path(__file__).resolve().parents[2] / "outputs" / "judgments"

    def _load_personas(self, path: Path) -> dict[str, dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Judge persona config not found at {path}")

        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        raw_personas = data.get("personas")
        if not isinstance(raw_personas, list) or not raw_personas:
            raise ValueError("Persona configuration must provide a non-empty 'personas' list")

        personas: dict[str, dict[str, Any]] = {}
        for entry in raw_personas:
            if not isinstance(entry, dict):
                continue
            identifier = entry.get("id") or entry.get("name")
            persona_id = str(identifier or "").strip()
            if not persona_id:
                raise ValueError("Each persona requires an 'id'")
            hydrated = deepcopy(entry)
            hydrated["id"] = persona_id
            hydrated.setdefault("name", persona_id.replace("_", " ").title())
            hydrated.setdefault("evaluation_criteria", [])
            hydrated.setdefault("scoring_dimensions", [])
            personas[persona_id] = hydrated

        logger.info("Loaded %d judge personas", len(personas))
        return personas

    def _get_persona(self, persona_id: str) -> dict[str, Any]:
        try:
            return self.personas[persona_id]
        except KeyError as exc:  # pragma: no cover - defensive
            available = ", ".join(sorted(self.personas))
            raise KeyError(f"Unknown persona '{persona_id}'. Available: {available}") from exc

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def _build_system_prompt(self, persona: Mapping[str, Any]) -> str:
        name = persona.get("name", persona["id"])  # type: ignore[index]
        role_description = persona.get("role_description", "")
        tone = persona.get("tone", "")
        focus_areas = persona.get("focus_areas") or []
        evaluation_criteria = persona.get("evaluation_criteria") or []
        scoring_dimensions = persona.get("scoring_dimensions") or []
        prompt_template = persona.get("prompt_template")

        focus_block = "\n".join(f"- {item}" for item in focus_areas)
        criteria_block = "\n".join(f"- {item}" for item in evaluation_criteria)
        dimensions_block = "\n".join(
            f"- {dimension.get('name')}: {dimension.get('description')} ({dimension.get('scale')})"
            for dimension in scoring_dimensions
            if isinstance(dimension, Mapping)
        )

        base_prompt = textwrap.dedent(
            f"""
            You are {name}. {role_description}

            Tone: {tone}

            Focus areas:
            {focus_block}

            Evaluation criteria:
            {criteria_block}

            Scoring dimensions:
            {dimensions_block}

            Respond with a JSON object containing:
              - summary: string
              - strengths: list of strings
              - weaknesses: list of strings
              - recommendations: list of strings
              - scores: list of objects with keys dimension, value, scale, justification
              - overall_rating: optional float on a 0-5 scale (use persona-specific judgement)

            Justify every score briefly. If information is missing, note assumptions
            explicitly. Keep responses concise:
              - summary <= 80 words
              - each list item <= 30 words, maximum 3 items per list
              - score justification <= 25 words
            Return a single JSON object with no code fences or additional commentary.
            """
        ).strip()

        if isinstance(prompt_template, str) and prompt_template.strip():
            base_prompt += "\n\n" + prompt_template.strip()

        return base_prompt

    def _build_user_prompt(
        self,
        *,
        report: AcademicReport | str,
        attack_trace: Mapping[str, Any] | str | Path | None,
        experiment_results: Mapping[str, Any] | None,
    ) -> str:
        report_md = report if isinstance(report, str) else report.to_markdown()
        attack_trace_block = self._format_optional_block("Attack Trace", attack_trace)
        experiments_block = self._format_optional_block("Experiment Results", experiment_results)

        return textwrap.dedent(
            f"""
            Evaluate the following machine unlearning vulnerability report. Operate strictly within
            your persona's focus area and scoring rubric.

            ## Report (Markdown)
            {report_md}

            {attack_trace_block}

            {experiments_block}
            """
        ).strip()

    def _format_optional_block(
        self,
        title: str,
        payload: Mapping[str, Any] | str | Path | None,
        *,
        max_chars: int = 6000,
    ) -> str:
        if payload is None:
            return f"## {title}\nNot provided."

        if isinstance(payload, (str, Path)):
            text = Path(payload).read_text(encoding="utf-8") if isinstance(payload, Path) else payload
        else:
            text = json.dumps(payload, indent=2, ensure_ascii=False)

        text = text.strip()
        if len(text) > max_chars:
            text = text[: max_chars - 3] + "..."

        return f"## {title}\n{text}"

    # ------------------------------------------------------------------
    # LLM runner
    # ------------------------------------------------------------------
    def _build_llm_runner(self) -> LLMRunner:
        settings = load_model_settings(self.model_config_name, _MODEL_FALLBACK, logger=logger)
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENROUTER_API_KEY is required to initialise JudgeAgent")

        base_model_name = settings["model_name"]
        base_config_dict = deepcopy(settings.get("config", {}))
        base_config = ChatGPTConfig(**base_config_dict)

        model_cache: dict[Any, Any] = {}

        def _resolve_model_for_persona(persona: Mapping[str, Any]):
            persona_model_name = persona.get("model_name")
            persona_config = persona.get("model_config")

            if persona_model_name or isinstance(persona_config, Mapping):
                resolved_name = str(persona_model_name or base_model_name)
                resolved_config = deepcopy(base_config_dict)
                if isinstance(persona_config, Mapping):
                    for key, value in persona_config.items():
                        if value is not None:
                            resolved_config[key] = value
                cache_key = (resolved_name, tuple(sorted(resolved_config.items())))
                if cache_key not in model_cache:
                    persona_cfg = ChatGPTConfig(**resolved_config)
                    model_cache[cache_key] = ModelFactory.create(
                        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
                        model_type=resolved_name,
                        url="https://openrouter.ai/api/v1",
                        api_key=api_key,
                        model_config_dict=persona_cfg.as_dict(),
                    )
                return model_cache[cache_key]

            default_key = (base_model_name, tuple(sorted(base_config_dict.items())))
            if default_key not in model_cache:
                model_cache[default_key] = ModelFactory.create(
                    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
                    model_type=base_model_name,
                    url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                    model_config_dict=base_config.as_dict(),
                )
            return model_cache[default_key]

        def _runner(persona: dict[str, Any], system_prompt: str, user_prompt: str) -> str:
            backend_model = _resolve_model_for_persona(persona)
            agent = ChatAgent(
                system_message=BaseMessage.make_assistant_message(
                    role_name=f"{persona['name']}Judge",
                    content=system_prompt,
                ),
                model=backend_model,
            )
            try:
                response = agent.step(
                    BaseMessage.make_user_message("ReportOrchestrator", user_prompt)
                )
            finally:
                agent.reset()

            if not getattr(response, "msgs", None):
                raise RuntimeError("Judge model returned no messages")
            content = response.msgs[-1].content
            if content is None:
                raise RuntimeError("Judge model response lacked content")
            return content

        return _runner

    # ------------------------------------------------------------------
    # Response parsing and hydration
    # ------------------------------------------------------------------
    def _parse_response(self, raw: str) -> dict[str, Any]:
        text = (raw or "").strip()
        if not text:
            raise ValueError("Judge response was empty")

        candidates = [text]
        if text.startswith("```"):
            candidates.append(self._strip_code_fence(text))
        candidates.append(self._extract_json_substring(text))

        for candidate in candidates:
            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                continue

        raise ValueError("Unable to parse judge response as JSON")

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
        return match.group(1).strip() if match else text

    @staticmethod
    def _extract_json_substring(text: str) -> str:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return text
        return match.group(0)

    def _hydrate_evaluation(
        self,
        persona: Mapping[str, Any],
        payload: Mapping[str, Any],
        raw_response: str,
    ) -> JudgeEvaluation:
        scores_payload = payload.get("scores") or []
        scores: list[JudgeScore] = []
        for entry in scores_payload:
            if not isinstance(entry, Mapping):
                continue
            dimension = str(entry.get("dimension") or "").strip()
            if not dimension:
                continue
            try:
                value = float(entry.get("value"))
            except (TypeError, ValueError):
                continue
            scale = str(entry.get("scale") or "1-5").strip()
            justification = str(entry.get("justification") or "").strip()
            scores.append(
                JudgeScore(
                    dimension=dimension,
                    value=value,
                    scale=scale,
                    justification=justification,
                )
            )

        strengths = self._string_list(payload.get("strengths"))
        weaknesses = self._string_list(payload.get("weaknesses"))
        recommendations = self._string_list(payload.get("recommendations"))

        summary = str(payload.get("summary") or "").strip()
        if not summary:
            raise ValueError("Judge summary is required in the response")

        overall_rating = payload.get("overall_rating")
        if overall_rating is not None:
            try:
                overall_rating = float(overall_rating)
            except (TypeError, ValueError):
                overall_rating = None

        return JudgeEvaluation(
            persona_id=str(persona["id"]),
            persona_name=str(persona.get("name", persona["id"])),
            summary=summary,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            scores=scores,
            overall_rating=overall_rating,
            raw_response=raw_response,
        )

    def _build_failure_evaluation(
        self,
        *,
        persona: Mapping[str, Any],
        run_id: str,
        error: Exception | str,
        raw_response: str | None,
    ) -> JudgeEvaluation:
        message = str(error) or "Unknown error"
        summary = f"Evaluation failed for run {run_id}: {message}"
        return JudgeEvaluation(
            persona_id=str(persona["id"]),
            persona_name=str(persona.get("name", persona["id"])),
            summary=summary,
            strengths=[],
            weaknesses=[summary],
            recommendations=[
                "Re-run the judge evaluation after addressing the persona response formatting issue."
            ],
            scores=[],
            overall_rating=None,
            raw_response=raw_response,
        )

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if isinstance(value, str):
            normalized = value.strip()
            return [normalized] if normalized else []
        if isinstance(value, list):
            results: list[str] = []
            for item in value:
                text = str(item).strip()
                if text:
                    results.append(text)
            return results
        return []

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _write_persona_report(
        self,
        evaluation: JudgeEvaluation,
        persona: Mapping[str, Any],
        run_id: str,
    ) -> Path:
        filename = f"judge_{self._slugify(evaluation.persona_name)}_{run_id}.md"
        path = self.output_dir / filename

        lines = [
            f"# Judge Evaluation — {evaluation.persona_name}",
            "",
            f"**Persona ID**: {evaluation.persona_id}",
            f"**Run ID**: {run_id}",
        ]

        focus_areas = persona.get("focus_areas") or []
        if focus_areas:
            lines.append("**Focus Areas**:")
            for item in focus_areas:
                lines.append(f"- {item}")

        lines.extend(
            [
                "",
                "## Summary",
                evaluation.summary,
                "",
                "## Strengths",
            ]
        )

        if evaluation.strengths:
            for item in evaluation.strengths:
                lines.append(f"- {item}")
        else:
            lines.append("- None highlighted")

        lines.extend(["", "## Weaknesses"])
        if evaluation.weaknesses:
            for item in evaluation.weaknesses:
                lines.append(f"- {item}")
        else:
            lines.append("- None documented")

        lines.extend(["", "## Recommendations"])
        if evaluation.recommendations:
            for item in evaluation.recommendations:
                lines.append(f"- {item}")
        else:
            lines.append("- No specific actions provided")

        lines.extend(["", "## Scores"])
        if evaluation.scores:
            for score in evaluation.scores:
                justification = f" — {score.justification}" if score.justification else ""
                lines.append(
                    f"- {score.dimension}: {score.value:.2f} ({score.scale}){justification}"
                )
        else:
            lines.append("- No quantitative scores returned")

        if evaluation.overall_rating is not None:
            lines.extend(
                [
                    "",
                    "## Overall Rating",
                    f"{evaluation.overall_rating:.2f} (0-5 scale)",
                ]
            )

        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _write_committee_summary(self, aggregate: CommitteeAggregate) -> Path:
        filename = f"summary_{aggregate.run_id}.md"
        path = self.output_dir / filename

        lines = [
            f"# Judge Committee Summary — {aggregate.run_id}",
            "",
        ]

        if aggregate.average_overall_rating is not None:
            lines.append(f"**Average Overall Rating**: {aggregate.average_overall_rating:.2f} (0-5)")
            lines.append("")

        if aggregate.dimension_averages:
            lines.append("## Average Dimension Scores")
            for dimension, value in sorted(aggregate.dimension_averages.items()):
                lines.append(f"- {dimension.title()}: {value:.2f}")
            lines.append("")

        lines.append("## Persona Highlights")
        for evaluation in aggregate.persona_evaluations:
            lines.append(f"### {evaluation.persona_name}")
            lines.append(evaluation.summary)
            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    # ------------------------------------------------------------------
    # Misc utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _slugify(text: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
        return slug.lower() or "judge"


__all__ = ["JudgeAgent"]
