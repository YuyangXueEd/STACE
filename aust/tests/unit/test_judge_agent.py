"""Unit tests for the JudgeAgent workforce."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aust.src.agents.judge import JudgeAgent
from aust.src.data_models.report import (
    AcademicReport,
    ReportMetadata,
    ReportSection,
    ReportSectionType,
)


class _StubRunner:
    """Deterministic LLM runner for tests."""

    def __call__(self, persona: dict, system_prompt: str, user_prompt: str) -> str:  # noqa: D401
        scores: list[dict[str, object]] = []
        for dimension in persona.get("scoring_dimensions", []):
            scores.append(
                {
                    "dimension": dimension.get("name", "Dimension"),
                    "value": 4.0,
                    "scale": dimension.get("scale", "1-5"),
                    "justification": f"Stub justification for {dimension.get('name')}",
                }
            )

        payload = {
            "summary": f"Stub summary for {persona.get('name')}",
            "strengths": ["Strong evidence"],
            "weaknesses": ["Needs more evaluation"],
            "recommendations": ["Add more tests"],
            "scores": scores,
            "overall_rating": 4.2,
        }
        return json.dumps(payload)


def _build_sample_report() -> AcademicReport:
    metadata = ReportMetadata(
        report_id="report_stub",
        task_id="task-123",
        task_type="concept_erasure",
        target_model_name="ExampleNet",
        target_model_version="1.0",
        total_iterations=3,
        vulnerability_found=True,
        highest_confidence=0.82,
    )
    report = AcademicReport(metadata=metadata)
    report.add_section(
        ReportSection(
            section_type=ReportSectionType.INTRODUCTION,
            title="Introduction",
            content="This is a stub introduction.",
            order=0,
        )
    )
    report.add_section(
        ReportSection(
            section_type=ReportSectionType.METHODS,
            title="Methods",
            content="Stub methods.",
            order=1,
        )
    )
    report.add_section(
        ReportSection(
            section_type=ReportSectionType.RESULTS,
            title="Results",
            content="Stub results.",
            order=2,
        )
    )
    report.add_section(
        ReportSection(
            section_type=ReportSectionType.CONCLUSION,
            title="Conclusion",
            content="Stub conclusion.",
            order=3,
        )
    )
    return report


@pytest.fixture(name="judge_agent")
def fixture_judge_agent(tmp_path: Path) -> JudgeAgent:
    project_root = Path(__file__).resolve().parents[3]
    personas_path = project_root / "aust" / "configs" / "personas" / "judges.yaml"
    agent = JudgeAgent(
        persona_config_path=personas_path,
        output_dir=tmp_path,
        llm_runner=_StubRunner(),
    )
    return agent


def test_evaluate_single_persona_creates_markdown(judge_agent: JudgeAgent, tmp_path: Path) -> None:
    report = _build_sample_report()
    evaluation = judge_agent.evaluate(
        "sceptical_reviewer",
        report=report,
        attack_trace={"iterations": []},
        experiment_results={"metrics": {"asr": 0.2}},
        run_id="run123",
    )

    assert evaluation.persona_id == "sceptical_reviewer"
    assert evaluation.summary.startswith("Stub summary")
    assert evaluation.scores, "Expected stub scores to be hydrated"

    file_path = tmp_path / "judge_sceptical_reviewer_run123.md"
    assert file_path.exists(), "Persona evaluation markdown should be written"
    content = file_path.read_text(encoding="utf-8")
    assert "## Summary" in content
    assert "## Scores" in content


def test_run_committee_writes_summary(judge_agent: JudgeAgent, tmp_path: Path) -> None:
    report = _build_sample_report()
    aggregate = judge_agent.run_committee(
        report=report,
        attack_trace={"iterations": []},
        experiment_results={"metrics": {"asr": 0.2}},
        run_id="committee-run",
    )

    assert len(aggregate.persona_evaluations) == 5
    assert aggregate.average_overall_rating is not None
    assert aggregate.dimension_averages, "Expected aggregate dimension scores"

    summary_path = tmp_path / "summary_committee-run.md"
    assert summary_path.exists(), "Summary markdown should be generated"
    summary_content = summary_path.read_text(encoding="utf-8")
    assert "Judge Committee Summary" in summary_content
