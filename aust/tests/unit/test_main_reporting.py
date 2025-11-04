"""Tests for report integration in aust.scripts.main."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from aust.scripts import main
from aust.src.data_models.loop_state import InnerLoopState


@pytest.fixture
def temp_state(tmp_path: Path) -> InnerLoopState:
    """Create a minimal InnerLoopState with attack trace files in place."""

    task_id = "task_test123"
    output_dir = tmp_path / task_id
    attack_traces_dir = output_dir / "attack_traces"
    attack_traces_dir.mkdir(parents=True)

    json_path = attack_traces_dir / f"trace_{task_id}.json"
    json_path.write_text("{}", encoding="utf-8")

    md_path = attack_traces_dir / f"trace_{task_id}.md"
    md_path.write_text("# Trace", encoding="utf-8")

    state = InnerLoopState(
        task_id=task_id,
        task_type="concept_erasure",
        task_description="Evaluate concept erasure",
        max_iterations=1,
        output_dir=output_dir,
        attack_trace_file=md_path,
    )

    return state


def test_generate_report_for_state_invokes_reporter(temp_state: InnerLoopState, tmp_path: Path):
    """ReporterAgent should be instantiated and invoked with expected arguments."""

    report_output = tmp_path / "report.md"

    with patch("aust.scripts.main.ReporterAgent") as mock_reporter_cls:
        mock_reporter = mock_reporter_cls.return_value
        mock_report = Mock(name="AcademicReport")
        mock_reporter.generate_report.return_value = mock_report
        mock_reporter.save_report.return_value = report_output

        result = main.generate_report_for_state(temp_state)

    attack_trace_dir = temp_state.output_dir / "attack_traces"
    expected_json = attack_trace_dir / f"trace_{temp_state.task_id}.json"
    expected_md = attack_trace_dir / f"trace_{temp_state.task_id}.md"

    assert result == (mock_report, report_output)

    mock_reporter_cls.assert_called_once_with(output_dir=temp_state.output_dir)
    mock_reporter.generate_report.assert_called_once_with(
        inner_loop_state=temp_state,
        attack_trace_json_path=expected_json,
        attack_trace_md_path=expected_md,
        retrieved_papers=None,
    )
    mock_reporter.save_report.assert_called_once_with(mock_report, temp_state.task_id)
