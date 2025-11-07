"""
Unit tests for AC4: Long-Term Report Generation via LongTermMemoryAgent (Story 5.2).

Tests cover:
1. Loading long-report prompt configuration
2. Reading per-iteration trace files
3. Generating the composite report (with mocked LLM agent)
4. Handling error scenarios gracefully
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from aust.src.agents.long_term_memory_agent import LongTermMemoryAgent


class TestLongReportPromptLoading:
    """Validate long report prompt loading."""

    def test_load_long_report_prompts_success(self, tmp_path: Path) -> None:
        """Ensure prompts are loaded when YAML exists."""
        prompts_dir = tmp_path / "aust" / "configs" / "prompts"
        prompts_dir.mkdir(parents=True)

        prompt_file = prompts_dir / "long_report_generator.yaml"
        prompt_file.write_text(
            yaml.dump(
                {
                    "system_prompt": "Context: {attack_trace}",
                    "sections": {
                        "introduction": {"prompt_template": "Intro"},
                        "summary": {"prompt_template": "Summary"},
                    },
                }
            ),
            encoding="utf-8",
        )

        agent = LongTermMemoryAgent(memory_dir=tmp_path / "memory")
        prompts = agent._load_long_report_prompts(tmp_path)

        assert prompts["system_prompt"] == "Context: {attack_trace}"
        assert "introduction" in prompts["sections"]
        assert "summary" in prompts["sections"]

    def test_load_long_report_prompts_missing(self, tmp_path: Path) -> None:
        """Ensure FileNotFoundError raised when prompt file missing."""
        agent = LongTermMemoryAgent(memory_dir=tmp_path / "memory")

        with pytest.raises(FileNotFoundError):
            agent._load_long_report_prompts(tmp_path)


class TestIterationTraceLoading:
    """Exercise iteration trace helper."""

    def test_load_iteration_traces_multiple_files(self, tmp_path: Path) -> None:
        traces_dir = tmp_path / "attack_traces"
        traces_dir.mkdir()

        for idx in range(1, 4):
            trace = {
                "task_id": "test_task",
                "iteration": {"iteration_number": idx, "final_status": "success"},
            }
            (traces_dir / f"attack_trace_iter_{idx:02d}.json").write_text(
                json.dumps(trace),
                encoding="utf-8",
            )

        agent = LongTermMemoryAgent(memory_dir=tmp_path / "memory")
        traces = agent._load_iteration_traces(traces_dir)

        assert len(traces) == 3
        assert traces[0]["iteration"]["iteration_number"] == 1
        assert traces[-1]["iteration"]["iteration_number"] == 3

    def test_load_iteration_traces_handles_corruption(self, tmp_path: Path) -> None:
        traces_dir = tmp_path / "attack_traces"
        traces_dir.mkdir()

        valid_trace = {"iteration": {"iteration_number": 1}}
        (traces_dir / "attack_trace_iter_01.json").write_text(
            json.dumps(valid_trace),
            encoding="utf-8",
        )
        # Corrupted file
        (traces_dir / "attack_trace_iter_02.json").write_text("{invalid json", encoding="utf-8")

        agent = LongTermMemoryAgent(memory_dir=tmp_path / "memory")
        traces = agent._load_iteration_traces(traces_dir)

        assert len(traces) == 1
        assert traces[0]["iteration"]["iteration_number"] == 1

    def test_load_iteration_traces_missing_directory(self, tmp_path: Path) -> None:
        agent = LongTermMemoryAgent(memory_dir=tmp_path / "memory")
        traces = agent._load_iteration_traces(tmp_path / "missing")
        assert traces == []


class TestGenerateLongTermReport:
    """Verify end-to-end report synthesis with mocks."""

    @pytest.fixture
    def sample_traces(self, tmp_path: Path) -> Path:
        traces_dir = tmp_path / "attack_traces"
        traces_dir.mkdir()
        trace_payload = {
            "task_id": "test_task",
            "iteration": {
                "iteration_number": 1,
                "final_status": "success",
                "hypothesis": {"attack_type": "coref_probing"},
            },
        }
        (traces_dir / "attack_trace_iter_01.json").write_text(
            json.dumps(trace_payload),
            encoding="utf-8",
        )
        return traces_dir

    def test_generate_long_term_report_with_mocked_agent(
        self, tmp_path: Path, sample_traces: Path
    ) -> None:
        agent = LongTermMemoryAgent(memory_dir=tmp_path / "memory")

        prompts = {
            "system_prompt": "Trace: {attack_trace}",
            "sections": {
                "introduction": {"prompt_template": "intro"},
                "generated attacking methods": {"prompt_template": "methods"},
                "summary": {"prompt_template": "summary"},
                "discussion": {"prompt_template": "discussion"},
            },
        }

        mock_report_agent = MagicMock()
        mock_report_agent.step.return_value = MagicMock(
            msgs=[MagicMock(content="Section content")]
        )

        with patch.object(agent, "_load_long_report_prompts", return_value=prompts):
            with patch.object(agent, "_create_report_agent", return_value=mock_report_agent):
                report_path = agent.generate_long_term_report(
                    task_id="test_task",
                    traces_dir=sample_traces,
                    output_dir=tmp_path,
                )

        assert report_path is not None
        assert report_path.exists()
        report_text = report_path.read_text(encoding="utf-8")
        assert "# Vulnerability Assessment Report" in report_text
        assert "Section content" in report_text

    def test_generate_long_term_report_no_traces(self, tmp_path: Path) -> None:
        agent = LongTermMemoryAgent(memory_dir=tmp_path / "memory")
        report_path = agent.generate_long_term_report(
            task_id="test_task",
            traces_dir=tmp_path / "missing",
            output_dir=tmp_path,
        )
        assert report_path is None

    def test_generate_long_term_report_handles_llm_failure(
        self, tmp_path: Path, sample_traces: Path
    ) -> None:
        agent = LongTermMemoryAgent(memory_dir=tmp_path / "memory")

        prompts = {
            "system_prompt": "Trace: {attack_trace}",
            "sections": {
                "introduction": {"prompt_template": "intro"},
            },
        }

        failing_agent = MagicMock()
        failing_agent.step.side_effect = RuntimeError("LLM failure")

        with patch.object(agent, "_load_long_report_prompts", return_value=prompts):
            with patch.object(agent, "_create_report_agent", return_value=failing_agent):
                report_path = agent.generate_long_term_report(
                    task_id="test_task",
                    traces_dir=sample_traces,
                    output_dir=tmp_path,
                )

        assert report_path is not None
        content = report_path.read_text(encoding="utf-8")
        assert "Error generating content" in content
        assert "LLM failure" in content


class TestReportAgentHelpers:
    """Cover helper utilities for report generation."""

    def test_create_report_agent_formats_prompt(self, tmp_path: Path) -> None:
        agent = LongTermMemoryAgent(memory_dir=tmp_path / "memory")
        system_prompt_template = "Payload: {attack_trace}"
        attack_trace = '[{"iteration": 1}]'

        with patch("camel.models.ModelFactory") as mock_factory:
            with patch("camel.agents.ChatAgent") as mock_chat_agent:
                with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
                    agent._create_report_agent(system_prompt_template, attack_trace)

        assert mock_chat_agent.called
        system_message = mock_chat_agent.call_args.kwargs["system_message"]
        assert attack_trace in system_message.content

    def test_generate_section_content_success(self, tmp_path: Path) -> None:
        agent = LongTermMemoryAgent(memory_dir=tmp_path / "memory")
        mock_llm = MagicMock()
        response = MagicMock()
        response.msgs = [MagicMock(content="Generated text")]
        mock_llm.step.return_value = response

        content = agent._generate_section_content(mock_llm, "prompt")
        assert content == "Generated text"

    def test_generate_section_content_failure(self, tmp_path: Path) -> None:
        agent = LongTermMemoryAgent(memory_dir=tmp_path / "memory")
        mock_llm = MagicMock()
        mock_llm.step.side_effect = RuntimeError("Timeout")

        content = agent._generate_section_content(mock_llm, "prompt")
        assert "Error generating content" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
