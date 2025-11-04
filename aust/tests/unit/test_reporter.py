"""
Unit tests for ReporterAgent and report data models (Story 4.1).

Tests report structure generation, section outlines, metadata creation,
and file output functionality.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aust.src.agents.reporter import ReporterAgent
from aust.src.data_models.loop_state import ExitCondition, InnerLoopState, IterationResult
from aust.src.data_models.report import (
    AcademicReport,
    ReportMetadata,
    ReportSection,
    ReportSectionType,
)
from aust.src.data_models.hypothesis import Hypothesis
from aust.src.data_models.debate import DebateSession


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory for tests."""
    return tmp_path / "test_reports"


@pytest.fixture
def sample_inner_loop_state():
    """Create sample inner loop state for testing."""
    hypothesis = Hypothesis(
        hypothesis_id="hyp_001",
        attack_type="membership_inference",
        description="Test hypothesis",
        experiment_design="Measure model responses for known training examples and compare baselines.",
        target_type="data_memorization",
    )

    iteration = IterationResult(
        iteration_number=1,
        hypothesis=hypothesis,
        debate_session=DebateSession(session_id="d1", exchanges=[]),
        experiment_executed=True,
        vulnerability_detected=True,
        vulnerability_confidence=0.87,
    )

    state = InnerLoopState(
        task_id="test_task_001",
        task_type="concept_erasure",
        task_description="Test vulnerability assessment",
        max_iterations=10,
        iterations=[iteration],
        task_spec={
            "model_name": "Stable Diffusion",
            "model_version": "1.4",
            "unlearning_method": "ESD",
            "unlearned_target": "nudity",
        },
    )

    state.mark_complete(ExitCondition.VULNERABILITY_FOUND, "Vulnerability detected")

    return state


@pytest.fixture
def sample_attack_trace_data():
    """Create sample attack trace data."""
    return {
        "task_id": "test_task_001",
        "task_type": "concept_erasure",
        "configuration": {
            "max_iterations": 10,
            "enable_debate": True,
            "generator_model": "anthropic/claude-3.5-sonnet",
            "critic_model": "anthropic/claude-3.5-sonnet",
        },
        "iterations": [
            {
                "iteration_number": 1,
                "hypothesis": {
                    "attack_type": "membership_inference",
                    "description": "Test attack",
                },
                "evaluation": {
                    "vulnerability_detected": True,
                    "vulnerability_confidence": 0.87,
                },
            }
        ],
        "summary": {
            "total_iterations": 1,
            "vulnerability_found": True,
        },
    }


@pytest.fixture
def reporter_agent(temp_output_dir):
    """Create ReporterAgent instance for testing."""
    return ReporterAgent(output_dir=temp_output_dir)


class TestReportDataModels:
    """Test report data models."""

    def test_report_section_creation(self):
        """Test creating a report section."""
        section = ReportSection(
            section_type=ReportSectionType.INTRODUCTION,
            title="Introduction",
            content="Test content for introduction.",
            order=1,
        )

        assert section.section_type == ReportSectionType.INTRODUCTION
        assert section.title == "Introduction"
        assert section.has_content
        assert section.word_count == 4

    def test_report_section_add_citation(self):
        """Test adding citations to a section."""
        section = ReportSection(
            section_type=ReportSectionType.RESULTS,
            title="Results",
            content="Test",
            order=4,
        )

        section.add_citation("arxiv:2101.00001")
        section.add_citation("arxiv:2102.00002")
        section.add_citation("arxiv:2101.00001")  # Duplicate

        assert len(section.citations) == 2
        assert "arxiv:2101.00001" in section.citations

    def test_report_metadata_creation(self):
        """Test creating report metadata."""
        metadata = ReportMetadata(
            report_id="report_001",
            task_id="task_001",
            task_type="concept_erasure",
            target_model_name="Test Model",
            total_iterations=5,
            vulnerability_found=True,
            highest_confidence=0.92,
        )

        assert metadata.report_id == "report_001"
        assert metadata.vulnerability_found
        assert metadata.highest_confidence == 0.92

    def test_report_metadata_duration_calculation(self):
        """Test duration calculation in metadata."""
        start = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 15, 12, 30, 0, tzinfo=timezone.utc)

        metadata = ReportMetadata(
            report_id="report_001",
            task_id="task_001",
            task_type="concept_erasure",
            inner_loop_started_at=start,
            inner_loop_completed_at=end,
        )

        assert metadata.duration_hours == 2.5

    def test_academic_report_section_management(self):
        """Test adding and retrieving sections."""
        metadata = ReportMetadata(
            report_id="report_001", task_id="task_001", task_type="concept_erasure"
        )
        report = AcademicReport(metadata=metadata)

        intro_section = ReportSection(
            section_type=ReportSectionType.INTRODUCTION,
            title="Introduction",
            content="Intro content",
            order=1,
        )

        report.add_section(intro_section)

        assert len(report.sections) == 1
        assert report.get_section(ReportSectionType.INTRODUCTION) is not None
        assert report.get_section(ReportSectionType.METHODS) is None

    def test_academic_report_completeness_check(self):
        """Test report completeness validation."""
        metadata = ReportMetadata(
            report_id="report_001", task_id="task_001", task_type="concept_erasure"
        )
        report = AcademicReport(metadata=metadata)

        # Incomplete report
        assert not report.is_complete

        # Add required sections
        for section_type, order in [
            (ReportSectionType.INTRODUCTION, 1),
            (ReportSectionType.METHODS, 2),
            (ReportSectionType.RESULTS, 4),
            (ReportSectionType.CONCLUSION, 6),
        ]:
            report.add_section(
                ReportSection(
                    section_type=section_type,
                    title=section_type.value.title(),
                    content="Content",
                    order=order,
                )
            )

        # Should be complete now
        assert report.is_complete

    def test_academic_report_citation_aggregation(self):
        """Test aggregating all citations across sections."""
        metadata = ReportMetadata(
            report_id="report_001", task_id="task_001", task_type="concept_erasure"
        )
        report = AcademicReport(metadata=metadata)

        # Add sections with citations
        intro = ReportSection(
            section_type=ReportSectionType.INTRODUCTION,
            title="Introduction",
            content="Test",
            order=1,
            citations=["arxiv:2101.00001", "arxiv:2102.00002"],
        )

        methods = ReportSection(
            section_type=ReportSectionType.METHODS,
            title="Methods",
            content="Test",
            order=2,
            citations=["arxiv:2102.00002", "arxiv:2103.00003"],  # One duplicate
        )

        report.add_section(intro)
        report.add_section(methods)

        all_citations = report.all_citations

        assert len(all_citations) == 3  # Duplicates removed
        assert "arxiv:2101.00001" in all_citations
        assert "arxiv:2102.00002" in all_citations
        assert "arxiv:2103.00003" in all_citations

    def test_academic_report_markdown_generation(self, sample_inner_loop_state):
        """Test generating Markdown output."""
        metadata = ReportMetadata(
            report_id="report_001",
            task_id=sample_inner_loop_state.task_id,
            task_type=sample_inner_loop_state.task_type,
            target_model_name="Test Model",
            total_iterations=1,
            vulnerability_found=True,
            highest_confidence=0.87,
        )

        report = AcademicReport(metadata=metadata)

        intro = ReportSection(
            section_type=ReportSectionType.INTRODUCTION,
            title="Introduction",
            content="This is the introduction.",
            order=1,
        )

        report.add_section(intro)

        markdown = report.to_markdown()

        assert "# Vulnerability Assessment Report" in markdown
        assert "**Task Type**: concept_erasure" in markdown
        assert "## Introduction" in markdown
        assert "This is the introduction." in markdown


class TestReporterAgent:
    """Test ReporterAgent functionality."""

    def test_reporter_initialization(self, temp_output_dir):
        """Test reporter agent initialization."""
        reporter = ReporterAgent(output_dir=temp_output_dir)

        assert reporter.output_dir == temp_output_dir
        assert reporter.template_path.name == "report_template.md"

    def test_create_metadata_from_state(
        self, reporter_agent, sample_inner_loop_state, sample_attack_trace_data
    ):
        """Test metadata creation from inner loop state."""
        metadata = reporter_agent._create_metadata(
            sample_inner_loop_state, sample_attack_trace_data
        )

        assert metadata.task_id == "test_task_001"
        assert metadata.task_type == "concept_erasure"
        assert metadata.target_model_name == "Stable Diffusion"
        assert metadata.unlearning_method == "ESD"
        assert metadata.total_iterations == 1
        assert metadata.vulnerability_found == True
        assert metadata.generator_model == "anthropic/claude-3.5-sonnet"

    def test_generate_report_structure(
        self, reporter_agent, sample_inner_loop_state, tmp_path
    ):
        """Test generating complete report structure."""
        # Create mock attack trace files
        trace_json = tmp_path / "trace.json"
        trace_md = tmp_path / "trace.md"

        trace_data = {
            "task_id": "test_task_001",
            "configuration": {
                "generator_model": "test-model",
                "critic_model": "test-critic",
            },
            "iterations": [
                {
                    "iteration_number": 1,
                    "evaluation": {
                        "vulnerability_detected": True,
                        "vulnerability_confidence": 0.87,
                    },
                    "hypothesis": {"attack_type": "membership_inference"},
                }
            ],
        }

        trace_json.write_text(json.dumps(trace_data))
        trace_md.write_text("# Attack Trace")

        report = reporter_agent.generate_report(
            inner_loop_state=sample_inner_loop_state,
            attack_trace_json_path=trace_json,
            attack_trace_md_path=trace_md,
        )

        assert report.metadata.task_id == "test_task_001"
        assert len(report.sections) == 6  # All 6 sections
        assert report.get_section(ReportSectionType.INTRODUCTION) is not None
        assert report.get_section(ReportSectionType.METHODS) is not None
        assert report.get_section(ReportSectionType.EXPERIMENTS) is not None
        assert report.get_section(ReportSectionType.RESULTS) is not None
        assert report.get_section(ReportSectionType.DISCUSSION) is not None
        assert report.get_section(ReportSectionType.CONCLUSION) is not None

    def test_introduction_outline_generation(
        self, reporter_agent, sample_inner_loop_state, sample_attack_trace_data
    ):
        """Test Introduction section outline generation."""
        metadata = reporter_agent._create_metadata(
            sample_inner_loop_state, sample_attack_trace_data
        )

        intro_section = reporter_agent._create_introduction_outline(metadata)

        assert intro_section.section_type == ReportSectionType.INTRODUCTION
        assert intro_section.title == "Introduction"
        assert intro_section.has_content
        assert "Problem Statement" in intro_section.content
        assert "Objectives" in intro_section.content
        assert "Stable Diffusion" in intro_section.content

    def test_methods_outline_generation(
        self, reporter_agent, sample_inner_loop_state, sample_attack_trace_data
    ):
        """Test Methods section outline generation."""
        metadata = reporter_agent._create_metadata(
            sample_inner_loop_state, sample_attack_trace_data
        )

        methods_section = reporter_agent._create_methods_outline(
            metadata, sample_attack_trace_data
        )

        assert methods_section.section_type == ReportSectionType.METHODS
        assert "Hypothesis Generation Process" in methods_section.content
        assert "Multi-Agent Debate" in methods_section.content
        assert "anthropic/claude-3.5-sonnet" in methods_section.content

    def test_results_outline_includes_iteration_summary(
        self, reporter_agent, sample_inner_loop_state, sample_attack_trace_data
    ):
        """Test Results section includes iteration summaries."""
        metadata = reporter_agent._create_metadata(
            sample_inner_loop_state, sample_attack_trace_data
        )

        results_section = reporter_agent._create_results_outline(
            metadata, sample_inner_loop_state, sample_attack_trace_data
        )

        assert "Total iterations: 1" in results_section.content
        assert "Successful attacks: 1 / 1" in results_section.content
        assert "membership_inference" in results_section.content
        assert "87" in results_section.content  # Confidence percentage

    def test_save_report_creates_file(
        self, reporter_agent, sample_inner_loop_state, tmp_path
    ):
        """Test saving report to file."""
        metadata = ReportMetadata(
            report_id="report_test_001",
            task_id=sample_inner_loop_state.task_id,
            task_type="concept_erasure",
        )

        report = AcademicReport(metadata=metadata)

        intro = ReportSection(
            section_type=ReportSectionType.INTRODUCTION,
            title="Introduction",
            content="Test content",
            order=1,
        )

        report.add_section(intro)

        save_path = reporter_agent.save_report(report, sample_inner_loop_state.task_id)

        assert save_path.exists()
        assert save_path.name == "report_report_test_001.md"
        assert "# Vulnerability Assessment Report" in save_path.read_text()

    def test_generate_report_with_missing_trace(
        self, reporter_agent, sample_inner_loop_state, tmp_path
    ):
        """Test report generation handles missing attack trace gracefully."""
        # Non-existent trace files
        trace_json = tmp_path / "nonexistent.json"
        trace_md = tmp_path / "nonexistent.md"

        report = reporter_agent.generate_report(
            inner_loop_state=sample_inner_loop_state,
            attack_trace_json_path=trace_json,
            attack_trace_md_path=trace_md,
        )

        # Should still generate report with available data
        assert report is not None
        assert len(report.sections) == 6


class TestReportIntegration:
    """Integration tests for complete report generation workflow."""

    def test_full_report_generation_workflow(
        self, reporter_agent, sample_inner_loop_state, tmp_path
    ):
        """Test complete workflow from state to saved report."""
        # Create attack trace
        trace_json = tmp_path / "trace.json"
        trace_md = tmp_path / "trace.md"

        trace_data = {
            "task_id": sample_inner_loop_state.task_id,
            "configuration": {
                "generator_model": "test-gen",
                "critic_model": "test-critic",
                "max_debate_rounds": 3,
            },
            "iterations": [
                {
                    "iteration_number": 1,
                    "hypothesis": {"attack_type": "membership_inference"},
                    "evaluation": {
                        "vulnerability_detected": True,
                        "vulnerability_confidence": 0.87,
                    },
                }
            ],
        }

        trace_json.write_text(json.dumps(trace_data))
        trace_md.write_text("# Attack Trace\n\nTest trace")

        # Generate report
        report = reporter_agent.generate_report(
            inner_loop_state=sample_inner_loop_state,
            attack_trace_json_path=trace_json,
            attack_trace_md_path=trace_md,
        )

        # Save report
        save_path = reporter_agent.save_report(report, sample_inner_loop_state.task_id)

        # Verify
        assert save_path.exists()
        assert report.is_complete  # Has all required sections
        assert report.total_word_count > 100  # Has substantial content
        content = save_path.read_text()
        assert "Stable Diffusion" in content
        assert "ESD" in content
        assert "nudity" in content
