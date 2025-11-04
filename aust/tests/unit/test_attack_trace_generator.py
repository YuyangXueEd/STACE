"""
Unit tests for AttackTraceGenerator (Story 4.3).

Tests dual-format trace generation (JSON + Markdown) with:
- Iteration narratives
- Hypothesis evolution
- Failure analysis
- RAG context integration
- Experiment results formatting
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aust.src.data_models.critic import CriticFeedback
from aust.src.data_models.debate import DebateExchange, DebateSession
from aust.src.data_models.hypothesis import Hypothesis
from aust.src.data_models.loop_state import ExitCondition, InnerLoopState, IterationResult
from aust.src.utils.attack_trace_generator import AttackTraceGenerator


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory for tests."""
    return tmp_path / "test_output"


@pytest.fixture
def trace_generator(temp_output_dir):
    """Create AttackTraceGenerator instance for tests."""
    return AttackTraceGenerator(output_dir=temp_output_dir, task_id="test_task_001")


@pytest.fixture
def sample_hypothesis():
    """Create sample hypothesis for testing."""
    return Hypothesis(
        hypothesis_id="hyp_001",
        attack_type="membership_inference",
        description="Test if model memorizes specific training examples",
        experiment_design="Compare model outputs on memorized vs. novel prompts and measure overlap.",
        target_type="data_memorization",
        confidence_score=0.75,
        novelty_score=0.82,
    )


@pytest.fixture
def sample_debate_session(sample_hypothesis):
    """Create sample debate session for testing."""
    critic_feedback = CriticFeedback(
        overall_assessment="Hypothesis is promising but needs more specificity",
        novelty_score=0.82,
        feasibility_score=0.75,
        rigor_score=0.68,
        strengths=["Well-grounded in literature", "Clear experiment design"],
        weaknesses=["Could use more specific metrics", "Baseline comparison needs detail"],
        suggestions=[
            "Specify which training examples to test",
            "Add quantitative success metric",
        ],
    )

    exchange = DebateExchange(
        round_number=1,
        initial_hypothesis=sample_hypothesis,
        critic_feedback=critic_feedback,
        generator_model="anthropic/claude-3.5-sonnet",
        critic_model="anthropic/claude-3.5-sonnet",
    )

    return DebateSession(
        iteration_number=1,
        task_id="test_task_001",
        task_type="concept_erasure",
        exchanges=[exchange],
        total_rounds=1,
        convergence_reached=False,
        quality_threshold_met=True,
    )


@pytest.fixture
def sample_iteration_result(sample_hypothesis, sample_debate_session):
    """Create sample iteration result for testing."""
    return IterationResult(
        iteration_number=1,
        hypothesis=sample_hypothesis,
        debate_session=sample_debate_session,
        rag_queries=["machine unlearning memorization", "data forgetting attacks"],
        retrieved_paper_count=5,
        retrieved_paper_ids=["arxiv:2101.00001", "arxiv:2102.00002", "arxiv:2103.00003"],
        experiment_executed=True,
        experiment_results={
            "success": True,
            "execution_time_seconds": 120.5,
            "metrics": {
                "memorization_score": 0.85,
                "baseline_score": 0.15,
            },
            "observations": "Model successfully outputs training examples verbatim.",
        },
        evaluator_feedback="High confidence vulnerability detected",
        vulnerability_detected=True,
        vulnerability_confidence=0.87,
        started_at=datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        completed_at=datetime(2025, 1, 15, 10, 3, 0, tzinfo=timezone.utc),
    )


class TestAttackTraceGeneratorInitialization:
    """Test trace generator initialization."""

    def test_init_creates_directories(self, trace_generator, temp_output_dir):
        """Test that initialization creates necessary directories."""
        assert (temp_output_dir / "attack_traces").exists()
        assert trace_generator.task_id == "test_task_001"

    def test_init_sets_file_paths(self, trace_generator):
        """Test that file paths are correctly set."""
        assert "test_task_001" in str(trace_generator.json_trace_file)
        assert "test_task_001" in str(trace_generator.md_trace_file)
        assert trace_generator.json_trace_file.suffix == ".json"
        assert trace_generator.md_trace_file.suffix == ".md"

    def test_initialize_trace_creates_header_files(self, trace_generator):
        """Test that initialize_trace creates both JSON and MD files."""
        trace_generator.initialize_trace(
            task_type="concept_erasure",
            task_description="Test vulnerability discovery",
            task_spec={"model_name": "Test Model", "unlearning_method": "ESD"},
            max_iterations=10,
            enable_debate=True,
            generator_model="anthropic/claude-3.5-sonnet",
            critic_model="anthropic/claude-3.5-sonnet",
        )

        assert trace_generator.json_trace_file.exists()
        assert trace_generator.md_trace_file.exists()

    def test_initialize_trace_json_structure(self, trace_generator):
        """Test that JSON trace has correct initial structure."""
        trace_generator.initialize_trace(
            task_type="concept_erasure",
            task_description="Test vulnerability discovery",
            task_spec=None,
            max_iterations=10,
            enable_debate=True,
            generator_model="test-model",
            critic_model="test-critic",
        )

        json_data = json.loads(trace_generator.json_trace_file.read_text())

        assert json_data["task_id"] == "test_task_001"
        assert json_data["task_type"] == "concept_erasure"
        assert json_data["iterations"] == []
        assert json_data["summary"] is None
        assert json_data["configuration"]["max_iterations"] == 10

    def test_initialize_trace_markdown_contains_header(self, trace_generator):
        """Test that Markdown trace contains proper header."""
        trace_generator.initialize_trace(
            task_type="concept_erasure",
            task_description="Test vulnerability discovery",
            task_spec=None,
            max_iterations=10,
            enable_debate=True,
            generator_model="test-model",
            critic_model="test-critic",
        )

        md_content = trace_generator.md_trace_file.read_text()

        assert "# Attack Trace: test_task_001" in md_content
        assert "**Task Type**: concept_erasure" in md_content
        assert "**Max Iterations**: 10" in md_content


class TestAttackTraceIterationAppend:
    """Test appending iterations to traces."""

    def test_append_iteration_json_adds_entry(
        self, trace_generator, sample_iteration_result
    ):
        """Test that appending iteration adds entry to JSON."""
        trace_generator.initialize_trace(
            task_type="test",
            task_description="test",
            task_spec=None,
            max_iterations=10,
            enable_debate=True,
            generator_model="test",
            critic_model="test",
        )

        trace_generator.append_iteration(sample_iteration_result, iteration_number=1)

        json_data = json.loads(trace_generator.json_trace_file.read_text())
        assert len(json_data["iterations"]) == 1
        assert json_data["iterations"][0]["iteration_number"] == 1

    def test_append_iteration_json_includes_all_fields(
        self, trace_generator, sample_iteration_result
    ):
        """Test that JSON iteration entry includes all required fields."""
        trace_generator.initialize_trace(
            task_type="test",
            task_description="test",
            task_spec=None,
            max_iterations=10,
            enable_debate=True,
            generator_model="test",
            critic_model="test",
        )

        trace_generator.append_iteration(sample_iteration_result, iteration_number=1)

        json_data = json.loads(trace_generator.json_trace_file.read_text())
        iteration = json_data["iterations"][0]

        # Check top-level fields
        assert "hypothesis" in iteration
        assert "debate_session" in iteration
        assert "rag_info" in iteration
        assert "experiment" in iteration
        assert "evaluation" in iteration
        assert "timing" in iteration
        assert "outcome_summary" in iteration
        assert "key_learning" in iteration

        # Check nested structures
        assert iteration["rag_info"]["retrieved_paper_count"] == 5
        assert iteration["experiment"]["executed"] is True
        assert iteration["evaluation"]["vulnerability_detected"] is True
        assert iteration["evaluation"]["vulnerability_confidence"] == 0.87

    def test_append_iteration_markdown_contains_narrative(
        self, trace_generator, sample_iteration_result
    ):
        """Test that Markdown iteration contains narrative elements."""
        trace_generator.initialize_trace(
            task_type="test",
            task_description="test",
            task_spec=None,
            max_iterations=10,
            enable_debate=True,
            generator_model="test",
            critic_model="test",
        )

        trace_generator.append_iteration(sample_iteration_result, iteration_number=1)

        md_content = trace_generator.md_trace_file.read_text()

        # Check narrative sections
        assert "### Iteration 1" in md_content
        assert "#### Hypothesis Description" in md_content
        assert "#### Experiment Design" in md_content
        assert "#### Debate Refinement" in md_content
        assert "#### Literature Context (RAG)" in md_content
        assert "#### Experiment Execution" in md_content
        assert "#### Evaluation Results" in md_content
        assert "#### Key Learning" in md_content

    def test_append_iteration_markdown_includes_debate_feedback(
        self, trace_generator, sample_iteration_result
    ):
        """Test that Markdown includes critic feedback from debate."""
        trace_generator.initialize_trace(
            task_type="test",
            task_description="test",
            task_spec=None,
            max_iterations=10,
            enable_debate=True,
            generator_model="test",
            critic_model="test",
        )

        trace_generator.append_iteration(sample_iteration_result, iteration_number=1)

        md_content = trace_generator.md_trace_file.read_text()

        assert "**Round 1 - Critic Feedback**" in md_content
        assert "**Strengths**:" in md_content
        assert "Well-grounded in literature" in md_content
        assert "**Weaknesses**:" in md_content
        assert "**Suggestions**:" in md_content

    def test_append_multiple_iterations(self, trace_generator, sample_iteration_result):
        """Test appending multiple iterations sequentially."""
        trace_generator.initialize_trace(
            task_type="test",
            task_description="test",
            task_spec=None,
            max_iterations=10,
            enable_debate=True,
            generator_model="test",
            critic_model="test",
        )

        # Append 3 iterations
        for i in range(1, 4):
            iter_result = sample_iteration_result.model_copy()
            iter_result.iteration_number = i
            trace_generator.append_iteration(iter_result, iteration_number=i)

        json_data = json.loads(trace_generator.json_trace_file.read_text())
        assert len(json_data["iterations"]) == 3
        assert [it["iteration_number"] for it in json_data["iterations"]] == [1, 2, 3]


class TestAttackTraceFinalization:
    """Test trace finalization with summary."""

    def test_finalize_adds_summary_to_json(
        self, trace_generator, sample_iteration_result
    ):
        """Test that finalization adds summary to JSON."""
        trace_generator.initialize_trace(
            task_type="test",
            task_description="test",
            task_spec=None,
            max_iterations=10,
            enable_debate=True,
            generator_model="test",
            critic_model="test",
        )

        trace_generator.append_iteration(sample_iteration_result, iteration_number=1)

        # Create final state
        final_state = InnerLoopState(
            task_id="test_task_001",
            task_type="test",
            task_description="test",
            max_iterations=10,
            iterations=[sample_iteration_result],
        )
        final_state.mark_complete(
            ExitCondition.VULNERABILITY_FOUND, "High confidence vulnerability"
        )

        trace_generator.finalize_trace(final_state)

        json_data = json.loads(trace_generator.json_trace_file.read_text())

        assert json_data["summary"] is not None
        assert json_data["summary"]["total_iterations"] == 1
        assert json_data["summary"]["exit_condition"] == "vulnerability_found"
        assert json_data["summary"]["vulnerability_found"] is True

    def test_finalize_adds_summary_to_markdown(
        self, trace_generator, sample_iteration_result
    ):
        """Test that finalization adds summary section to Markdown."""
        trace_generator.initialize_trace(
            task_type="test",
            task_description="test",
            task_spec=None,
            max_iterations=10,
            enable_debate=True,
            generator_model="test",
            critic_model="test",
        )

        trace_generator.append_iteration(sample_iteration_result, iteration_number=1)

        final_state = InnerLoopState(
            task_id="test_task_001",
            task_type="test",
            task_description="test",
            max_iterations=10,
            iterations=[sample_iteration_result],
        )
        final_state.mark_complete(
            ExitCondition.VULNERABILITY_FOUND, "High confidence vulnerability"
        )

        trace_generator.finalize_trace(final_state)

        md_content = trace_generator.md_trace_file.read_text()

        assert "## Final Summary" in md_content
        assert "**Total Iterations**: 1" in md_content
        assert "**Vulnerability Found**: Yes" in md_content
        assert "### Hypothesis Evolution Analysis" in md_content
        assert "### Failure Analysis" in md_content

    def test_finalize_returns_both_paths(
        self, trace_generator, sample_iteration_result
    ):
        """Test that finalize returns both JSON and MD paths."""
        trace_generator.initialize_trace(
            task_type="test",
            task_description="test",
            task_spec=None,
            max_iterations=10,
            enable_debate=True,
            generator_model="test",
            critic_model="test",
        )

        final_state = InnerLoopState(
            task_id="test_task_001",
            task_type="test",
            task_description="test",
            max_iterations=10,
            iterations=[sample_iteration_result],
        )
        final_state.mark_complete(ExitCondition.MAX_ITERATIONS, "Max iterations")

        json_path, md_path = trace_generator.finalize_trace(final_state)

        assert json_path.exists()
        assert md_path.exists()
        assert json_path.suffix == ".json"
        assert md_path.suffix == ".md"


class TestNarrativeGeneration:
    """Test narrative generation helpers."""

    def test_evolution_narrative_tracks_attack_types(
        self, trace_generator, sample_hypothesis
    ):
        """Test that evolution narrative tracks attack type progression."""
        # Create multiple iterations with different attack types
        iter1 = IterationResult(
            iteration_number=1,
            hypothesis=sample_hypothesis,
            debate_session=DebateSession(
                iteration_number=1,
                task_id="test_task_001",
                task_type="concept_erasure",
                exchanges=[],
            ),
        )

        hyp2 = sample_hypothesis.model_copy()
        hyp2.attack_type = "gradient_leakage"
        iter2 = IterationResult(
            iteration_number=2,
            hypothesis=hyp2,
            debate_session=DebateSession(
                iteration_number=2,
                task_id="test_task_001",
                task_type="concept_erasure",
                exchanges=[],
            ),
        )

        final_state = InnerLoopState(
            task_id="test",
            task_type="test",
            task_description="test",
            max_iterations=10,
            iterations=[iter1, iter2],
        )

        narrative = trace_generator._generate_evolution_narrative(final_state)

        assert "membership_inference" in narrative
        assert "gradient_leakage" in narrative

    def test_failure_narrative_analyzes_non_vulnerable_iterations(
        self, trace_generator, sample_iteration_result
    ):
        """Test that failure narrative identifies non-vulnerable iterations."""
        # Create failed iteration
        failed_iter = sample_iteration_result.model_copy()
        failed_iter.vulnerability_detected = False
        failed_iter.vulnerability_confidence = 0.1
        failed_iter.evaluator_feedback = "No vulnerability detected in this approach"

        final_state = InnerLoopState(
            task_id="test",
            task_type="test",
            task_description="test",
            max_iterations=10,
            iterations=[failed_iter],
        )

        narrative = trace_generator._generate_failure_narrative(final_state)

        assert "**Failed Iterations**: 1 / 1" in narrative
        assert "No vulnerability detected" in narrative

    def test_failure_narrative_handles_all_successes(self, trace_generator):
        """Test failure narrative when all iterations succeed."""
        final_state = InnerLoopState(
            task_id="test",
            task_type="test",
            task_description="test",
            max_iterations=10,
            iterations=[],
        )

        narrative = trace_generator._generate_failure_narrative(final_state)

        assert "All iterations resulted in detected vulnerabilities" in narrative


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_iteration_list(self, trace_generator):
        """Test finalization with no iterations."""
        trace_generator.initialize_trace(
            task_type="test",
            task_description="test",
            task_spec=None,
            max_iterations=10,
            enable_debate=True,
            generator_model="test",
            critic_model="test",
        )

        final_state = InnerLoopState(
            task_id="test",
            task_type="test",
            task_description="test",
            max_iterations=10,
            iterations=[],
        )
        final_state.mark_complete(ExitCondition.ERROR, "Failed before first iteration")

        trace_generator.finalize_trace(final_state)

        json_data = json.loads(trace_generator.json_trace_file.read_text())
        assert json_data["summary"]["total_iterations"] == 0

    def test_iteration_without_experiment_results(
        self, trace_generator, sample_hypothesis
    ):
        """Test iteration where experiment was not executed."""
        iter_result = IterationResult(
            iteration_number=1,
            hypothesis=sample_hypothesis,
            debate_session=DebateSession(
                iteration_number=1,
                task_id="test_task_001",
                task_type="concept_erasure",
                exchanges=[],
            ),
            experiment_executed=False,
            experiment_results=None,
        )

        trace_generator.initialize_trace(
            task_type="test",
            task_description="test",
            task_spec=None,
            max_iterations=10,
            enable_debate=True,
            generator_model="test",
            critic_model="test",
        )

        trace_generator.append_iteration(iter_result, iteration_number=1)

        md_content = trace_generator.md_trace_file.read_text()
        assert "**Status**: Not executed" in md_content

    def test_iteration_without_rag_queries(self, trace_generator, sample_hypothesis):
        """Test iteration with no RAG queries."""
        iter_result = IterationResult(
            iteration_number=1,
            hypothesis=sample_hypothesis,
            debate_session=DebateSession(
                iteration_number=1,
                task_id="test_task_001",
                task_type="concept_erasure",
                exchanges=[],
            ),
            rag_queries=[],
            retrieved_paper_count=0,
            retrieved_paper_ids=[],
        )

        trace_generator.initialize_trace(
            task_type="test",
            task_description="test",
            task_spec=None,
            max_iterations=10,
            enable_debate=True,
            generator_model="test",
            critic_model="test",
        )

        trace_generator.append_iteration(iter_result, iteration_number=1)

        json_data = json.loads(trace_generator.json_trace_file.read_text())
        assert json_data["iterations"][0]["rag_info"]["retrieved_paper_count"] == 0
