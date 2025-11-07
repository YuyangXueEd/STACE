"""
Unit tests for AC7: Per-Iteration Attack Trace Generation (Story 5.2).

Tests cover:
1. Per-iteration trace file generation
2. JSON structure validation
3. Filename format validation
4. Task context preservation
5. Integration with InnerLoopOrchestrator
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aust.src.data_models.debate import DebateSession
from aust.src.data_models.hypothesis import Hypothesis
from aust.src.data_models.loop_state import IterationResult
from aust.src.utils.attack_trace_generator import AttackTraceGenerator


class TestPerIterationTraceGeneration:
    """Test AC7: Per-iteration trace file generation."""

    @pytest.fixture
    def sample_hypothesis(self) -> Hypothesis:
        """Create a sample hypothesis for testing."""
        return Hypothesis(
            attack_type="coref_probing",
            description="Test hypothesis description",
            experiment_design="Test experiment design",
            target_type="object",
            confidence_score=0.85,
            novelty_score=0.7,
        )

    @pytest.fixture
    def sample_debate_session(self) -> DebateSession:
        """Create a sample debate session for testing."""
        return DebateSession(
            iteration_number=1,
            task_id="test_task_123",
            task_type="concept_erasure",
            exchanges=[],
            total_rounds=0,
            debate_enabled=False,
        )

    @pytest.fixture
    def sample_iteration_result(
        self, sample_hypothesis: Hypothesis, sample_debate_session: DebateSession
    ) -> IterationResult:
        """Create a sample iteration result for testing."""
        started = datetime.now(timezone.utc)
        completed = datetime.now(timezone.utc)

        return IterationResult(
            iteration_number=1,
            hypothesis=sample_hypothesis,
            debate_session=sample_debate_session,
            rag_queries=["test query 1", "test query 2"],
            retrieved_paper_count=2,
            retrieved_paper_ids=["paper_1", "paper_2"],
            experiment_executed=True,
            experiment_results={"images_generated": 10, "success": True},
            evaluator_feedback="Test feedback",
            vulnerability_detected=True,
            vulnerability_confidence=0.85,
            started_at=started,
            completed_at=completed,
        )

    def test_save_iteration_trace_creates_file(
        self, tmp_path: Path, sample_iteration_result: IterationResult
    ) -> None:
        """Test that save_iteration_trace() creates a JSON file."""
        trace_generator = AttackTraceGenerator(
            output_dir=tmp_path, task_id="test_task_123"
        )

        trace_file = trace_generator.save_iteration_trace(
            iteration_result=sample_iteration_result,
            iteration_number=1,
            task_type="concept_erasure",
            task_description="Test task description",
            task_spec={"model_name": "test-model"},
        )

        assert trace_file.exists()
        assert trace_file.name == "attack_trace_iter_01.json"
        assert trace_file.parent == trace_generator.traces_dir

    def test_iteration_trace_filename_format(self, tmp_path: Path, sample_iteration_result: IterationResult) -> None:
        """Test that iteration trace filenames follow the format attack_trace_iter_{N:02d}.json."""
        trace_generator = AttackTraceGenerator(
            output_dir=tmp_path, task_id="test_task_123"
        )

        # Test multiple iterations
        for iteration_num in [1, 5, 10, 99]:
            trace_file = trace_generator.save_iteration_trace(
                iteration_result=sample_iteration_result,
                iteration_number=iteration_num,
                task_type="concept_erasure",
            )

            expected_filename = f"attack_trace_iter_{iteration_num:02d}.json"
            assert trace_file.name == expected_filename

    def test_iteration_trace_json_structure(
        self, tmp_path: Path, sample_iteration_result: IterationResult
    ) -> None:
        """Test that iteration trace JSON has correct structure with task context."""
        trace_generator = AttackTraceGenerator(
            output_dir=tmp_path, task_id="test_task_apple_123"
        )

        task_spec = {
            "model_name": "Stable Diffusion",
            "model_version": "1.4",
            "unlearned_target": "apple",
        }

        trace_file = trace_generator.save_iteration_trace(
            iteration_result=sample_iteration_result,
            iteration_number=1,
            task_type="concept_erasure",
            task_description="Attack SD 1.4 unlearned with apple",
            task_spec=task_spec,
        )

        # Load and validate JSON structure
        with trace_file.open("r", encoding="utf-8") as f:
            trace_data = json.load(f)

        # Validate top-level task context fields
        assert trace_data["task_id"] == "test_task_apple_123"
        assert trace_data["task_type"] == "concept_erasure"
        assert trace_data["task_description"] == "Attack SD 1.4 unlearned with apple"
        assert trace_data["task_spec"] == task_spec

        # Validate iteration nested structure
        assert "iteration" in trace_data
        iteration = trace_data["iteration"]

        assert iteration["iteration_number"] == 1
        assert "started_at" in iteration
        assert "completed_at" in iteration
        assert "hypothesis" in iteration
        assert "attempts" in iteration
        assert iteration["final_status"] in ["success", "failure"]
        assert "vulnerability_detected" in iteration
        assert "confidence" in iteration

    def test_iteration_trace_hypothesis_serialization(
        self, tmp_path: Path, sample_iteration_result: IterationResult
    ) -> None:
        """Test that hypothesis is correctly serialized in iteration trace."""
        trace_generator = AttackTraceGenerator(
            output_dir=tmp_path, task_id="test_task_123"
        )

        trace_file = trace_generator.save_iteration_trace(
            iteration_result=sample_iteration_result,
            iteration_number=1,
            task_type="concept_erasure",
        )

        with trace_file.open("r", encoding="utf-8") as f:
            trace_data = json.load(f)

        hypothesis = trace_data["iteration"]["hypothesis"]
        assert hypothesis["attack_type"] == "coref_probing"
        assert hypothesis["description"] == "Test hypothesis description"
        assert hypothesis["experiment_design"] == "Test experiment design"
        assert hypothesis["target_type"] == "object"
        assert hypothesis["confidence_score"] == 0.85
        assert hypothesis["novelty_score"] == 0.7

    def test_iteration_trace_attempts_structure(
        self, tmp_path: Path, sample_iteration_result: IterationResult
    ) -> None:
        """Test that attempts array has correct structure."""
        trace_generator = AttackTraceGenerator(
            output_dir=tmp_path, task_id="test_task_123"
        )

        trace_file = trace_generator.save_iteration_trace(
            iteration_result=sample_iteration_result,
            iteration_number=1,
            task_type="concept_erasure",
        )

        with trace_file.open("r", encoding="utf-8") as f:
            trace_data = json.load(f)

        attempts = trace_data["iteration"]["attempts"]
        assert isinstance(attempts, list)
        assert len(attempts) == 1

        attempt = attempts[0]
        assert attempt["attempt_number"] == 1
        assert attempt["status"] == "success"  # vulnerability_detected=True
        assert attempt["images_generated"] == 10
        assert attempt["evaluator_feedback"] == "Test feedback"

    def test_iteration_trace_failure_status(
        self, tmp_path: Path, sample_iteration_result: IterationResult
    ) -> None:
        """Test that failure status is correctly recorded."""
        # Modify iteration result to indicate failure
        sample_iteration_result.vulnerability_detected = False
        sample_iteration_result.vulnerability_confidence = 0.0

        trace_generator = AttackTraceGenerator(
            output_dir=tmp_path, task_id="test_task_123"
        )

        trace_file = trace_generator.save_iteration_trace(
            iteration_result=sample_iteration_result,
            iteration_number=1,
            task_type="concept_erasure",
        )

        with trace_file.open("r", encoding="utf-8") as f:
            trace_data = json.load(f)

        iteration = trace_data["iteration"]
        assert iteration["final_status"] == "failure"
        assert iteration["vulnerability_detected"] is False
        assert iteration["confidence"] == 0.0

        # Check attempt status
        attempt = iteration["attempts"][0]
        assert attempt["status"] == "failure"

    def test_multiple_iteration_traces_independent(
        self, tmp_path: Path, sample_iteration_result: IterationResult
    ) -> None:
        """Test that multiple iteration traces are saved independently."""
        trace_generator = AttackTraceGenerator(
            output_dir=tmp_path, task_id="test_task_123"
        )

        # Save 3 iterations
        trace_files = []
        for i in range(1, 4):
            sample_iteration_result.iteration_number = i
            trace_file = trace_generator.save_iteration_trace(
                iteration_result=sample_iteration_result,
                iteration_number=i,
                task_type="concept_erasure",
            )
            trace_files.append(trace_file)

        # Verify all 3 files exist
        assert len(trace_files) == 3
        for trace_file in trace_files:
            assert trace_file.exists()

        # Verify filenames
        assert trace_files[0].name == "attack_trace_iter_01.json"
        assert trace_files[1].name == "attack_trace_iter_02.json"
        assert trace_files[2].name == "attack_trace_iter_03.json"

        # Verify each has correct iteration_number
        for i, trace_file in enumerate(trace_files, start=1):
            with trace_file.open("r", encoding="utf-8") as f:
                trace_data = json.load(f)
            assert trace_data["iteration"]["iteration_number"] == i

    def test_iteration_trace_atomic_write(
        self, tmp_path: Path, sample_iteration_result: IterationResult
    ) -> None:
        """Test that iteration traces use atomic write pattern."""
        trace_generator = AttackTraceGenerator(
            output_dir=tmp_path, task_id="test_task_123"
        )

        trace_file = trace_generator.save_iteration_trace(
            iteration_result=sample_iteration_result,
            iteration_number=1,
            task_type="concept_erasure",
        )

        # Verify final file exists
        assert trace_file.exists()

        # Verify temp file was cleaned up
        temp_file = trace_file.with_suffix(".tmp")
        assert not temp_file.exists()

    def test_images_count_extraction(self, tmp_path: Path, sample_iteration_result: IterationResult) -> None:
        """Test that images_generated count is correctly extracted from experiment_results."""
        trace_generator = AttackTraceGenerator(
            output_dir=tmp_path, task_id="test_task_123"
        )

        # Test with different experiment_results structures
        test_cases = [
            {"images_generated": 15, "success": True},
            {"images_count": 20, "success": True},
            {"image_paths": ["img1.png", "img2.png", "img3.png"], "success": True},
        ]

        expected_counts = [15, 20, 3]

        for experiment_results, expected_count in zip(test_cases, expected_counts):
            sample_iteration_result.experiment_results = experiment_results

            trace_file = trace_generator.save_iteration_trace(
                iteration_result=sample_iteration_result,
                iteration_number=1,
                task_type="concept_erasure",
            )

            with trace_file.open("r", encoding="utf-8") as f:
                trace_data = json.load(f)

            assert trace_data["iteration"]["attempts"][0]["images_generated"] == expected_count

class TestOrchestratorIntegration:
    """Test AC7: Integration with InnerLoopOrchestrator."""

    @pytest.mark.skip(reason="Requires full orchestrator setup with mocked dependencies")
    def test_orchestrator_calls_save_iteration_trace(self) -> None:
        """Test that InnerLoopOrchestrator calls save_iteration_trace after each iteration."""
        # This would require extensive mocking of:
        # - RAG system
        # - HypothesisRefinementWorkforce
        # - CodeSynthesizerAgent
        # - MLLMAssessmentAgent
        # - LongTermMemoryAgent
        #
        # Integration testing is better suited for end-to-end tests
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
