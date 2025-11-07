"""
Unit tests for MLLM Evaluator Agent.

Tests cover:
- MLLMAssessmentAgent initialization
- Concept leakage assessment
- Batch assessment
- MLLMEvaluator workflow orchestration
- Error handling
"""

import pytest
import tempfile
from pathlib import Path
from PIL import Image
from unittest.mock import Mock, patch, MagicMock

from aust.src.agents.mllm_evaluator import (
    MLLMAssessmentAgent,
    MLLMAssessmentResult,
    MLLMEvaluator,
    EvaluationInputs,
)
from aust.src.toolkits.concept_unlearn_evaluation_toolkit import EvaluationMetrics


@pytest.fixture
def mllm_agent():
    """Create MLLM assessment agent for testing."""
    with patch("aust.src.agents.mllm_evaluator.ModelFactory"):
        agent = MLLMAssessmentAgent(vlm_model="gpt-5-nano", device="cpu")
        return agent


@pytest.fixture
def evaluator():
    """Create MLLM evaluator for testing."""
    with patch("aust.src.agents.mllm_evaluator.ConceptUnlearnEvaluationToolkit"):
        with patch("aust.src.agents.mllm_evaluator.ImageAnalysisToolkit"):
            with patch("aust.src.agents.mllm_evaluator.MLLMAssessmentAgent"):
                evaluator = MLLMEvaluator(device="cpu")
                return evaluator


@pytest.fixture
def temp_image():
    """Create temporary test image."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new("RGB", (256, 256), color="red")
        img.save(f.name)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


class TestMLLMAssessmentResult:
    """Test MLLMAssessmentResult dataclass."""

    def test_result_initialization(self):
        """Test result initialization."""
        result = MLLMAssessmentResult(
            concept="nudity",
            detected=True,
            confidence=0.95,
            explanation="Image contains explicit content",
            model_used="gpt-4v",
        )

        assert result.concept == "nudity"
        assert result.detected is True
        assert result.confidence == 0.95
        assert "explicit content" in result.explanation
        assert result.model_used == "gpt-4v"


class TestMLLMAssessmentAgent:
    """Test MLLMAssessmentAgent class."""

    def test_agent_initialization(self, mllm_agent):
        """Test agent initialization."""
        assert mllm_agent.vlm_model == "gpt-5-nano"
        assert mllm_agent.device == "cpu"
        assert mllm_agent.model is not None

    def test_image_to_base64(self, mllm_agent, temp_image):
        """Test image to base64 conversion."""
        base64_str = mllm_agent._image_to_base64(temp_image)

        assert isinstance(base64_str, str)
        assert len(base64_str) > 0

    def test_assess_concept_leakage(self, mllm_agent, temp_image):
        """Test concept leakage assessment."""
        # Mock VLM query response
        mock_response = '{"detected": true, "confidence": 0.85, "explanation": "Test explanation"}'
        mllm_agent._query_vlm = Mock(return_value=mock_response)

        result = mllm_agent.assess_concept_leakage(
            temp_image,
            "nudity",
        )

        assert isinstance(result, MLLMAssessmentResult)
        assert result.concept == "nudity"
        assert result.detected is True
        assert result.confidence == 0.85
        assert "Test explanation" in result.explanation

    def test_assess_concept_leakage_invalid_json(self, mllm_agent, temp_image):
        """Test assessment with invalid JSON response."""
        # Mock VLM query with non-JSON response
        mock_response = "The image does not contain nudity."
        mllm_agent._query_vlm = Mock(return_value=mock_response)

        result = mllm_agent.assess_concept_leakage(
            temp_image,
            "nudity",
        )

        assert isinstance(result, MLLMAssessmentResult)
        # Should fallback to text parsing
        assert result.confidence == 0.5  # Default uncertainty

    def test_batch_assess(self, mllm_agent, temp_image):
        """Test batch assessment."""
        # Create multiple test images
        image_paths = [temp_image] * 3

        # Mock assess_concept_leakage to return varied results
        mock_results = [
            MLLMAssessmentResult("nudity", True, 0.9, "Explicit", "gpt-4v"),
            MLLMAssessmentResult("nudity", False, 0.2, "Safe", "gpt-4v"),
            MLLMAssessmentResult("nudity", True, 0.7, "Borderline", "gpt-4v"),
        ]

        with patch.object(mllm_agent, 'assess_concept_leakage', side_effect=mock_results):
            results, avg_conf, detection_rate = mllm_agent.batch_assess(
                image_paths,
                "nudity",
            )

            assert len(results) == 3
            assert 0.0 <= avg_conf <= 1.0
            assert detection_rate == 2/3  # 2 detected out of 3


class TestEvaluationInputs:
    """Test EvaluationInputs dataclass."""

    def test_inputs_initialization(self):
        """Test inputs initialization."""
        inputs = EvaluationInputs(
            unlearned_model_path="/path/to/model",
            target_concept="violence",
            test_prompts=["prompt1", "prompt2"],
        )

        assert inputs.unlearned_model_path == "/path/to/model"
        assert inputs.target_concept == "violence"
        assert len(inputs.test_prompts) == 2
        assert inputs.vlm_backend == "gpt-5-nano"  # Default


class TestMLLMEvaluator:
    """Test MLLMEvaluator class."""

    def test_evaluator_initialization(self, evaluator):
        """Test evaluator initialization."""
        assert evaluator.device == "cpu"
        assert evaluator.eval_toolkit is not None
        assert evaluator.image_toolkit is not None
        assert evaluator.mllm_agent is not None

    def test_evaluate_workflow(self, evaluator):
        """Test full evaluation workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            inputs = EvaluationInputs(
                unlearned_model_path="/path/to/model",
                target_concept="nudity",
                test_prompts=["a person", "a photo"],
            )

            # Mock toolkit methods
            evaluator.eval_toolkit.detect_concept_leakage_clip = Mock(return_value=0.5)
            evaluator.eval_toolkit.compute_fid = Mock(return_value=50.0)
            evaluator.eval_toolkit.compute_prompt_perplexity = Mock(return_value=15.0)
            evaluator.eval_toolkit.generate_evaluation_report = Mock()

            # Mock MLLM agent
            evaluator.mllm_agent.batch_assess = Mock(
                return_value=([], 0.7, 0.3)  # results, avg_conf, detection_rate
            )

            # Create dummy generated directory
            gen_dir = Path(tmpdir) / "generated"
            gen_dir.mkdir()
            # Create dummy image
            img = Image.new("RGB", (64, 64))
            img.save(gen_dir / "test.png")

            metrics = evaluator.evaluate(inputs, tmpdir)

            assert isinstance(metrics, EvaluationMetrics)
            # ASR should be set from detection rate
            assert metrics.asr == 0.3

    def test_evaluate_no_generated_images(self, evaluator):
        """Test evaluation with no generated images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            inputs = EvaluationInputs(
                unlearned_model_path="/path/to/model",
                target_concept="nudity",
            )

            # Mock methods
            evaluator.eval_toolkit.generate_evaluation_report = Mock()

            metrics = evaluator.evaluate(inputs, tmpdir)

            assert isinstance(metrics, EvaluationMetrics)
            # Metrics should be None or default values when no images
            assert metrics.clip_similarity is None or metrics.clip_similarity == 0.0


class TestErrorHandling:
    """Test error handling in MLLM evaluator."""

    def test_assess_invalid_image_path(self, mllm_agent):
        """Test assessment with invalid image path."""
        with pytest.raises(Exception):
            mllm_agent.assess_concept_leakage(
                "/nonexistent/image.png",
                "concept",
            )

    def test_evaluate_missing_output_dir_creates_it(self, evaluator):
        """Test that evaluate creates output directory if missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "subdir" / "output"

            inputs = EvaluationInputs(
                unlearned_model_path="/path/to/model",
                target_concept="nudity",
            )

            # Mock methods to prevent actual computation
            evaluator.eval_toolkit.generate_evaluation_report = Mock()

            evaluator.evaluate(inputs, str(output_dir))

            assert output_dir.exists()


# ============================================================================
# Story 5.1: Confidence Threshold Tests
# ============================================================================


class TestMLLMConfidenceThreshold:
    """Test 80% confidence threshold enforcement (Story 5.1)."""

    def test_high_confidence_detection_accepted(self, mllm_agent):
        """Test that detections with confidence >= 0.8 are accepted."""
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp_file:
            # Create dummy image
            from PIL import Image

            img = Image.new("RGB", (100, 100), color="white")
            img.save(tmp_file.name)

            # Mock VLM response with high confidence detection
            mllm_agent._query_vlm = Mock(
                return_value='{"detected": true, "confidence": 0.85, "explanation": "Clear evidence"}'
            )

            result = mllm_agent.assess_concept_leakage(tmp_file.name, "nudity")

            assert result.detected is True  # Should accept
            assert result.confidence == 0.85

    def test_low_confidence_detection_rejected(self, mllm_agent):
        """Test that detections with confidence < 0.8 are rejected."""
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp_file:
            from PIL import Image

            img = Image.new("RGB", (100, 100), color="white")
            img.save(tmp_file.name)

            # Mock VLM response with low confidence detection
            mllm_agent._query_vlm = Mock(
                return_value='{"detected": true, "confidence": 0.75, "explanation": "Unclear evidence"}'
            )

            result = mllm_agent.assess_concept_leakage(tmp_file.name, "nudity")

            assert result.detected is False  # Should reject despite detected=true
            assert result.confidence == 0.75  # Confidence preserved for logging

    def test_exact_threshold_detection_accepted(self, mllm_agent):
        """Test that detection at exactly 0.8 confidence is accepted."""
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp_file:
            from PIL import Image

            img = Image.new("RGB", (100, 100), color="white")
            img.save(tmp_file.name)

            # Mock VLM response with threshold confidence
            mllm_agent._query_vlm = Mock(
                return_value='{"detected": true, "confidence": 0.8, "explanation": "Threshold case"}'
            )

            result = mllm_agent.assess_concept_leakage(tmp_file.name, "nudity")

            assert result.detected is True  # Should accept at threshold
            assert result.confidence == 0.8

    def test_non_detection_unaffected_by_threshold(self, mllm_agent):
        """Test that non-detections are unaffected by confidence threshold."""
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp_file:
            from PIL import Image

            img = Image.new("RGB", (100, 100), color="white")
            img.save(tmp_file.name)

            # Mock VLM response with non-detection (low confidence)
            mllm_agent._query_vlm = Mock(
                return_value='{"detected": false, "confidence": 0.3, "explanation": "No evidence"}'
            )

            result = mllm_agent.assess_concept_leakage(tmp_file.name, "nudity")

            assert result.detected is False  # Should remain false
            assert result.confidence == 0.3

    def test_very_low_confidence_detection_rejected(self, mllm_agent):
        """Test that very low confidence detections are clearly rejected."""
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp_file:
            from PIL import Image

            img = Image.new("RGB", (100, 100), color="white")
            img.save(tmp_file.name)

            # Mock VLM response with very low confidence
            mllm_agent._query_vlm = Mock(
                return_value='{"detected": true, "confidence": 0.5, "explanation": "Uncertain"}'
            )

            result = mllm_agent.assess_concept_leakage(tmp_file.name, "nudity")

            assert result.detected is False  # Should reject
            assert result.confidence == 0.5
