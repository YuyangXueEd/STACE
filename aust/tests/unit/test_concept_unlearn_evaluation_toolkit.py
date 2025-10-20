"""
Unit tests for ConceptUnlearnEvaluationToolkit.

Tests cover:
- ASR computation
- CLIP-based concept leakage detection
- FID computation
- CLIP Score computation
- Prompt Perplexity computation
- Report generation
- Error handling
"""

import pytest
import torch
import tempfile
import json
from pathlib import Path
from PIL import Image
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from aust.src.toolkits.concept_unlearn_evaluation_toolkit import (
    ConceptUnlearnEvaluationToolkit,
    EvaluationMetrics,
)


@pytest.fixture
def toolkit():
    """Create GPU-backed toolkit instance for testing."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for ConceptUnlearnEvaluationToolkit tests")

    with patch("aust.src.toolkits.concept_unlearn_evaluation_toolkit.CLIPProcessor"):
        with patch("aust.src.toolkits.concept_unlearn_evaluation_toolkit.CLIPModel"):
            with patch("aust.src.toolkits.concept_unlearn_evaluation_toolkit.GPT2Tokenizer"):
                with patch("aust.src.toolkits.concept_unlearn_evaluation_toolkit.GPT2LMHeadModel"):
                    toolkit = ConceptUnlearnEvaluationToolkit(device="cuda")
                    return toolkit


@pytest.fixture
def temp_image():
    """Create temporary test image."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        # Create simple test image
        img = Image.new("RGB", (256, 256), color="red")
        img.save(f.name)
        yield f.name
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_image_dir():
    """Create temporary directory with test images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        # Create test images
        for i in range(3):
            img = Image.new("RGB", (256, 256), color=("red", "green", "blue")[i])
            img.save(tmpdir_path / f"image_{i}.png")
        yield tmpdir_path


class TestEvaluationMetrics:
    """Test EvaluationMetrics dataclass."""

    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = EvaluationMetrics()
        assert metrics.asr is None
        assert metrics.clip_similarity is None
        assert metrics.fid is None

    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = EvaluationMetrics(
            asr=0.25,
            clip_similarity=0.75,
            fid=50.0,
            clip_score=0.85,
            ppl=15.2,
        )
        result = metrics.to_dict()

        assert result["robustness"]["asr"] == 0.25
        assert result["robustness"]["clip_similarity"] == 0.75
        assert result["utility"]["fid"] == 50.0
        assert result["utility"]["clip_score"] == 0.85
        assert result["utility"]["ppl"] == 15.2


class TestConceptUnlearnEvaluationToolkit:
    """Test ConceptUnlearnEvaluationToolkit class."""

    def test_toolkit_initialization(self, toolkit):
        """Test toolkit initialization."""
        assert toolkit.device == "cuda"
        assert toolkit.clip_processor is not None
        assert toolkit.clip_model is not None
        assert toolkit.lm_tokenizer is not None
        assert toolkit.lm_model is not None

    def test_compute_attack_success_rate(self, toolkit):
        """Test ASR computation."""
        # Test normal case
        asr = toolkit.compute_attack_success_rate(
            detected_count=25,
            total_count=100,
        )
        assert asr == 0.25

        # Test zero total count
        asr_zero = toolkit.compute_attack_success_rate(
            detected_count=0,
            total_count=0,
        )
        assert asr_zero == 0.0

        # Test 100% detection
        asr_full = toolkit.compute_attack_success_rate(
            detected_count=100,
            total_count=100,
        )
        assert asr_full == 1.0

    def test_detect_concept_leakage_clip(self, toolkit, temp_image):
        """Test CLIP-based concept leakage detection."""
        # Mock CLIP model outputs
        mock_outputs = MagicMock()
        mock_image_embeds = torch.randn(1, 512)
        mock_text_embeds = torch.randn(1, 512)
        # Normalize for consistent similarity
        mock_image_embeds = mock_image_embeds / mock_image_embeds.norm(p=2, dim=-1, keepdim=True)
        mock_text_embeds = mock_text_embeds / mock_text_embeds.norm(p=2, dim=-1, keepdim=True)

        mock_outputs.image_embeds = mock_image_embeds
        mock_outputs.text_embeds = mock_text_embeds

        toolkit.clip_model.return_value = mock_outputs

        similarity = toolkit.detect_concept_leakage_clip(
            temp_image,
            "nudity",
        )

        assert isinstance(similarity, float)
        assert -1.0 <= similarity <= 1.0

    def test_detect_concept_leakage_detector_not_available(self, toolkit):
        """Test detector-based leakage when detectors not available."""
        result = toolkit.detect_concept_leakage_detector(
            "test.png",
            detector_type="nudenet",
        )

        assert "error" in result or "status" in result

    def test_compute_fid_missing_directory(self, toolkit):
        """Test FID computation with missing directory."""
        with pytest.raises(FileNotFoundError):
            toolkit.compute_fid(
                "/nonexistent/gen",
                "/nonexistent/ref",
            )

    def test_compute_fid(self, toolkit, temp_image_dir):
        """Test FID computation with valid directories."""
        # Create second directory
        with tempfile.TemporaryDirectory() as tmpdir2:
            tmpdir2_path = Path(tmpdir2)
            for i in range(3):
                img = Image.new("RGB", (256, 256), color="white")
                img.save(tmpdir2_path / f"ref_{i}.png")

            # Mock FID metric
            toolkit.fid_metric.compute = Mock(return_value=torch.tensor(25.0))

            fid = toolkit.compute_fid(
                str(temp_image_dir),
                str(tmpdir2_path),
            )

            assert isinstance(fid, float)
            assert fid >= 0.0

    def test_compute_clip_score(self, toolkit, temp_image):
        """Test CLIP Score computation."""
        # Mock CLIP model
        mock_outputs = MagicMock()
        mock_image_embeds = torch.randn(1, 512)
        mock_text_embeds = torch.randn(1, 512)
        mock_image_embeds = mock_image_embeds / mock_image_embeds.norm(p=2, dim=-1, keepdim=True)
        mock_text_embeds = mock_text_embeds / mock_text_embeds.norm(p=2, dim=-1, keepdim=True)

        mock_outputs.image_embeds = mock_image_embeds
        mock_outputs.text_embeds = mock_text_embeds
        toolkit.clip_model.return_value = mock_outputs

        clip_score = toolkit.compute_clip_score(
            temp_image,
            "a red image",
        )

        assert isinstance(clip_score, float)
        assert -1.0 <= clip_score <= 1.0

    def test_compute_prompt_perplexity(self, toolkit):
        """Test prompt perplexity computation."""
        # Mock LM model
        toolkit.lm_model.return_value = MagicMock(loss=torch.tensor(2.5))

        ppl = toolkit.compute_prompt_perplexity("test prompt")

        assert isinstance(ppl, float)
        assert ppl > 0.0

    def test_generate_evaluation_report(self, toolkit):
        """Test evaluation report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_dict = {
                "robustness": {"asr": 0.25, "clip_similarity": 0.75},
                "utility": {"fid": 50.0, "clip_score": 0.85, "ppl": 15.2},
            }

            output_path = Path(tmpdir) / "report"
            toolkit.generate_evaluation_report(metrics_dict, str(output_path))

            # Check JSON report
            json_path = output_path.with_suffix(".json")
            assert json_path.exists()
            with open(json_path) as f:
                loaded = json.load(f)
                assert loaded == metrics_dict

            # Check Markdown report
            md_path = output_path.with_suffix(".md")
            assert md_path.exists()
            content = md_path.read_text()
            assert "Robustness Metrics" in content
            assert "Utility Metrics" in content

    def test_get_tools(self, toolkit):
        """Test getting toolkit tools."""
        tools = toolkit.get_tools()

        assert len(tools) >= 6  # At least 6 tools (excluding optional detector)
        tool_names = [t.func.__name__ for t in tools]

        assert "compute_attack_success_rate" in tool_names
        assert "detect_concept_leakage_clip" in tool_names
        assert "compute_fid" in tool_names
        assert "compute_clip_score" in tool_names
        assert "compute_prompt_perplexity" in tool_names
        assert "generate_evaluation_report" in tool_names


class TestErrorHandling:
    """Test error handling in toolkit."""

    def test_clip_similarity_invalid_image(self, toolkit):
        """Test CLIP similarity with invalid image path."""
        with pytest.raises(Exception):
            toolkit.detect_concept_leakage_clip(
                "/nonexistent/image.png",
                "concept",
            )

    def test_perplexity_empty_prompt(self, toolkit):
        """Test perplexity with empty prompt."""
        # Mock LM model
        toolkit.lm_model.return_value = MagicMock(loss=torch.tensor(0.0))

        ppl = toolkit.compute_prompt_perplexity("")
        assert isinstance(ppl, float)
