"""
Unit tests for NudeNet Toolkit integration.

Tests the NudeNetToolkit wrapper for objective nudity detection.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from aust.src.toolkits.nudenet_toolkit import NudeNetToolkit


class TestNudeNetToolkitInitialization:
    """Test NudeNet toolkit initialization."""

    @patch("aust.src.toolkits.nudenet_toolkit.NudeDetector")
    def test_init_success(self, mock_detector_class):
        """Test successful initialization."""
        mock_detector_class.return_value = Mock()

        toolkit = NudeNetToolkit()

        assert toolkit.detector is not None
        mock_detector_class.assert_called_once()

    @patch("aust.src.toolkits.nudenet_toolkit.NudeDetector",
           side_effect=ImportError("NudeNet not installed"))
    def test_init_import_error(self, mock_detector_class):
        """Test initialization handles ImportError gracefully."""
        toolkit = NudeNetToolkit()

        assert toolkit.detector is None

    @patch("aust.src.toolkits.nudenet_toolkit.NudeDetector",
           side_effect=Exception("Detector init failed"))
    def test_init_general_error(self, mock_detector_class):
        """Test initialization handles general errors gracefully."""
        toolkit = NudeNetToolkit()

        assert toolkit.detector is None


class TestNudeNetDetection:
    """Test nudity detection functionality."""

    def test_detect_nudity_no_detector(self):
        """Test detection returns error when detector unavailable."""
        toolkit = NudeNetToolkit()
        toolkit.detector = None

        result = toolkit.detect_nudity("test.jpg")

        assert "error" in result
        assert not result["nudity_detected"]
        assert result["confidence"] == 0.0

    @patch("aust.src.toolkits.nudenet_toolkit.NudeDetector")
    def test_detect_nudity_exposed_content(self, mock_detector_class):
        """Test detection of exposed nudity."""
        mock_detector = Mock()
        mock_detector.detect.return_value = [
            {"class": "FEMALE_BREAST_EXPOSED", "score": 0.92, "box": [0, 0, 10, 10]},
            {"class": "FACE_FEMALE", "score": 0.85, "box": [20, 20, 30, 30]},
        ]
        mock_detector_class.return_value = mock_detector

        toolkit = NudeNetToolkit()
        result = toolkit.detect_nudity("test.jpg")

        assert result["nudity_detected"] is True
        assert result["confidence"] == 0.92
        assert result["exposed_count"] == 1
        assert len(result["detections"]) == 2

    @patch("aust.src.toolkits.nudenet_toolkit.NudeDetector")
    def test_detect_nudity_covered_only(self, mock_detector_class):
        """Test detection with only covered content (not nudity)."""
        mock_detector = Mock()
        mock_detector.detect.return_value = [
            {"class": "FEMALE_BREAST_COVERED", "score": 0.88, "box": [0, 0, 10, 10]},
            {"class": "FACE_FEMALE", "score": 0.90, "box": [20, 20, 30, 30]},
        ]
        mock_detector_class.return_value = mock_detector

        toolkit = NudeNetToolkit()
        result = toolkit.detect_nudity("test.jpg")

        assert result["nudity_detected"] is False
        assert result["confidence"] == 0.0
        assert result["exposed_count"] == 0

    @patch("aust.src.toolkits.nudenet_toolkit.NudeDetector")
    def test_detect_nudity_detection_error(self, mock_detector_class):
        """Test handling of detection errors."""
        mock_detector = Mock()
        mock_detector.detect.side_effect = Exception("Detection failed")
        mock_detector_class.return_value = mock_detector

        toolkit = NudeNetToolkit()
        result = toolkit.detect_nudity("test.jpg")

        assert "error" in result
        assert not result["nudity_detected"]


class TestNudeNetBatchDetection:
    """Test batch nudity detection."""

    def test_detect_nudity_batch_no_detector(self):
        """Test batch detection without detector."""
        toolkit = NudeNetToolkit()
        toolkit.detector = None

        result = toolkit.detect_nudity_batch(["img1.jpg", "img2.jpg"])

        assert "error" in result
        assert result["total_images"] == 2
        assert result["nudity_detected_count"] == 0

    @patch("aust.src.toolkits.nudenet_toolkit.NudeDetector")
    def test_detect_nudity_batch_mixed_results(self, mock_detector_class):
        """Test batch detection with mixed results."""
        mock_detector = Mock()

        # Mock different results for different images
        def side_effect_detect(image_path):
            if "img1" in image_path:
                return [{"class": "FEMALE_BREAST_EXPOSED", "score": 0.90, "box": [0, 0, 10, 10]}]
            elif "img2" in image_path:
                return []  # No nudity
            else:
                return [{"class": "MALE_GENITALIA_EXPOSED", "score": 0.85, "box": [0, 0, 10, 10]}]

        mock_detector.detect.side_effect = side_effect_detect
        mock_detector_class.return_value = mock_detector

        toolkit = NudeNetToolkit()
        result = toolkit.detect_nudity_batch(["img1.jpg", "img2.jpg", "img3.jpg"])

        assert result["total_images"] == 3
        assert result["nudity_detected_count"] == 2
        assert result["detection_rate"] == pytest.approx(2/3)
        assert result["max_confidence"] == 0.90
        assert len(result["per_image_results"]) == 3

    @patch("aust.src.toolkits.nudenet_toolkit.NudeDetector")
    def test_detect_nudity_batch_empty_list(self, mock_detector_class):
        """Test batch detection with empty image list."""
        mock_detector_class.return_value = Mock()

        toolkit = NudeNetToolkit()
        result = toolkit.detect_nudity_batch([])

        assert result["total_images"] == 0
        assert result["nudity_detected_count"] == 0
        assert result["detection_rate"] == 0.0
        assert result["avg_confidence"] == 0.0


class TestMLLMEvaluatorIntegration:
    """Test integration with MLLM Evaluator."""

    def test_is_nudity_task(self):
        """Test nudity task detection."""
        from aust.src.agents.mllm_evaluator import MLLMEvaluatorAgent

        agent = MLLMEvaluatorAgent()

        assert agent._is_nudity_task("nudity") is True
        assert agent._is_nudity_task("NUDITY concept") is True
        assert agent._is_nudity_task("nsfw content") is True
        assert agent._is_nudity_task("explicit images") is True
        assert agent._is_nudity_task("Van Gogh style") is False
        assert agent._is_nudity_task("violence") is False
