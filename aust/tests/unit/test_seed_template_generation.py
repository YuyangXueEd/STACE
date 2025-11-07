"""
Unit tests for AC2: Seed Template Generation from Successful Attacks (Story 5.2).

Tests cover:
1. Detecting successful attacks from iteration traces
2. Generating seed templates using LLM
3. Saving templates as YAML files
4. Duplicate detection using similarity
5. Template structure validation
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from aust.src.agents.reporter import ReporterAgent


class TestSuccessfulAttackDetection:
    """Test detection of successful attacks worth saving."""

    def test_check_successful_iteration_true(self, tmp_path: Path) -> None:
        """Test detection of successful attack above threshold."""
        reporter = ReporterAgent(output_dir=tmp_path)

        iteration_trace = {
            "task_id": "test_task",
            "iteration": {
                "iteration_number": 1,
                "vulnerability_detected": True,
                "confidence": 0.85,
            }
        }

        is_successful = reporter.check_successful_iteration(iteration_trace)
        assert is_successful is True

    def test_check_successful_iteration_false_low_confidence(self, tmp_path: Path) -> None:
        """Test that low confidence attacks are not saved."""
        reporter = ReporterAgent(output_dir=tmp_path)

        iteration_trace = {
            "iteration": {
                "vulnerability_detected": True,
                "confidence": 0.75,  # Below default threshold of 0.8
            }
        }

        is_successful = reporter.check_successful_iteration(iteration_trace)
        assert is_successful is False

    def test_check_successful_iteration_false_no_vulnerability(self, tmp_path: Path) -> None:
        """Test that failed attacks are not saved."""
        reporter = ReporterAgent(output_dir=tmp_path)

        iteration_trace = {
            "iteration": {
                "vulnerability_detected": False,
                "confidence": 0.95,  # High confidence but no vulnerability
            }
        }

        is_successful = reporter.check_successful_iteration(iteration_trace)
        assert is_successful is False

    def test_check_successful_iteration_custom_threshold(self, tmp_path: Path) -> None:
        """Test custom confidence threshold."""
        reporter = ReporterAgent(output_dir=tmp_path)

        iteration_trace = {
            "iteration": {
                "vulnerability_detected": True,
                "confidence": 0.75,
            }
        }

        # Should pass with lower threshold
        is_successful = reporter.check_successful_iteration(
            iteration_trace, confidence_threshold=0.7
        )
        assert is_successful is True


class TestSeedTemplateGeneration:
    """Test LLM-based seed template generation."""

    @pytest.fixture
    def mock_prompts_file(self, tmp_path: Path) -> Path:
        """Create mock short report prompts file."""
        prompts_dir = tmp_path / "aust" / "configs" / "prompts"
        prompts_dir.mkdir(parents=True)

        prompt_file = prompts_dir / "short_report_generator.yaml"
        prompts = {
            "system_prompt": "Generate template from: {attack_trace}"
        }

        with prompt_file.open("w", encoding="utf-8") as f:
            yaml.dump(prompts, f)

        return tmp_path

    @pytest.fixture
    def sample_iteration_trace(self) -> dict:
        """Create sample iteration trace."""
        return {
            "task_id": "test_task_123",
            "task_type": "concept_erasure",
            "iteration": {
                "iteration_number": 1,
                "vulnerability_detected": True,
                "confidence": 0.85,
                "hypothesis": {
                    "attack_type": "coref_probing",
                    "description": "Test methodology",
                    "experiment_design": "Test experiment design"
                }
            }
        }

    def test_generate_seed_template_success(
        self, tmp_path: Path, mock_prompts_file: Path, sample_iteration_trace: dict
    ) -> None:
        """Test successful seed template generation."""
        reporter = ReporterAgent(output_dir=tmp_path)

        # Mock LLM response
        mock_agent = MagicMock()
        mock_response = MagicMock()
        mock_message = MagicMock()

        template_json = {
            "task_id": "test_task_123",
            "target_type": "object",
            "attack_type": "coref_probing",
            "methodology": "Test methodology description",
            "experiment_design": "Test experiment design",
            "source_paper": "empirical",
            "confidence_score": "0.85",
            "novelty_score": "0.7"
        }

        mock_message.content = json.dumps(template_json)
        mock_response.msgs = [mock_message]
        mock_agent.step.return_value = mock_response

        with patch.object(reporter, "_get_project_root", return_value=mock_prompts_file):
            with patch("camel.models.ModelFactory"):
                with patch("camel.agents.ChatAgent", return_value=mock_agent):
                    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
                        result = reporter.generate_seed_template(sample_iteration_trace)

        assert result is not None
        assert result["task_id"] == "test_task_123"
        assert result["target_type"] == "object"
        assert result["attack_type"] == "coref_probing"
        assert result["methodology"] == "Test methodology description"

    def test_generate_seed_template_with_markdown_json(
        self, tmp_path: Path, mock_prompts_file: Path, sample_iteration_trace: dict
    ) -> None:
        """Test parsing JSON from markdown code blocks."""
        reporter = ReporterAgent(output_dir=tmp_path)

        mock_agent = MagicMock()
        mock_response = MagicMock()
        mock_message = MagicMock()

        template_json = {"task_id": "test", "target_type": "object", "attack_type": "test"}

        # Wrap in markdown code block
        mock_message.content = f"```json\n{json.dumps(template_json)}\n```"
        mock_response.msgs = [mock_message]
        mock_agent.step.return_value = mock_response

        with patch.object(reporter, "_get_project_root", return_value=mock_prompts_file):
            with patch("camel.models.ModelFactory"):
                with patch("camel.agents.ChatAgent", return_value=mock_agent):
                    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
                        result = reporter.generate_seed_template(sample_iteration_trace)

        assert result is not None
        assert result["task_id"] == "test"

    def test_generate_seed_template_invalid_json(
        self, tmp_path: Path, mock_prompts_file: Path, sample_iteration_trace: dict
    ) -> None:
        """Test handling of invalid JSON response."""
        reporter = ReporterAgent(output_dir=tmp_path)

        mock_agent = MagicMock()
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Invalid JSON {not valid"
        mock_response.msgs = [mock_message]
        mock_agent.step.return_value = mock_response

        with patch.object(reporter, "_get_project_root", return_value=mock_prompts_file):
            with patch("camel.models.ModelFactory"):
                with patch("camel.agents.ChatAgent", return_value=mock_agent):
                    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
                        result = reporter.generate_seed_template(sample_iteration_trace)

        assert result is None

    def test_generate_seed_template_missing_prompts(
        self, tmp_path: Path, sample_iteration_trace: dict
    ) -> None:
        """Test handling of missing prompt file."""
        reporter = ReporterAgent(output_dir=tmp_path)

        result = reporter.generate_seed_template(sample_iteration_trace)
        assert result is None


class TestTemplateSaving:
    """Test saving templates as YAML files."""

    @pytest.fixture
    def sample_seed_template(self) -> dict:
        """Create sample seed template."""
        return {
            "task_id": "attack_abc123",
            "target_type": "object",
            "attack_type": "coref_probing",
            "methodology": "Test methodology for coref probing attack",
            "experiment_design": "Test experiment design with specific parameters",
            "source_paper": "empirical",
            "confidence_score": "0.85",
            "novelty_score": "0.7"
        }

    def test_save_successful_attack_template(
        self, tmp_path: Path, sample_seed_template: dict
    ) -> None:
        """Test saving template to YAML file."""
        reporter = ReporterAgent(output_dir=tmp_path)

        templates_dir = tmp_path / "hypothesis"
        task_context = {"model_name": "SD-1.4", "unlearned_target": "apple"}

        template_file = reporter.save_successful_attack_template(
            sample_seed_template,
            task_context=task_context,
            templates_dir=templates_dir
        )

        assert template_file is not None
        assert template_file.exists()
        assert template_file.parent.name == "object"  # target_type directory
        assert template_file.name == "attack_abc123.yaml"

        # Verify YAML structure
        with template_file.open("r", encoding="utf-8") as f:
            saved_template = yaml.safe_load(f)

        assert "seed_template" in saved_template
        seed = saved_template["seed_template"]

        assert seed["id"] == "attack_abc123"
        assert seed["target_type"] == "object"
        assert seed["source_paper"] == "empirical"
        assert "created_at" in seed
        assert seed["task_context"] == task_context

        # Verify default_hypothesis structure
        hyp = seed["default_hypothesis"]
        assert hyp["attack_type"] == "coref_probing"
        assert hyp["description"] == "Test methodology for coref probing attack"
        assert hyp["experiment_design"] == "Test experiment design with specific parameters"
        assert hyp["confidence_score"] == 0.85
        assert hyp["novelty_score"] == 0.7

    def test_save_template_creates_target_type_directory(
        self, tmp_path: Path, sample_seed_template: dict
    ) -> None:
        """Test that target type directory is created if missing."""
        reporter = ReporterAgent(output_dir=tmp_path)

        templates_dir = tmp_path / "hypothesis"
        assert not (templates_dir / "object").exists()

        template_file = reporter.save_successful_attack_template(
            sample_seed_template,
            templates_dir=templates_dir
        )

        assert template_file is not None
        assert (templates_dir / "object").exists()
        assert (templates_dir / "object").is_dir()


class TestDuplicateDetection:
    """Test duplicate template detection using similarity."""

    def test_is_duplicate_template_no_existing(self, tmp_path: Path) -> None:
        """Test that first template is not considered duplicate."""
        reporter = ReporterAgent(output_dir=tmp_path)

        new_template = {
            "methodology": "New methodology",
            "experiment_design": "New experiment"
        }

        target_dir = tmp_path / "templates"
        target_dir.mkdir()

        is_duplicate = reporter._is_duplicate_template(new_template, target_dir)
        assert is_duplicate is False


class TestTemplateCaptureFromTraces:
    """Test save_successful_templates_from_traces helper."""

    @pytest.fixture
    def successful_iteration_trace(self) -> dict:
        return {
            "task_id": "test_task_123",
            "task_type": "concept_erasure",
            "iteration": {
                "iteration_number": 1,
                "vulnerability_detected": True,
                "confidence": 0.9,
                "hypothesis": {
                    "attack_type": "coref_probing",
                    "description": "Method description",
                    "experiment_design": "Experiment plan",
                },
            },
        }

    def test_successful_attack_saved(self, tmp_path: Path, successful_iteration_trace: dict) -> None:
        """Verify that successful attacks are converted to templates."""
        reporter = ReporterAgent(output_dir=tmp_path)

        traces_dir = tmp_path / "attack_traces"
        traces_dir.mkdir()
        trace_file = traces_dir / "attack_trace_iter_01.json"
        trace_file.write_text(json.dumps(successful_iteration_trace), encoding="utf-8")

        mock_template = {
            "task_id": "attack_success_01",
            "target_type": "object",
            "methodology": "Mock methodology",
            "experiment_design": "Mock experiment",
        }
        saved_path = tmp_path / "aust" / "configs" / "hypothesis" / "object" / "attack_success_01.yaml"

        with patch.object(reporter, "generate_seed_template", return_value=mock_template) as mock_generate:
            with patch.object(
                reporter,
                "save_successful_attack_template",
                return_value=saved_path,
            ) as mock_save:
                saved = reporter.save_successful_templates_from_traces(
                    "test_task_123",
                    traces_dir=traces_dir,
                    task_context={"task_id": "test_task_123"},
                )

        assert saved == [saved_path]
        mock_generate.assert_called_once()
        mock_save.assert_called_once()

    def test_no_successful_attacks_returns_empty(self, tmp_path: Path, successful_iteration_trace: dict) -> None:
        """Verify that non-successful traces are ignored."""
        reporter = ReporterAgent(output_dir=tmp_path)

        traces_dir = tmp_path / "attack_traces"
        traces_dir.mkdir()

        failing_trace = dict(successful_iteration_trace)
        failing_trace["iteration"] = dict(successful_iteration_trace["iteration"])
        failing_trace["iteration"]["vulnerability_detected"] = False
        failing_trace["iteration"]["confidence"] = 0.2

        (traces_dir / "attack_trace_iter_01.json").write_text(
            json.dumps(failing_trace),
            encoding="utf-8",
        )

        with patch.object(reporter, "generate_seed_template") as mock_generate:
            saved = reporter.save_successful_templates_from_traces(
                "test_task_123",
                traces_dir=traces_dir,
                task_context={"task_id": "test_task_123"},
            )

        assert saved == []
        mock_generate.assert_not_called()

    def test_is_duplicate_template_high_similarity(self, tmp_path: Path) -> None:
        """Test detection of highly similar template."""
        reporter = ReporterAgent(output_dir=tmp_path)

        target_dir = tmp_path / "templates"
        target_dir.mkdir()

        # Create existing template
        existing = {
            "seed_template": {
                "id": "existing_1",
                "default_hypothesis": {
                    "description": "Test methodology using coref probing",
                    "experiment_design": "Generate images with specific prompts"
                }
            }
        }

        existing_file = target_dir / "existing_1.yaml"
        with existing_file.open("w", encoding="utf-8") as f:
            yaml.dump(existing, f)

        # New template with high similarity
        new_template = {
            "methodology": "Test methodology using coref probing",
            "experiment_design": "Generate images with specific prompts"
        }

        is_duplicate = reporter._is_duplicate_template(new_template, target_dir)
        assert is_duplicate is True

    def test_is_duplicate_template_low_similarity(self, tmp_path: Path) -> None:
        """Test that different templates are not considered duplicates."""
        reporter = ReporterAgent(output_dir=tmp_path)

        target_dir = tmp_path / "templates"
        target_dir.mkdir()

        # Create existing template
        existing = {
            "seed_template": {
                "default_hypothesis": {
                    "description": "Completely different approach",
                    "experiment_design": "Different experiment design"
                }
            }
        }

        existing_file = target_dir / "existing_1.yaml"
        with existing_file.open("w", encoding="utf-8") as f:
            yaml.dump(existing, f)

        # New template with low similarity
        new_template = {
            "methodology": "Novel methodology using advanced techniques",
            "experiment_design": "Innovative experiment with unique parameters"
        }

        is_duplicate = reporter._is_duplicate_template(new_template, target_dir)
        assert is_duplicate is False

    def test_calculate_text_similarity(self, tmp_path: Path) -> None:
        """Test text similarity calculation."""
        reporter = ReporterAgent(output_dir=tmp_path)

        # Identical text
        sim = reporter._calculate_text_similarity("hello world", "hello world")
        assert sim == 1.0

        # Completely different
        sim = reporter._calculate_text_similarity("hello world", "foo bar")
        assert sim == 0.0

        # Partial overlap
        sim = reporter._calculate_text_similarity("hello world", "hello foo")
        assert 0.0 < sim < 1.0

        # Empty strings
        sim = reporter._calculate_text_similarity("", "hello")
        assert sim == 0.0


class TestEndToEndWorkflow:
    """Test end-to-end template generation workflow."""

    @pytest.fixture
    def setup_environment(self, tmp_path: Path):
        """Set up test environment with all required files."""
        # Create prompt file
        prompts_dir = tmp_path / "aust" / "configs" / "prompts"
        prompts_dir.mkdir(parents=True)

        prompt_file = prompts_dir / "short_report_generator.yaml"
        prompts = {"system_prompt": "Generate from: {attack_trace}"}

        with prompt_file.open("w", encoding="utf-8") as f:
            yaml.dump(prompts, f)

        return tmp_path

    def test_full_workflow_successful_attack(
        self, tmp_path: Path, setup_environment: Path
    ) -> None:
        """Test complete workflow from detection to saving."""
        reporter = ReporterAgent(output_dir=tmp_path)

        # Step 1: Detect successful attack
        iteration_trace = {
            "task_id": "test_task",
            "iteration": {
                "vulnerability_detected": True,
                "confidence": 0.85,
            }
        }

        is_successful = reporter.check_successful_iteration(iteration_trace)
        assert is_successful is True

        # Step 2: Generate seed template (mocked)
        mock_template = {
            "task_id": "test_task",
            "target_type": "object",
            "attack_type": "test_attack",
            "methodology": "Test method",
            "experiment_design": "Test design",
            "source_paper": "empirical",
            "confidence_score": "0.85",
            "novelty_score": "0.7"
        }

        # Step 3: Save template
        templates_dir = tmp_path / "hypothesis"
        template_file = reporter.save_successful_attack_template(
            mock_template,
            task_context={"model": "test"},
            templates_dir=templates_dir
        )

        assert template_file is not None
        assert template_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
