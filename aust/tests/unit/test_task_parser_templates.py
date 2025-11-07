"""
Unit tests for AC1: Task Parser Seed Template Selection (Story 5.2).

Tests cover:
1. target_type classification in TaskSpec
2. target_type parsing from TaskParserAgent
3. Template loading by target_type
4. Fallback to starter_template.yaml
5. HypothesisRefinementWorkforce integration with seed templates
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from aust.src.agents.task_parser_agent import TaskParserAgent, TaskParserResult
from aust.src.data_models.task_spec import TaskSpec


class TestTargetTypeInTaskSpec:
    """Test AC1: target_type field in TaskSpec data model."""

    def test_task_spec_accepts_target_type(self) -> None:
        """Test that TaskSpec accepts target_type field."""
        spec = TaskSpec(
            task_type="concept_erasure",
            model_name="stable-diffusion",
            model_version="1.4",
            unlearned_model_path="/path/to/model",
            unlearned_target="apple",
            target_type="object",
            user_prompt="Test prompt",
        )
        assert spec.target_type == "object"

    def test_task_spec_target_type_optional(self) -> None:
        """Test that target_type is optional in TaskSpec."""
        spec = TaskSpec(
            task_type="concept_erasure",
            model_name="stable-diffusion",
            unlearned_model_path="/path/to/model",
            unlearned_target="apple",
            user_prompt="Test prompt",
        )
        assert spec.target_type is None

    def test_task_spec_assemble_with_target_type(self) -> None:
        """Test TaskSpec.assemble() preserves target_type from parser result."""
        parser_result = TaskParserResult(
            fields={
                "model_name": "stable-diffusion",
                "model_version": "1.4",
                "unlearned_target": "nudity",
                "unlearning_method": "esd",
                "target_type": "abstract",
            },
            raw_response='{"target_type": "abstract"}',
        )

        spec = TaskSpec.assemble(
            task_type="concept_erasure",
            base_model_path=None,
            unlearned_model_path="/path/to/model",
            user_prompt="Remove nudity from model",
            parser_result=parser_result,
            overrides=None,
        )

        assert spec.target_type == "abstract"
        assert spec.metadata["field_sources"]["target_type"] == "task_parser_agent"

    def test_task_spec_assemble_target_type_cli_override(self) -> None:
        """Test TaskSpec.assemble() prioritizes CLI overrides for target_type."""
        parser_result = TaskParserResult(
            fields={
                "model_name": "stable-diffusion",
                "unlearned_target": "apple",
                "target_type": "abstract",  # Wrong classification from LLM
            },
            raw_response="{}",
        )

        spec = TaskSpec.assemble(
            task_type="concept_erasure",
            base_model_path=None,
            unlearned_model_path="/path/to/model",
            user_prompt="Remove apple from model",
            parser_result=parser_result,
            overrides={"target_type": "object"},  # User corrects it
        )

        assert spec.target_type == "object"
        assert spec.metadata["field_sources"]["target_type"] == "cli"
        assert spec.metadata["overrides_used"]["target_type"] is True


class TestTaskParserTargetTypeClassification:
    """Test AC1: TaskParserAgent classifies target_type."""

    def test_parser_classifies_object_target(self) -> None:
        """Test that TaskParserAgent classifies concrete objects correctly."""
        agent = TaskParserAgent()
        mock_response = MagicMock()
        mock_response.msgs = [
            MagicMock(
                content='{"model_name": "stable-diffusion", "model_version": "1.4", '
                '"unlearned_target": "dog", "unlearning_method": null, '
                '"target_type": "object"}'
            )
        ]

        with patch.object(agent._agent, "step", return_value=mock_response):
            with patch.object(agent._agent, "reset"):
                result = agent.parse_prompt(
                    "Remove dog from Stable Diffusion 1.4", task_type="concept_erasure"
                )

        normalized = result.normalized_dict()
        assert normalized["target_type"] == "object"
        assert normalized["unlearned_target"] == "dog"

    def test_parser_classifies_abstract_target(self) -> None:
        """Test that TaskParserAgent classifies abstract concepts correctly."""
        agent = TaskParserAgent()
        mock_response = MagicMock()
        mock_response.msgs = [
            MagicMock(
                content='{"model_name": "stable-diffusion", '
                '"unlearned_target": "violence", '
                '"target_type": "abstract"}'
            )
        ]

        with patch.object(agent._agent, "step", return_value=mock_response):
            with patch.object(agent._agent, "reset"):
                result = agent.parse_prompt(
                    "Remove violence concept", task_type="concept_erasure"
                )

        normalized = result.normalized_dict()
        assert normalized["target_type"] == "abstract"

    def test_parser_classifies_style_target(self) -> None:
        """Test that TaskParserAgent classifies style targets correctly."""
        agent = TaskParserAgent()
        mock_response = MagicMock()
        mock_response.msgs = [
            MagicMock(
                content='{"model_name": "stable-diffusion", '
                '"unlearned_target": "Van Gogh", '
                '"target_type": "style"}'
            )
        ]

        with patch.object(agent._agent, "step", return_value=mock_response):
            with patch.object(agent._agent, "reset"):
                result = agent.parse_prompt(
                    "Erase Van Gogh style", task_type="concept_erasure"
                )

        normalized = result.normalized_dict()
        assert normalized["target_type"] == "style"

    def test_parser_fallback_classification_when_missing(self) -> None:
        """Test heuristic classification when LLM omits target_type."""
        agent = TaskParserAgent()
        mock_response = MagicMock()
        mock_response.msgs = [
            MagicMock(
                content='{"model_name": "stable-diffusion", '
                '"unlearned_target": "apple", '
                '"target_type": null}'
            )
        ]

        with patch.object(agent._agent, "step", return_value=mock_response):
            with patch.object(agent._agent, "reset"):
                result = agent.parse_prompt(
                    "Erase the apple concept from the model", task_type="concept_erasure"
                )

        normalized = result.normalized_dict()
        assert normalized["target_type"] == "object"


class TestTemplateLoading:
    """Test AC1: Template loading by target_type."""

    def test_load_templates_for_object_type(self, tmp_path: Path) -> None:
        """Test loading templates from object/ subdirectory."""
        # Create temporary template directory structure
        template_dir = tmp_path / "hypothesis"
        object_dir = template_dir / "object"
        object_dir.mkdir(parents=True)

        # Create a test template
        template_data = {
            "seed_template": {
                "id": "test_object_template",
                "summary": "Test object template",
                "target_type": "object",
                "default_hypothesis": {
                    "attack_type": "test_attack",
                    "description": "Test description",
                },
            }
        }
        template_file = object_dir / "test_template.yaml"
        with template_file.open("w") as f:
            yaml.dump(template_data, f)

        # Mock the _HYPOTHESIS_TEMPLATES_DIR to point to our temp directory
        with patch(
            "aust.src.agents.task_parser_agent._HYPOTHESIS_TEMPLATES_DIR", template_dir
        ):
            templates = TaskParserAgent.load_hypothesis_templates("object")

        assert len(templates) == 1
        assert templates[0]["id"] == "test_object_template"
        assert templates[0]["target_type"] == "object"

    def test_load_templates_for_abstract_type(self, tmp_path: Path) -> None:
        """Test loading templates from abstract/ subdirectory."""
        template_dir = tmp_path / "hypothesis"
        abstract_dir = template_dir / "abstract"
        abstract_dir.mkdir(parents=True)

        template_data = {
            "seed_template": {
                "id": "test_abstract_template",
                "summary": "Test abstract template",
                "target_type": "abstract",
            }
        }
        template_file = abstract_dir / "nudity_template.yaml"
        with template_file.open("w") as f:
            yaml.dump(template_data, f)

        with patch(
            "aust.src.agents.task_parser_agent._HYPOTHESIS_TEMPLATES_DIR", template_dir
        ):
            templates = TaskParserAgent.load_hypothesis_templates("abstract")

        assert len(templates) == 1
        assert templates[0]["id"] == "test_abstract_template"

    def test_load_multiple_templates_for_target_type(self, tmp_path: Path) -> None:
        """Test loading multiple templates from the same target_type directory."""
        template_dir = tmp_path / "hypothesis"
        style_dir = template_dir / "style"
        style_dir.mkdir(parents=True)

        # Create two templates
        for i in range(2):
            template_data = {
                "seed_template": {
                    "id": f"style_template_{i}",
                    "summary": f"Style template {i}",
                }
            }
            with (style_dir / f"template_{i}.yaml").open("w") as f:
                yaml.dump(template_data, f)

        with patch(
            "aust.src.agents.task_parser_agent._HYPOTHESIS_TEMPLATES_DIR", template_dir
        ):
            templates = TaskParserAgent.load_hypothesis_templates("style")

        assert len(templates) == 2
        template_ids = {t["id"] for t in templates}
        assert "style_template_0" in template_ids
        assert "style_template_1" in template_ids

    def test_fallback_to_starter_template_on_missing_dir(self, tmp_path: Path) -> None:
        """Test fallback to starter_template.yaml when target_type dir doesn't exist."""
        template_dir = tmp_path / "hypothesis"
        template_dir.mkdir(parents=True)

        # Create only the starter template
        starter_data = {
            "seed_template": {
                "id": "starter_template",
                "summary": "Fallback starter template",
            }
        }
        with (template_dir / "starter_template.yaml").open("w") as f:
            yaml.dump(starter_data, f)

        with patch(
            "aust.src.agents.task_parser_agent._HYPOTHESIS_TEMPLATES_DIR", template_dir
        ):
            templates = TaskParserAgent.load_hypothesis_templates("nonexistent_type")

        assert len(templates) == 1
        assert templates[0]["id"] == "starter_template"

    def test_fallback_to_starter_template_on_empty_dir(self, tmp_path: Path) -> None:
        """Test fallback to starter_template.yaml when target_type dir is empty."""
        template_dir = tmp_path / "hypothesis"
        object_dir = template_dir / "object"
        object_dir.mkdir(parents=True)

        # Create starter template but leave object/ empty
        starter_data = {
            "seed_template": {
                "id": "starter_template",
                "summary": "Fallback starter",
            }
        }
        with (template_dir / "starter_template.yaml").open("w") as f:
            yaml.dump(starter_data, f)

        with patch(
            "aust.src.agents.task_parser_agent._HYPOTHESIS_TEMPLATES_DIR", template_dir
        ):
            templates = TaskParserAgent.load_hypothesis_templates("object")

        assert len(templates) == 1
        assert templates[0]["id"] == "starter_template"

    def test_fallback_to_starter_template_on_none(self, tmp_path: Path) -> None:
        """Test fallback to starter_template.yaml when target_type is None."""
        template_dir = tmp_path / "hypothesis"
        template_dir.mkdir(parents=True)

        starter_data = {
            "seed_template": {
                "id": "starter_template",
                "summary": "Default starter",
            }
        }
        with (template_dir / "starter_template.yaml").open("w") as f:
            yaml.dump(starter_data, f)

        with patch(
            "aust.src.agents.task_parser_agent._HYPOTHESIS_TEMPLATES_DIR", template_dir
        ):
            templates = TaskParserAgent.load_hypothesis_templates(None)

        assert len(templates) == 1
        assert templates[0]["id"] == "starter_template"


class TestHypothesisWorkforceIntegration:
    """Test AC1: HypothesisRefinementWorkforce accepts seed templates."""

    @pytest.mark.skip(reason="Requires mocking CAMEL agents and full environment setup")
    def test_workforce_accepts_seed_templates(self) -> None:
        """Test that HypothesisRefinementWorkforce accepts seed_templates parameter."""
        from aust.src.agents.hypothesis_workforce import HypothesisRefinementWorkforce

        seed_templates = [
            {
                "id": "test_template",
                "summary": "Test template",
                "default_hypothesis": {"attack_type": "test"},
            }
        ]

        # This would require extensive mocking of CAMEL infrastructure
        # and OpenRouter API keys, so we skip in actual test run
        # workforce = HypothesisRefinementWorkforce(seed_templates=seed_templates)
        # assert workforce._seed_templates == seed_templates

    def test_template_rendering_with_multiple_seeds(self) -> None:
        """Test that _render_starter_template() handles multiple seed templates."""
        from aust.src.agents.hypothesis_workforce import HypothesisRefinementWorkforce

        # Create a mock workforce instance with seed templates
        # This tests the rendering logic without requiring full initialization
        seed_templates = [
            {
                "id": "template_1",
                "summary": "First template",
                "default_hypothesis": {
                    "attack_type": "type_1",
                    "description": "Description 1",
                },
            },
            {
                "id": "template_2",
                "summary": "Second template",
                "default_hypothesis": {
                    "attack_type": "type_2",
                    "description": "Description 2",
                },
            },
        ]

        # Mock the instance attributes without full initialization
        mock_workforce = MagicMock(spec=HypothesisRefinementWorkforce)
        mock_workforce._seed_templates = seed_templates
        mock_workforce._normalize_text.side_effect = lambda value: value

        # Call the actual _render_starter_template method
        rendered = HypothesisRefinementWorkforce._render_starter_template(
            mock_workforce, stage="starter"
        )

        # Verify both templates are rendered
        assert "Template 1: template_1" in rendered
        assert "Template 2: template_2" in rendered
        assert "First template" in rendered
        assert "Second template" in rendered
        assert "type_1" in rendered
        assert "type_2" in rendered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
