"""
Configuration loader for hypothesis generation.

Loads task templates (previously referred to as seed templates), prompts, and other configuration from YAML files.
"""

import random
from pathlib import Path
from typing import Any, Optional

import yaml

from aust.src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ConfigLoader:
    """Loads and manages configuration for hypothesis generation."""

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration loader.

        Args:
            config_dir: Path to config directory (default: aust/configs)
        """
        if config_dir is None:
            # Default to aust/configs
            self.config_dir = Path(__file__).parent.parent.parent / "configs"
        else:
            self.config_dir = Path(config_dir)

        logger.info(f"ConfigLoader initialized with config_dir: {self.config_dir}")

        # Cache loaded configs
        self._task_templates_cache: dict[str, dict] = {}
        self._prompts_cache: dict[str, dict] = {}

    def load_task_templates(self, task_type: str) -> dict[str, Any]:
        """
        Load task templates for a task type.

        Args:
            task_type: Task type (e.g., "data_based_unlearning", "concept_erasure")

        Returns:
            Dictionary with task template configuration

        Raises:
            FileNotFoundError: If template file doesn't exist
            ValueError: If template file is invalid
        """
        if task_type in self._task_templates_cache:
            return self._task_templates_cache[task_type]

        template_file = self.config_dir / "tasks" / f"{task_type}.yaml"

        if not template_file.exists():
            raise FileNotFoundError(
                f"Seed template file not found: {template_file}. "
                f"Available task types: {self.get_available_task_types()}"
            )

        logger.info(f"Loading task templates from {template_file}")

        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # Validate structure
            if not isinstance(config, dict):
                raise ValueError("Seed template file must contain a YAML dictionary")

            seed_templates = config.get("seed_templates") or []

            if not seed_templates:
                # Fallback: load from starter template only
                logger.debug(
                    "No task templates defined in %s; using starter_template.yaml",
                    template_file,
                )
                config["seed_templates"] = self._load_starter_task_templates()
            else:
                # Resolve file references if present
                resolved_templates = []
                for template in seed_templates:
                    if isinstance(template, str) and template.startswith("file:"):
                        # Format: "file:prompts/starter_template.yaml"
                        ref_path = template[5:]  # Remove "file:" prefix
                        resolved_templates.extend(self._load_template_from_file(ref_path))
                    else:
                        resolved_templates.append(template)
                config["seed_templates"] = resolved_templates

            # Cache and return
            self._task_templates_cache[task_type] = config
            logger.info(
                f"Loaded {len(config['seed_templates'])} task templates for {task_type}"
            )
            return config

        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML file {template_file}: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to load task templates: {e}") from e

    def get_task_template(
        self,
        task_type: str,
        iteration: int,
    ) -> Optional[dict]:
        """
        Get a task template for hypothesis generation.

        Args:
            task_type: Task type (e.g., "data_based_unlearning")
            iteration: Current iteration number (1-indexed)

        Returns:
            Selected task template dict, or None if no template should be used

        Raises:
            ValueError: If required templates are not available
        """
        # Only use task templates for iteration 1
        if iteration != 1:
            logger.debug(f"No task template for iteration {iteration} (only iter 1)")
            return None

        config = self.load_task_templates(task_type)
        templates = config["seed_templates"]

        if not templates:
            raise ValueError(f"No task templates available for task type '{task_type}'")

        # Check if template_selection specifies a specific template for iteration 1
        template_selection = config.get("template_selection", {})
        if iteration == 1 and template_selection.get("iteration_1_strategy") == "specific":
            template_id = template_selection.get("iteration_1_template_id")
            if template_id:
                for template in templates:
                    if template.get("id") == template_id:
                        logger.info(f"Selected specific task template for iteration 1: {template_id}")
                        return template
                raise ValueError(
                    f"Configured template id '{template_id}' not found for task type '{task_type}'."
                )

        template = random.choice(templates)
        logger.info(f"Selected task template: {template.get('id', 'unknown')}")
        return template

    def _load_starter_task_templates(self) -> list[dict]:
        """Load task templates from starter_template.yaml."""
        return self._load_template_from_file("prompts/starter_template.yaml")

    def _load_template_from_file(self, relative_path: str) -> list[dict]:
        """
        Load task templates from an external file.

        Args:
            relative_path: Path relative to config_dir (e.g., "prompts/starter_template.yaml")

        Returns:
            List of template dictionaries

        Raises:
            ValueError: If file not found or invalid format
        """
        template_path = self.config_dir / relative_path
        if not template_path.exists():
            raise ValueError(f"Template file not found: {template_path}")

        with open(template_path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f)

        if not isinstance(payload, dict):
            raise ValueError(f"Template file must contain a YAML dictionary: {template_path}")

        # Handle both single template (seed_template) and list (seed_templates)
        if "seed_template" in payload:
            template = payload["seed_template"]
            if not isinstance(template, dict):
                raise ValueError(f"'seed_template' key must be a dictionary in {template_path}")
            template_id = template.get("id", "unknown")
            logger.info(f"Loaded task template '{template_id}' from {template_path}")
            return [template]
        elif "seed_templates" in payload:
            templates = payload["seed_templates"]
            if not isinstance(templates, list):
                raise ValueError(f"'seed_templates' key must be a list in {template_path}")
            logger.info(f"Loaded {len(templates)} task templates from {template_path}")
            return templates
        else:
            raise ValueError(
                f"Template file must contain 'seed_template' or 'seed_templates' key: {template_path}"
            )

    def get_available_task_types(self) -> list[str]:
        """
        Get list of available task types that define templates.

        Returns:
            List of task type names (without .yaml extension)
        """
        tasks_dir = self.config_dir / "tasks"
        if not tasks_dir.exists():
            return []

        task_files = tasks_dir.glob("*.yaml")
        return [f.stem for f in task_files]

    def load_prompt_config(self, agent_name: str, task_type: Optional[str] = None) -> dict:
        """
        Load prompt configuration for an agent.

        Args:
            agent_name: Agent name (e.g., "hypothesis_generator", "critic")
            task_type: Optional task type for task-specific prompts

        Returns:
            Dictionary with prompt configuration

        Raises:
            FileNotFoundError: If prompt file doesn't exist
        """
        # Try task-specific prompt first
        if task_type:
            cache_key = f"{agent_name}_{task_type}"
            if cache_key in self._prompts_cache:
                return self._prompts_cache[cache_key]

            prompt_file = self.config_dir / "prompts" / f"{agent_name}_{task_type}.yaml"
            if prompt_file.exists():
                return self._load_prompt_file(prompt_file, cache_key)

        # Fall back to generic prompt
        cache_key = agent_name
        if cache_key in self._prompts_cache:
            return self._prompts_cache[cache_key]

        prompt_file = self.config_dir / "prompts" / f"{agent_name}.yaml"
        if not prompt_file.exists():
            raise FileNotFoundError(
                f"Prompt configuration not found: {prompt_file}. "
                f"Available agents: {self.get_available_agents()}"
            )

        return self._load_prompt_file(prompt_file, cache_key)

    def _load_prompt_file(self, prompt_file: Path, cache_key: str) -> dict:
        """Load and cache a prompt configuration file."""
        logger.info(f"Loading prompt config from {prompt_file}")

        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            if not isinstance(config, dict):
                raise ValueError("Prompt config file must contain a YAML dictionary")

            # Cache and return
            self._prompts_cache[cache_key] = config
            return config

        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML file {prompt_file}: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to load prompt config: {e}") from e

    def get_available_agents(self) -> list[str]:
        """
        Get list of available agent implementations.

        Returns:
            List of agent module names (without file extension)
        """
        agents_dir = Path(__file__).parent.parent / "agents"
        if not agents_dir.exists():
            return []

        agent_names: set[str] = set()
        for path in agents_dir.iterdir():
            if path.is_file() and path.suffix == ".py" and path.stem != "__init__":
                agent_names.add(path.stem)
            elif path.is_dir() and (path / "__init__.py").exists():
                agent_names.add(path.name)

        return sorted(agent_names)


__all__ = ["ConfigLoader"]

