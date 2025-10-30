"""Unit tests for infrastructure setup validation."""

from pathlib import Path

import pytest


def test_directory_structure_exists() -> None:
    """Test that all required directories exist."""
    required_dirs = [
        "aust/src/agents",
        "aust/src/toolkits",
        "aust/src/rag",
        "aust/src/memory",
        "aust/src/loop",
        "aust/configs/prompts",
        "aust/configs/thresholds",
        "aust/configs/tasks",
        "aust/configs/personas",
        "outputs",
        "logs",
        "aust/outputs",
        "aust/experiments",
        "aust/tests/unit",
        "aust/tests/integration",
        "aust/scripts",
        "aust/logs",
        "aust/utils",
        "aust/rag_paper_db",
    ]

    project_root = Path(__file__).parent.parent.parent.parent
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        assert full_path.exists(), f"Directory {dir_path} does not exist"
        assert full_path.is_dir(), f"{dir_path} is not a directory"


def test_python_package_init_files_exist() -> None:
    """Test that __init__.py files exist in Python packages."""
    package_dirs = [
        "aust/src",
        "aust/src/agents",
        "aust/src/toolkits",
        "aust/src/rag",
        "aust/src/memory",
        "aust/src/loop",
        "aust/tests",
        "aust/tests/unit",
        "aust/tests/integration",
    ]

    project_root = Path(__file__).parent.parent.parent.parent
    for dir_path in package_dirs:
        init_file = project_root / dir_path / "__init__.py"
        assert init_file.exists(), f"__init__.py missing in {dir_path}"


def test_requirements_txt_exists() -> None:
    """Test that requirements.txt exists and is not empty."""
    project_root = Path(__file__).parent.parent.parent.parent
    requirements_file = project_root / "requirements.txt"

    assert requirements_file.exists(), "requirements.txt does not exist"
    assert requirements_file.is_file(), "requirements.txt is not a file"

    content = requirements_file.read_text()
    assert len(content) > 0, "requirements.txt is empty"
    assert "torch" in content, "PyTorch not pinned in requirements.txt"
    assert "pytest" in content, "pytest not pinned in requirements.txt"


def test_gitignore_exists() -> None:
    """Test that .gitignore exists."""
    project_root = Path(__file__).parent.parent.parent.parent
    gitignore = project_root / ".gitignore"

    assert gitignore.exists(), ".gitignore does not exist"
    assert gitignore.is_file(), ".gitignore is not a file"


def test_gitmodules_exists() -> None:
    """Test that .gitmodules exists."""
    project_root = Path(__file__).parent.parent.parent.parent
    gitmodules = project_root / ".gitmodules"

    assert gitmodules.exists(), ".gitmodules does not exist"
    assert gitmodules.is_file(), ".gitmodules is not a file"


def test_readme_exists() -> None:
    """Test that README.md exists and contains expected sections."""
    project_root = Path(__file__).parent.parent.parent.parent
    readme = project_root / "README.md"

    assert readme.exists(), "README.md does not exist"
    assert readme.is_file(), "README.md is not a file"

    content = readme.read_text()
    expected_sections = [
        "# CAUST",
        "## Overview",
        "## Project Structure",
        "## Setup Instructions",
        "## Development Workflow",
    ]

    for section in expected_sections:
        assert section in content, f"README.md missing section: {section}"


def test_setup_py_exists() -> None:
    """Test that setup.py exists."""
    project_root = Path(__file__).parent.parent.parent.parent
    setup_py = project_root / "setup.py"

    assert setup_py.exists(), "setup.py does not exist"
    assert setup_py.is_file(), "setup.py is not a file"


def test_pyproject_toml_exists() -> None:
    """Test that pyproject.toml exists and contains tool configs."""
    project_root = Path(__file__).parent.parent.parent.parent
    pyproject = project_root / "pyproject.toml"

    assert pyproject.exists(), "pyproject.toml does not exist"
    assert pyproject.is_file(), "pyproject.toml is not a file"

    content = pyproject.read_text()
    assert "[tool.black]" in content, "black config missing in pyproject.toml"
    assert "[tool.ruff]" in content, "ruff config missing in pyproject.toml"
    assert "[tool.mypy]" in content, "mypy config missing in pyproject.toml"
    assert "[tool.pytest.ini_options]" in content, "pytest config missing in pyproject.toml"


def test_logging_config_module_exists() -> None:
    """Test that logging configuration module exists."""
    project_root = Path(__file__).parent.parent.parent.parent
    logging_config = project_root / "aust" / "src" / "logging_config.py"

    assert logging_config.exists(), "logging_config.py does not exist"
    assert logging_config.is_file(), "logging_config.py is not a file"
