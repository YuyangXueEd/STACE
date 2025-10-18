# Coding Standards

## Core Standards

- **Languages & Runtimes:** Python 3.11.5+
- **Style & Linting:**
  - Formatter: `black` (line length 100)
  - Linter: `ruff` (replaces flake8, pylint, isort)
  - Type checking: `mypy` in strict mode
  - Config: `pyproject.toml` in repo root
- **Test Organization:**
  - Unit tests: `tests/unit/test_{module_name}.py`
  - Integration tests: `tests/integration/test_{workflow_name}.py`
  - Test naming: `test_{function_name}_{scenario}_{expected_result}`

## Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Classes | PascalCase | `HypothesisGenerator`, `LoopState` |
| Functions/Methods | snake_case | `generate_hypothesis`, `execute_experiment` |
| Variables | snake_case | `task_id`, `retrieved_papers` |
| Constants | UPPER_SNAKE_CASE | `MAX_ITERATIONS`, `DEFAULT_TIMEOUT` |
| Private methods | _leading_underscore | `_validate_config`, `_retry_with_backoff` |
| Pydantic models | PascalCase (fields snake_case) | `class LoopState: task_id: str` |

## Critical Rules

- **Never use print() in production code**: All output must use the `logging` module. Use `logger.info()`, `logger.debug()`, etc.
- **All Pydantic models must have validation**: Use Field() with constraints, custom validators for complex logic
- **Repository methods must handle file I/O errors**: Wrap file operations in try/except, raise custom exceptions with context
- **Agent LLM calls must go through OpenRouter client wrapper**: Never call OpenRouter API directly; use centralized client for rate limit handling
- **Kubernetes GPU jobs must have timeout**: Always set `activeDeadlineSeconds` in job spec (30 minutes per NFR6)
- **Loop state must be persisted after every state transition**: Call `loop_state_repository.save_state()` after each orchestrator state change
- **Never hardcode API keys or secrets**: Load from environment variables or Kubernetes secrets
- **All datetime objects must be timezone-aware**: Use `datetime.now(timezone.utc)`, never naive datetimes
- **CAMEL framework first for agent development**: When developing agents, consult `external/camel` for docs and examples. Use prebuilt CAMEL components (BaseToolkit, ChatAgent, FunctionTool, etc.) before building custom implementations
- **Always read external tool READMEs first**: When integrating external tools (ESD, DeepUnlearn, etc.), ALWAYS read their official README to understand use cases, hyperparameters, and correct usage. Never make up parameters or usage patterns

## Language-Specific Guidelines

### Python Specifics

- **Type hints**: Mandatory for all function signatures; use `from typing import` for complex types
- **Async/await**: Use for concurrent judge execution; avoid mixing sync/async without clear separation
- **Context managers**: Use `with` statements for file I/O, resource management
- **F-strings**: Preferred for string formatting (not %-formatting or .format())
- **Pathlib**: Use `pathlib.Path` for file paths, not string concatenation
- **Dataclasses vs Pydantic**: Use Pydantic for data models with validation; dataclasses for simple internal structures
