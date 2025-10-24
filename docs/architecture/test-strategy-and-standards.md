# Test Strategy and Standards

## Testing Philosophy

- **Approach:** Test-after development (not strict TDD) due to aggressive 3-week timeline; prioritize critical path components
- **Coverage Goals:**
  - Overall: ≥ 70% line coverage
  - Critical components (orchestrators, agents, repositories): ≥ 85%
  - Non-critical (utils, config loaders): ≥ 50%
- **Test Pyramid:** 60% unit tests, 30% integration tests, 10% end-to-end tests

## Test Types and Organization

### Unit Tests

- **Framework:** pytest 7.4.3
- **File Convention:** `tests/unit/test_{module}.py` (e.g., `test_hypothesis_generator.py`)
- **Location:** `tests/unit/` directory
- **Mocking Library:** pytest-mock 3.12.0
- **Coverage Requirement:** ≥ 85% for agents, orchestrators, repositories

**AI Agent Requirements:**
- Generate tests for all public methods in agents, orchestrators, repositories
- Cover edge cases: empty inputs, max iterations reached, API failures
- Follow AAA pattern (Arrange, Act, Assert)
- Mock all external dependencies: OpenRouter API, Kubernetes API, file I/O, CAMEL-AI memory

**Example:**
```python
def test_hypothesis_generator_generates_valid_hypothesis(mocker):
    # Arrange
    mock_openrouter = mocker.patch("agents.hypothesis_generator.OpenRouterClient")
    mock_openrouter.return_value.chat_completion.return_value = {"hypothesis": "..."}
    generator = HypothesisGenerator(config=mock_config)
    
    # Act
    hypothesis = generator.generate_hypothesis(context=mock_context)
    
    # Assert
    assert hypothesis.hypothesis_id is not None
    assert hypothesis.attack_type in ["membership_inference", "model_inversion", ...]
    assert 0.0 <= hypothesis.confidence_score <= 1.0
```

### Integration Tests

- **Scope:** Test component interactions (orchestrator + agents + repositories), no external APIs
- **Location:** `tests/integration/`
- **Test Infrastructure:**
  - **File System**: Temp directories for outputs, loop state
  - **FAISS**: In-memory FAISS index with mock papers
  - **Mocked APIs**: OpenRouter and Kubernetes APIs mocked via pytest-mock

**Example Integration Test:**
```python
def test_inner_loop_completes_one_iteration_end_to_end(mock_openrouter, mock_k8s, tmp_path):
    # Arrange: Set up orchestrator with all dependencies
    orchestrator = InnerLoopOrchestrator(
        output_dir=tmp_path,
        config=test_config
    )
    
    # Act: Run one complete iteration
    orchestrator.start_task("test-task-001", "data_based_unlearning", {})
    result = orchestrator.execute_iteration()
    
    # Assert: Verify iteration result and state persistence
    assert result.iteration_number == 1
    assert result.hypothesis is not None
    assert result.experiment_results.execution_status == "success"
    assert (tmp_path / "test-task-001" / "loop_state.json").exists()
```

### End-to-End Tests

- **Framework:** pytest with Testcontainers Python 3.7.1
- **Scope:** Full workflow with real FAISS, real file system, mocked OpenRouter/K8s
- **Environment:** Dockerized test environment or staging Kubernetes cluster
- **Test Data:** Pre-defined seed templates, mock experiment results, sample papers

**Example E2E Test:**
```python
def test_full_inner_loop_discovers_vulnerability(staging_env):
    # Arrange: Real RAG system, real file persistence, mocked GPU execution
    task_id = f"e2e-test-{uuid.uuid4()}"
    
    # Act: Run complete inner loop (10 iterations max)
    result = run_task(task_id, "data_based_unlearning", max_iterations=10)
    
    # Assert: Attack trace generated, vulnerability found (or max iterations reached)
    assert result["status"] in ["vulnerability_found", "max_iterations_reached"]
    assert os.path.exists(f"aust/outputs/{task_id}/attack_trace.md")
    assert os.path.exists(f"aust/outputs/{task_id}/report.md")
```

## Test Data Management

- **Strategy:** Fixtures for common test data (mock configs, sample papers, mock LLM responses)
- **Fixtures:** `tests/fixtures/` directory
  - `mock_configs/`: Sample agent prompt configs, threshold configs
  - `mock_papers/`: 3-5 sample research papers (PDFs + extracted text)
  - `mock_experiment_results/`: Pre-generated experiment results for different scenarios
- **Factories:** Use factory functions for generating test data models (Hypothesis, IterationResult, etc.)
- **Cleanup:** pytest fixtures with `yield` for automatic cleanup of temp files, directories

## Continuous Testing

- **CI Integration:** Run `pytest tests/unit` and `pytest tests/integration` on every git push (GitHub Actions or GitLab CI)
- **Performance Tests:** Not implemented in MVP; future enhancement for load testing OpenRouter API usage
- **Security Tests:** `bandit` (Python security linter) run in CI; dependency scanning with `safety`
