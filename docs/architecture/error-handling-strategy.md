# Error Handling Strategy

## General Approach

- **Error Model:** Exception-based error handling with custom exception hierarchy; structured logging for all errors
- **Exception Hierarchy:**
  - `AUSTException` (base)
    - `LoopExecutionException` (inner/outer loop failures)
      - `HypothesisGenerationException`
      - `ExperimentExecutionException`
      - `EvaluationException`
    - `ExternalServiceException` (API failures)
      - `OpenRouterAPIException`
      - `KubernetesAPIException`
    - `ConfigurationException` (config validation failures)
    - `DataValidationException` (Pydantic validation errors)
- **Error Propagation:** Errors bubble up through orchestrator layers; orchestrators decide whether to retry, skip, or fail the task

## Logging Standards

- **Library:** Python `logging` module (stdlib)
- **Format:** JSON structured logging for machine parsing
  ```json
  {
    "timestamp": "2025-10-16T12:34:56.789Z",
    "level": "ERROR",
    "logger": "loop.inner_loop_orchestrator",
    "message": "Experiment execution failed",
    "correlation_id": "task-001-iter-3",
    "service": "AUST",
    "error_type": "ExperimentExecutionException",
    "stack_trace": "..."
  }
  ```
- **Levels:**
  - DEBUG: Agent interactions, detailed state transitions
  - INFO: Task start/completion, iteration results, successful operations
  - WARNING: Retryable failures (OpenRouter rate limits, GPU queue delays)
  - ERROR: Non-recoverable errors (invalid config, persistent API failures)
  - CRITICAL: System-level failures (unable to persist state, memory corruption)
- **Required Context:**
  - Correlation ID: `{task_id}-iter-{iteration_number}` for traceability
  - Service Context: Always "AUST" + component name (e.g., "AUST.hypothesis_generator")
  - User Context: Task metadata (task_type, iteration, current_state) - no PII/sensitive data

## Error Handling Patterns

### External API Errors

- **Retry Policy:**
  - OpenRouter API: 3 retries with exponential backoff (2s, 4s, 8s)
  - Kubernetes API: 3 retries with 5-second backoff
  - Retry on: 429 (rate limit), 500, 502, 503, 504 (server errors)
  - No retry on: 400 (bad request), 401/403 (auth errors), 404 (not found)
- **Circuit Breaker:** Not implemented in MVP; future enhancement for sustained OpenRouter outages
- **Timeout Configuration:**
  - OpenRouter API: 60s per request
  - Kubernetes job execution: 30 minutes per job (NFR6)
  - RAG retrieval: 10s (well above NFR8's < 5s target)
- **Error Translation:**
  - OpenRouter errors mapped to `OpenRouterAPIException` with original status code and message
  - Kubernetes errors mapped to `KubernetesAPIException` with job ID and pod logs

### Business Logic Errors

- **Custom Exceptions:**
  - `HypothesisGenerationException`: Raised when hypothesis generator fails to produce valid hypothesis
  - `ExperimentExecutionException`: Raised when GPU job fails or times out
  - `EvaluationException`: Raised when evaluator cannot assess results (e.g., VLM failure)
- **User-Facing Errors:** Errors logged to aust/outputs/{task_id}/error.log with user-friendly messages (avoid stack traces in user outputs)
- **Error Codes:** Simple string codes: "OPENROUTER_RATE_LIMIT", "GPU_TIMEOUT", "INVALID_CONFIG", "HYPOTHESIS_GENERATION_FAILED"

### Data Consistency

- **Transaction Strategy:** File-based atomic writes (write to temp file, then rename) for loop state persistence
- **Compensation Logic:** If loop state write fails mid-iteration, load previous valid state and retry iteration from last known checkpoint
- **Idempotency:** Experiment execution designed to be idempotent (same hypothesis + params = same result); retry-safe via job ID tracking
