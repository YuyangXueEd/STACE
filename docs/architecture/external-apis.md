# External APIs

AUST integrates with external services for LLM/VLM access and infrastructure management.

## OpenRouter API

- **Purpose:** Provides unified access to multiple LLM and VLM models (GPT-4o, Claude 3.5 Sonnet, GPT-4V) for agent intelligence
- **Documentation:** https://openrouter.ai/docs
- **Base URL(s):** https://openrouter.ai/api/v1
- **Authentication:** Bearer token authentication via API key (passed in `Authorization: Bearer $OPENROUTER_API_KEY` header)
- **Rate Limits:** Model-specific rate limits; varies by provider (e.g., GPT-4: 10k RPM, Claude: 5k RPM). Implement exponential backoff retry logic.

**Key Endpoints Used:**
- `POST /chat/completions` - Generates LLM responses for agents (Hypothesis Generator, Critic, Query Generator, Reporter, Judges)
- `POST /chat/completions` (with vision models) - VLM-based evaluation for concept-erasure leakage detection (Evaluator Agent)

**Integration Notes:**
- All agent LLM calls route through OpenRouter client wrapper for centralized rate limit handling, retry logic, and error management
- Model selection configurable per agent via `AgentPromptConfig` (allows different models for different agents)
- Timeout: 60s per request with 3 retries and exponential backoff (2s, 4s, 8s)
- Cost tracking: Log model usage (tokens, cost estimates) for budget monitoring
- Error handling: Graceful degradation on rate limits (wait + retry), fail tasks on persistent errors

## Kubernetes Job API

- **Purpose:** Submits and manages GPU jobs for experiment execution (DeepUnlearn and concept-erasure tools)
- **Documentation:** https://kubernetes.io/docs/reference/kubernetes-api/workload-resources/job-v1/
- **Base URL(s):** Cluster-specific Kubernetes API server (configured via kubeconfig)
- **Authentication:** Service account token (mounted in pod at `/var/run/secrets/kubernetes.io/serviceaccount/token`)
- **Rate Limits:** API server rate limits (typically 50 QPS per client)

**Key Endpoints Used:**
- `POST /apis/batch/v1/namespaces/{namespace}/jobs` - Creates GPU job for experiment execution
- `GET /apis/batch/v1/namespaces/{namespace}/jobs/{name}` - Polls job status
- `GET /api/v1/namespaces/{namespace}/pods` - Retrieves pod logs for experiment results
- `DELETE /apis/batch/v1/namespaces/{namespace}/jobs/{name}` - Cleans up completed jobs

**Integration Notes:**
- Experiment Executor submits GPU jobs with H200 resource requests (`nvidia.com/gpu: 1`)
- Job template includes persistent volume mounts for outputs/, experiment artifacts, and DeepUnlearn/concept-erasure codebases
- Timeout: 30 minutes per job (NFR6), job killed if exceeded
- Retry logic: Transient GPU unavailability (queue full) triggers 3 retries with 5-minute backoff
- Job cleanup: Completed jobs deleted after results retrieved to avoid quota limits
- Security: Jobs run with restricted security context (`runAsNonRoot`, `readOnlyRootFilesystem` where possible)
