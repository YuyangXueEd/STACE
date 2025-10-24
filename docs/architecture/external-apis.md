# External APIs

AUST integrates with external services for LLM/VLM access and infrastructure management.

## OpenRouter API

- **Purpose:** Provides unified access to multiple LLM and VLM models (GPT-5, Claude 3.5 Sonnet, GPT-4V) for agent intelligence
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
- Job template includes persistent volume mounts for aust/outputs/, experiment artifacts, and DeepUnlearn/concept-erasure codebases
- Timeout: 30 minutes per job (NFR6), job killed if exceeded
- Retry logic: Transient GPU unavailability (queue full) triggers 3 retries with 5-minute backoff
- Job cleanup: Completed jobs deleted after results retrieved to avoid quota limits
- Security: Jobs run with restricted security context (`runAsNonRoot`, `readOnlyRootFilesystem` where possible)

## External Reference Libraries

These repositories are cloned to `external/` for **agent reference only** (not runtime dependencies). They provide source code context for AI agents during code synthesis and refinement.

### Hugging Face Diffusers

- **Purpose:** Reference implementation for diffusion model pipelines (Stable Diffusion, SDXL, Flux) used in concept-erasure experiments
- **Repository:** https://github.com/huggingface/diffusers
- **Location:** `external/diffusers/`
- **Commit:** 9c3b58dcf16ebd027fd3d85ec703ad5a142b1e1e (main, cloned 2025-10-23)
- **Documentation:** https://huggingface.co/docs/diffusers

**Usage by AI Agents:**
- **Coding Agent (Story 1.6a):** References `src/diffusers/pipelines/` for SD/SDXL/Flux pipeline integration patterns
- **Refinement Agent:** Uses diffusers source as ground truth for API usage, model loading, and inference patterns
- **Key Directories:**
  - `src/diffusers/models/` - UNet, VAE, transformer architectures
  - `src/diffusers/pipelines/` - Full pipeline implementations (StableDiffusionPipeline, etc.)
  - `src/diffusers/schedulers/` - Noise schedulers (DDPM, DDIM, etc.)
  - `examples/` - Training and fine-tuning examples

**Note:** Runtime dependency `diffusers` package should also be added to `requirements.txt` for actual execution. The cloned repo is for **agent learning and reference**, not direct code import.

### DeepUnlearn

- **Purpose:** Data-based unlearning methods implementation (Amnesiac Unlearning, SCRUB, etc.) for machine learning models
- **Repository:** (Internal/research repository)
- **Location:** `external/DeepUnlearn/`
- **Commit:** 6830ff6e5fd72f4793890ee31904814892625acf (cloned 2025-04-20)
- **Local Documentation:** `external/DeepUnlearn/README.md`

**Usage by AI Agents:**
- **Experiment Executor (Story 1.6):** Reference for data-based unlearning pipeline implementation
- **Key Directories:**
  - `external/DeepUnlearn/munl/` - Core unlearning algorithms and methods
  - `external/DeepUnlearn/pipeline/` - Experiment execution pipelines
  - `external/DeepUnlearn/datasets/` - Dataset loading and preprocessing utilities
  - `external/DeepUnlearn/commands/` - CLI commands for running experiments

### ESD (Erasing Stable Diffusion)

- **Purpose:** Concept erasure implementation for Stable Diffusion models (SD, SDXL, Flux)
- **Repository:** https://github.com/rohitgandikota/erasing (or similar ESD repo)
- **Location:** `external/esd/`
- **Commit:** 4cbdac2d9c3b16aaa00da4ac13b1efcf5d6be4d4 (cloned 2025-08-17)
- **Local Documentation:** `external/esd/README.md`

**Usage by AI Agents:**
- **Concept Unlearning (Story 1.6/1.6a):** Reference for concept-erasure attack implementation
- **Key Files:**
  - `external/esd/esd_sd.py` - ESD implementation for Stable Diffusion 1.x/2.x
  - `external/esd/esd_sdxl.py` - ESD implementation for SDXL
  - `external/esd/esd_flux.py` - ESD implementation for Flux models
  - `external/esd/evalscripts/` - Evaluation scripts for concept erasure
  - `external/esd/notebooks/` - Example notebooks showing ESD usage

### CAMEL-AI

- **Purpose:** Multi-agent framework source code for deep customization, troubleshooting, and understanding CAMEL internals
- **Repository:** https://github.com/camel-ai/camel
- **Location:** `external/camel/`
- **Commit:** a9a407f746c18de0c322af1eb8e348be68b80631 (cloned 2025-10-17)
- **Local Documentation:** `external/camel/README.md`, `external/camel/docs/`

**Usage by AI Agents:**
- **Architecture Design:** Reference CAMEL's agent abstractions, role-based agent patterns, and multi-agent orchestration
- **Memory System Integration:** Understanding long-term memory, vector retrieval, and context management (Story 2.5)
- **Troubleshooting:** Deep dive into CAMEL internals when debugging agent behaviors or extending framework capabilities
- **Key Directories:**
  - `external/camel/camel/agents/` - Agent base classes and role-based implementations
  - `external/camel/camel/memories/` - Memory system implementations (vector, context, long-term)
  - `external/camel/camel/messages/` - Message passing and communication protocols
  - `external/camel/camel/societies/` - Multi-agent coordination patterns
  - `external/camel/docs/cookbooks/` - Practical examples and tutorials
  - `external/camel/docs/key_modules/` - Core module documentation

**Note:** CAMEL-AI is installed in **dev/editable mode** (pip install -e) per tech stack, allowing direct modification. The external/ clone serves as reference and version control baseline.
