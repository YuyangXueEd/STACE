# Security

## Input Validation

- **Validation Library:** Pydantic for data models; custom validators for complex logic
- **Validation Location:** All external inputs (user task configs, agent LLM responses) validated at API boundary (orchestrator entry points)
- **Required Rules:**
  - All task_id inputs validated against regex pattern (alphanumeric + hyphens only)
  - All task_type inputs must be in allowed enum ("data_based_unlearning", "concept_erasure")
  - All file paths validated against directory traversal attacks (no `../` in paths)
  - All LLM-generated hypotheses validated against output schema before execution

## Authentication & Authorization

- **Auth Method:** Kubernetes service account for pod-to-API-server authentication; no user-facing auth in MVP (single-user system)
- **Session Management:** Not applicable (no web frontend or sessions)
- **Required Patterns:**
  - Kubernetes jobs run with dedicated service account with minimal RBAC permissions (create/get/delete jobs, read pod logs)
  - OpenRouter API key loaded from Kubernetes secret (not hardcoded)
  - No privilege escalation: containers run as non-root user (UID 1000)

## Secrets Management

- **Development:** `.env` file for local development (git-ignored); never commit secrets to repo
- **Production:** Kubernetes secrets for OpenRouter API key; mounted as environment variables in pod
- **Code Requirements:**
  - NEVER hardcode secrets in code or config files
  - Access secrets via `os.getenv("OPENROUTER_API_KEY")` with validation (raise error if missing)
  - No secrets in logs or error messages (redact API keys in OpenRouter client logging)

## API Security

- **Rate Limiting:** OpenRouter rate limits enforced by API provider; client implements exponential backoff retry
- **CORS Policy:** Not applicable (no web frontend)
- **Security Headers:** Not applicable (no HTTP server)
- **HTTPS Enforcement:** OpenRouter API uses HTTPS by default; Kubernetes API uses TLS

## Data Protection

- **Encryption at Rest:** Kubernetes persistent volumes encrypted at rest (cloud provider default encryption)
- **Encryption in Transit:** OpenRouter API calls over HTTPS; Kubernetes API calls over TLS
- **PII Handling:** No PII collected in MVP; task metadata may include model names, datasets (not user PII)
- **Logging Restrictions:**
  - NEVER log OpenRouter API keys (redact in logs)
  - NEVER log full experiment results with potentially sensitive model weights (log metrics only)
  - NEVER log Kubernetes service account tokens

## Dependency Security

- **Scanning Tool:** `safety` (Python dependency vulnerability scanner)
- **Update Policy:** Review and update dependencies monthly; critical vulnerabilities patched within 7 days
- **Approval Process:** All new dependencies reviewed for license compatibility (MIT, Apache 2.0, BSD) and security audit (check GitHub issues, CVE database)

## Security Testing

- **SAST Tool:** `bandit` (Python static security analysis)
- **DAST Tool:** Not applicable (no web frontend or external-facing APIs)
- **Penetration Testing:** Not planned for MVP; future consideration for production deployment
