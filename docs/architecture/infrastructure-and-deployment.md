# Infrastructure and Deployment

## Infrastructure as Code

- **Tool:** Kubernetes YAML manifests (no additional IaC tool for MVP)
- **Location:** `docker/` directory
- **Approach:** Native Kubernetes manifests (job.yaml, pvc.yaml, deployment.yaml) version-controlled in git. Direct `kubectl apply` for deployment. Future consideration: Helm charts for parameterized deployments.

## Deployment Strategy

- **Strategy:** Blue-green deployment not applicable for MVP (single instance); rolling updates for future multi-instance deployments
- **CI/CD Platform:** TBD (GitHub Actions, GitLab CI, or manual deployment for MVP)
- **Pipeline Configuration:** `/.github/workflows/` (if GitHub Actions) or manual kubectl commands
- **Deployment Process:**
  1. Build Docker image from Dockerfile
  2. Push to container registry (Docker Hub, GCR, or local registry)
  3. Apply Kubernetes manifests: `kubectl apply -f docker/pvc.yaml && kubectl apply -f docker/job.yaml`
  4. Monitor job status via `kubectl get jobs` and `kubectl logs`

## Environments

- **Development:** Local development environment with Docker Compose (optional); minikube for Kubernetes testing
- **Staging:** Kubernetes cluster with limited GPU resources (1x H200 or lower-tier GPU); for integration testing and validation
- **Production:** Kubernetes cluster with dedicated H200 GPUs; persistent volumes for outputs; production OpenRouter API keys

## Environment Promotion Flow

```
Development (local) → Staging (K8s cluster, test GPUs) → Production (K8s cluster, H200 GPUs)
```

- Code tested locally → merge to `dev` branch → deploy to staging → validate with integration tests → merge to `main` → deploy to production
- Manual approval gate before production deployment
- Persistent volumes maintained separately per environment

## Rollback Strategy

- **Primary Method:** Revert to previous Docker image tag via `kubectl set image`
- **Trigger Conditions:** Task failure rate > 50%, persistent OpenRouter API failures, GPU job failures > 3 consecutive attempts
- **Recovery Time Objective:** < 10 minutes (time to kubectl apply previous manifest and restart)
- **Data Safety:** Loop state and outputs persisted on PVC; rollback does not affect completed tasks
