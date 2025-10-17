# Tech Stack

The technology stack selections below are the **single source of truth** for AUST development. All decisions are based on PRD requirements, performance targets (NFR6-NFR8), and the 3-week implementation timeline. Technologies are pinned to specific versions for reproducibility (NFR11).

## Cloud Infrastructure

- **Provider:** On-premises Kubernetes cluster (or cloud Kubernetes service if available - to be confirmed with user)
- **Key Services:**
  - Kubernetes for container orchestration and GPU job scheduling
  - Persistent Volumes for outputs, loop state, and memory storage
  - Job API for H200 GPU allocation and experiment execution
- **Deployment Regions:** Single cluster (on-premises or single cloud region)

## Technology Stack Table

| Category | Technology | Version | Purpose | Rationale |
|----------|-----------|---------|---------|-----------|
| **Language** | Python | 3.11.5 | Primary development language | Required by CAMEL-AI and DeepUnlearn; excellent ML/AI library ecosystem; team expertise |
| **Runtime** | Python | 3.11.5 | Application runtime | LTS Python version with modern features (match groups, better error messages) |
| **Agent Framework** | CAMEL-AI | dev/editable (latest main) | Multi-agent orchestration | Provides agent abstractions, memory system (FR10), and role-based agents; dev mode enables customization per PRD requirement |
| **LLM/VLM API** | OpenRouter | N/A (API service) | LLM/VLM access for agents | Supports multiple models (GPT-4o, Claude 3.5, GPT-4V) for flexibility; single API reduces integration complexity; per PRD NFR4 |
| **Containerization** | Docker | 24.0.7 | Application containerization | Provides reproducibility (NFR11), container isolation (NFR12), and supports Kubernetes deployment |
| **Orchestration** | Kubernetes | 1.28.x | Container orchestration and GPU management | Enables H200 GPU job scheduling (NFR1), persistent volumes (NFR14), and resource limits; production-grade resilience |
| **ML Library** | PyTorch | 2.8.0 | Deep learning framework | Required by DeepUnlearn and concept-erasure methods; CUDA support for H200 GPUs |
| **Embedding Model** | Sentence-Transformers | 2.2.2 | Text embeddings for RAG | Pre-trained models (all-MiniLM-L6-v2); fast local inference; avoids OpenRouter API calls for embeddings |
| **PDF Parsing** | PyMuPDF (fitz) | 1.23.8 | Extract text from research papers | Fast, reliable PDF text extraction; required for RAG paper corpus processing (FR5) |
| **Logging** | Python logging | 3.11 (stdlib) | Application logging | Structured logging with JSON formatting; no external dependencies; supports debugging and monitoring |
| **Configuration** | PyYAML | 6.0.1 | YAML config file parsing | Human-readable config files (prompts, thresholds, personas); easy versioning and iteration |
| **Testing** | pytest | 7.4.3 | Unit and integration testing | Standard Python testing framework; excellent plugin ecosystem; supports fixtures for complex test setups |
| **Testing Mocks** | pytest-mock | 3.12.0 | Mocking for external dependencies | Simplifies mocking OpenRouter API, GPU jobs, and external tools during testing |
| **Container Testing** | Testcontainers Python | 3.7.1 | Integration testing with containers | Enables testing with real FAISS, file system isolation; validates deployment configuration |
| **Git Submodule** | DeepUnlearn | commit SHA pinned | Data-based unlearning experiments | Provides unlearning method implementations per FR6; git submodule enables version control and local modifications |
| **Git Submodule** | Concept-Erasure Tools | TBD (specific repo to be selected) | Concept-erasure experiments | Provides concept-erasure implementations per FR7; specific tool to be selected in Epic 3 |
| **Dependency Management** | pip | 23.3.1 | Package installation | Standard Python package manager; requirements.txt with pinned versions (NFR11) |
| **IaC (if needed)** | Kubernetes YAML | N/A | Infrastructure as code | Native Kubernetes manifests (job.yaml, pvc.yaml); simple, version-controlled; no additional IaC tool needed for MVP |

## Elicitation: Tech Stack Review

**IMPORTANT**: This technology stack is the definitive selection for AUST. Please review the table above and confirm:

1. **Are there any gaps or missing technologies** that you expected to see based on the PRD?
2. **Do you disagree with any selections?** If so, which ones and why?
3. **Are there specific version requirements** for any technologies that differ from what's listed?
4. **Database selection**: The architecture currently uses FAISS (in-memory vector DB) and file-based storage (JSON for loop state, persistent volumes for outputs). Would you prefer a traditional database (PostgreSQL, MongoDB, etc.) for any components?
5. **Concept-Erasure Tool**: The specific concept-erasure repository is marked "TBD" - do you have a preferred tool (e.g., EraseDiff, specific concept ablation repo), or should we select during Epic 3 implementation?
6. **Cloud vs On-Premises**: Is AUST deploying to an on-premises Kubernetes cluster with H200s, or to a cloud provider (GCP, AWS, Azure) with GPU instances?

**Please provide feedback on the tech stack before we proceed to Data Models.** If everything looks correct, respond with "approved" or "proceed" to continue.

---
