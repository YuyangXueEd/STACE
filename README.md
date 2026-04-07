# STACE - Continual Automated Unlearning Safety Testing

**STACE** is an autonomous multi-agent system for red-teaming machine unlearning methods. It automatically generates hypotheses, designs experiments, and evaluates unlearning techniques to discover vulnerabilities in data-based and concept-erasure approaches.

## Overview

STACE uses a dual-loop architecture:
- **Inner Loop**: Generates hypotheses, designs experiments, executes them on H200 GPUs, and evaluates results
- **Outer Loop**: Synthesizes findings across iterations, judges novelty and impact, and produces comprehensive reports

The system leverages:
- Multi-agent orchestration via CAMEL-AI
- RAG (Retrieval Augmented Generation) for research paper knowledge
- GPU-accelerated experiment execution on Kubernetes
- Persistent memory for successful attack discoveries

## Project Structure

```
STACE/
├── aust/                       # Application package
│   ├── configs/                # Prompts, personas, thresholds, task templates
│   ├── experiments/            # Placeholder for experiment artifacts
│   ├── logs/                   # Runtime logs
│   ├── outputs/                # Persistent inner loop results
│   ├── rag_paper_db/           # Vector store for paper RAG
│   ├── scripts/                # CLI entry points for inner loop tooling
│   ├── src/                    # Source code
│   │   ├── agents/             # LLM-powered agent implementations
│   │   ├── loop/               # Orchestration and state management
│   │   ├── memory/             # Long-term memory system
│   │   ├── rag/                # Research paper retrieval subsystem
│   │   ├── toolkits/           # Integrations with external unlearning toolchains
│   │   └── logging_config.py   # Project-wide logging setup
│   ├── tests/                  # Test suite (unit + integration)
│   └── utils/                  # Helper scripts (e.g., paper downloads)
├── docker/                     # Docker and Kubernetes configs
│   ├── Dockerfile              # Container image definition
│   ├── job.yaml                # Kubernetes GPU job template
│   └── pvc.yaml                # Persistent volume claims
├── external/                   # Third-party submodules (DeepUnlearn, CAMEL)
├── logs/                       # Legacy log location (top-level)
├── requirements.txt            # Python dependencies
└── docs/                       # Project documentation
```

## Setup Instructions

### Prerequisites

- Python 3.11.5+
- Docker 24.0.7+
- Kubernetes 1.28.x with NVIDIA GPU support (H200)
- CUDA 12.1+ (for H200 GPUs)

### Local Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vios-s/STACE.git
   cd STACE
   ```

2. **Create Python virtual environment:**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip==23.3.1
   pip install -r requirements.txt
   ```

4. **Install CAMEL-AI in dev mode (Story 1.2):**
   ```bash
   # Will be added in Story 1.2
   ```

5. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys (OpenRouter, etc.)
   ```

6. **Run tests:**
   ```bash
   pytest tests/
   ```

### Docker Build

Build the Docker image:
```bash
docker build -t STACE:latest -f docker/Dockerfile .
```

Test the container:
```bash
docker run --rm STACE:latest
```

### Kubernetes Deployment

1. **Create persistent volumes:**
   ```bash
   kubectl apply -f docker/pvc.yaml
   ```

2. **Submit GPU job:**
   ```bash
   # Edit docker/job.yaml to set TASK_ID and TASK_TYPE
   kubectl apply -f docker/job.yaml
   ```

3. **Monitor job:**
   ```bash
   kubectl get jobs
   kubectl logs job/STACE-experiment-job
   ```

## Documentation

All project documentation is organized in the `docs/` directory:

### 📚 Main Documentation Files
- **[docs/architecture.md](docs/architecture.md)** - System architecture overview
- **[docs/prd.md](docs/prd.md)** - Product requirements document
- **[docs/brief.md](docs/brief.md)** - Project brief

### 📖 Story Documentation
- **[docs/stories/](docs/stories/)** - Implementation stories and documentation
  - Story 1.5 - Hypothesis Refinement Workforce
  - Story 1.0-1.8 - Integration summaries
  - Test results and implementation details
  - Inner loop orchestrator documentation

### 🤖 CAMEL-AI Resources
Complete documentation for using CAMEL-AI patterns:
- **[docs/camel-resources/README_CAMEL_RESOURCES.md](docs/camel-resources/README_CAMEL_RESOURCES.md)** - Master index and overview
- **[docs/camel-resources/CAMEL_QUICK_REFERENCE.md](docs/camel-resources/CAMEL_QUICK_REFERENCE.md)** - Quick lookup while coding
- **[docs/camel-resources/CAMEL_PATTERNS_GUIDE.md](docs/camel-resources/CAMEL_PATTERNS_GUIDE.md)** - Complete technical reference
- **[docs/camel-resources/CAMEL_INDEX.md](docs/camel-resources/CAMEL_INDEX.md)** - File index and navigation

### ⚙️ Configuration Guides
- **[docs/config-guides/CONCEPT_ERASURE_CONFIG_SUMMARY.md](docs/config-guides/CONCEPT_ERASURE_CONFIG_SUMMARY.md)** - Configuration reference
- **[docs/MAIN_PY_SUMMARY.md](docs/MAIN_PY_SUMMARY.md)** - Code structure summary

### 🏗️ Architecture Documentation
- **[docs/architecture/](docs/architecture/)** - Comprehensive architecture details
  - Components, workflows, data models
  - Tech stack and external APIs
  - Test strategy and security considerations

### 📋 Project Artifacts
- **[docs/epics/](docs/epics/)** - Epic-level planning and requirements
- **[docs/prds/](docs/prds/)** - Product requirement details

## Development Workflow

### Code Quality Standards

- **Formatter**: `black` (line length 100)
- **Linter**: `ruff`
- **Type Checking**: `mypy` (strict mode)

Run code quality checks:
```bash
black aust/ tests/
ruff check aust/ tests/
mypy aust/
```

### Testing

- **Unit tests**: `tests/unit/test_{module}.py`
- **Integration tests**: `tests/integration/test_{workflow}.py`

Run tests with coverage:
```bash
pytest tests/ --cov=aust --cov-report=html
```

### Logging

All production code uses the logging framework (no `print()` statements):

```python
from aust.logging_config import get_logger, set_correlation_id

logger = get_logger(__name__)

# Set correlation ID for request tracing
set_correlation_id("task_123")

# Log messages
logger.info("Starting experiment", extra={"experiment_id": "exp_001"})
logger.error("Experiment failed", extra={"error": str(e)})
```

## Configuration

Configuration files are located in `aust/configs/`:
- `prompts/`: Agent prompt templates
- `thresholds/`: Evaluation threshold configurations
- `tasks/`: Task-specific configurations
- `personas/`: Judge persona definitions

## Documentation

- [Product Requirements Document (PRD)](docs/prd.md)
- [Architecture Document](docs/architecture.md)
- [Tech Stack](docs/architecture/tech-stack.md)
- [Coding Standards](docs/architecture/coding-standards.md)

## Troubleshooting

### Docker build fails with CUDA errors
- Ensure NVIDIA Docker runtime is installed: `nvidia-docker --version`
- Verify GPU access: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`

### Kubernetes job fails to schedule
- Check GPU node labels: `kubectl get nodes --show-labels | grep nvidia`
- Verify PVC is bound: `kubectl get pvc`

### Import errors in Python
- Ensure `PYTHONPATH` includes project root: `export PYTHONPATH=/path/to/STACE:$PYTHONPATH`
- Check virtual environment is activated

### Logging not working
- Verify `logs/` directory exists and is writable
- Check log level in configuration (default: INFO)

## Contributing

Please follow the coding standards and ensure all tests pass before submitting changes.

## License

TBD

## Contact

Project repository: https://github.com/vios-s/STACE
