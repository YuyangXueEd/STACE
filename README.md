# CAUST - Continual Automated Unlearning Safety Testing

**CAUST** is an autonomous multi-agent system for red-teaming machine unlearning methods. It automatically generates hypotheses, designs experiments, and evaluates unlearning techniques to discover vulnerabilities in data-based and concept-erasure approaches.

## Overview

CAUST uses a dual-loop architecture:
- **Inner Loop**: Generates hypotheses, designs experiments, executes them on H200 GPUs, and evaluates results
- **Outer Loop**: Synthesizes findings across iterations, judges novelty and impact, and produces comprehensive reports

The system leverages:
- Multi-agent orchestration via CAMEL-AI
- RAG (Retrieval Augmented Generation) for research paper knowledge
- GPU-accelerated experiment execution on Kubernetes
- Persistent memory for successful attack discoveries

## Project Structure

```
CAUST/
├── aust/                       # Main application code
│   ├── agents/                 # LLM-powered agent implementations
│   ├── tools/                  # Experiment execution adapters
│   ├── rag/                    # RAG system for research papers
│   ├── memory/                 # Long-term memory system
│   ├── loop/                   # Orchestration and state management
│   ├── outputs/                # Persistent task results
│   ├── configs/                # Configuration files (prompts, thresholds)
│   ├── experiments/            # Experiment results
│   ├── submodules/             # External dependencies (DeepUnlearn)
│   └── external/               # Dev mode dependencies (CAMEL-AI)
├── docker/                     # Docker and Kubernetes configs
│   ├── Dockerfile              # Container image definition
│   ├── job.yaml                # Kubernetes GPU job template
│   └── pvc.yaml                # Persistent volume claims
├── scripts/                    # Utility scripts
├── tests/                      # Test suite (unit + integration)
├── logs/                       # Application logs
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
   git clone https://github.com/vios-s/CAUST.git
   cd CAUST
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
docker build -t caust:latest -f docker/Dockerfile .
```

Test the container:
```bash
docker run --rm caust:latest
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
   kubectl logs job/caust-experiment-job
   ```

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
- Ensure `PYTHONPATH` includes project root: `export PYTHONPATH=/path/to/CAUST:$PYTHONPATH`
- Check virtual environment is activated

### Logging not working
- Verify `logs/` directory exists and is writable
- Check log level in configuration (default: INFO)

## Contributing

Please follow the coding standards and ensure all tests pass before submitting changes.

## License

TBD

## Contact

Project repository: https://github.com/vios-s/CAUST
