# Source Tree

The following directory structure reflects the monorepo architecture, component organization, and file-based storage design:

```plaintext
CAUST/                                    # Monorepo root (https://github.com/vios-s/CAUST)
├── README.md                             # Project documentation
├── requirements.txt                      # Pinned Python dependencies (NFR11)
├── setup.py                              # Package setup for local development
├── .gitignore
├── .gitmodules                           # Git submodules config
│
├── docker/                               # Docker and Kubernetes configs
│   ├── Dockerfile                        # Main application container
│   ├── job.yaml                          # Kubernetes GPU job template
│   ├── pvc.yaml                          # Persistent volume claims
│   └── deployment.yaml                   # Kubernetes deployment (if needed)
│
├── configs/                              # Configuration files (YAML)
│   ├── prompts/                          # Agent prompt configurations
│   │   ├── hypothesis_generator_data_based.yaml
│   │   ├── hypothesis_generator_concept_erasure.yaml
│   │   ├── critic.yaml
│   │   ├── query_generator.yaml
│   │   ├── evaluator_data_based.yaml
│   │   ├── evaluator_concept_erasure.yaml
│   │   ├── reporter.yaml
│   │   └── judges.yaml
│   ├── thresholds/                       # Evaluation thresholds
│   │   ├── data_based.yaml
│   │   └── concept_erasure.yaml
│   ├── tasks/                            # Task-specific configs (seed templates)
│   │   ├── data_based_unlearning.yaml
│   │   └── concept_erasure.yaml
│   └── personas/                         # Judge personas
│       └── judges.yaml
│
├── loop/                                 # Loop orchestration
│   ├── __init__.py
│   ├── models.py                         # Pydantic models (LoopState, IterationResult, etc.)
│   ├── inner_loop_orchestrator.py        # Inner Loop Orchestrator
│   ├── outer_loop_orchestrator.py        # Outer Loop Orchestrator
│   ├── state_machine.py                  # State machine logic
│   └── repositories/                     # Repository pattern implementations
│       ├── __init__.py
│       ├── loop_state_repository.py      # Loop State Repository
│       ├── attack_trace_repository.py    # Attack Trace Repository
│       └── report_repository.py          # Report Repository
│
├── agents/                               # Agent implementations
│   ├── __init__.py
│   ├── base_agent.py                     # Base agent class (wraps CAMEL-AI)
│   ├── hypothesis_generator.py           # Hypothesis Generator Agent
│   ├── critic.py                         # Critic Agent
│   ├── query_generator.py                # Query Generator Agent
│   ├── evaluator.py                      # Evaluator Agent
│   ├── reporter.py                       # Reporter Agent
│   └── judge.py                          # Judge Agents
│
├── tools/                                # Experiment execution tools
│   ├── __init__.py
│   ├── experiment_executor.py            # Experiment Executor
│   ├── deepunlearn_tool.py               # DeepUnlearn FunctionTool adapter
│   ├── concept_erasure_tool.py           # Concept-Erasure FunctionTool adapter
│   └── kubernetes_client.py              # Kubernetes Job API client wrapper
│
├── rag/                                  # RAG system
│   ├── __init__.py
│   ├── rag_system.py                     # RAG System implementation
│   ├── indexer.py                        # Paper indexing logic
│   ├── retriever.py                      # Semantic search logic
│   ├── papers/                           # Research paper corpus (PDFs)
│   │   ├── data_unlearning/
│   │   ├── concept_erasure/
│   │   └── attack_methods/
│   ├── faiss_index.bin                   # FAISS vector index (generated)
│   └── paper_metadata.json               # Paper citations and metadata (generated)
│
├── memory/                               # Memory system
│   ├── __init__.py
│   ├── memory_system.py                  # Memory System (CAMEL-AI wrapper)
│   └── [CAMEL-AI managed storage]        # Memory persistence (implementation-specific)
│
├── outputs/                              # Persistent outputs (mounted PVC)
│   └── [task_id]/                        # Per-task outputs (created at runtime)
│       ├── loop_state.json
│       ├── attack_trace.json
│       ├── attack_trace.md
│       ├── report.md
│       ├── report.json
│       ├── iterations/
│       ├── experiments/
│       └── judges/
│
├── external/                             # External dependencies in dev mode
│   └── camel/                            # CAMEL-AI (pip install -e external/camel)
│       └── [CAMEL-AI source code]
│
├── submodules/                           # Git submodules
│   └── DeepUnlearn/                      # DeepUnlearn repository (git submodule)
│       └── [DeepUnlearn source code]
│
├── scripts/                              # Utility scripts
│   ├── setup_environment.sh              # Initial setup script
│   ├── index_papers.py                   # RAG indexing script
│   ├── run_task.py                       # Main entry point for running tasks
│   └── test_integration.py               # Integration test script
│
├── tests/                                # Test suite
│   ├── __init__.py
│   ├── unit/                             # Unit tests
│   │   ├── test_agents.py
│   │   ├── test_loop_orchestrator.py
│   │   ├── test_rag_system.py
│   │   └── ...
│   ├── integration/                      # Integration tests
│   │   ├── test_inner_loop.py
│   │   ├── test_outer_loop.py
│   │   └── test_full_workflow.py
│   └── fixtures/                         # Test fixtures and mocks
│       ├── mock_configs/
│       ├── mock_papers/
│       └── mock_experiment_results/
│
├── docs/                                 # Project documentation
│   ├── brainstorming-session-results.md
│   ├── brief.md
│   ├── prd.md
│   ├── architecture.md                   # This document
│   └── deployment-guide.md               # Deployment instructions (TBD)
│
└── .bmad-core/                           # BMAD framework metadata
    └── [BMAD config files]
```

**Key Directory Responsibilities:**
- **loop/**: Orchestration logic and state management
- **agents/**: LLM-powered agent implementations
- **tools/**: Experiment execution adapters and GPU job management
- **rag/**: Semantic search system and paper corpus
- **memory/**: Long-term memory for successful discoveries
- **outputs/**: Persistent task results (attack traces, reports, judgments)
- **configs/**: All configuration files (prompts, thresholds, tasks, personas)
- **external/**: CAMEL-AI in dev/editable mode
- **submodules/**: DeepUnlearn git submodule
- **tests/**: Unit and integration tests
- **docker/**: Containerization and Kubernetes manifests
