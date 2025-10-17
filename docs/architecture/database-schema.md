# Database Schema

AUST uses **file-based storage** with JSON serialization for MVP simplicity and persistence via Kubernetes persistent volumes. No traditional database (SQL or NoSQL) is used in the MVP; all data is stored as JSON files with schema validation via Pydantic models.

## File-Based Storage Structure

```
outputs/
└── {task_id}/
    ├── loop_state.json                    # LoopState model
    ├── attack_trace.json                  # AttackTrace model (JSON)
    ├── attack_trace.md                    # AttackTrace model (Markdown)
    ├── report.md                          # Report model (Markdown)
    ├── report.json                        # Report model (JSON)
    ├── iterations/
    │   ├── iteration_1.json               # IterationResult model
    │   ├── iteration_2.json
    │   └── ...
    ├── experiments/
    │   ├── {experiment_id}/
    │   │   ├── model_checkpoint.pth       # Experiment artifacts
    │   │   ├── generated_images/
    │   │   └── logs.txt
    │   └── ...
    └── judges/
        ├── judge_security_expert.json     # JudgeEvaluation model
        ├── judge_ml_researcher.json
        └── ...

rag/
├── papers/
│   ├── data_unlearning/
│   │   ├── paper1.pdf
│   │   ├── paper2.pdf
│   │   └── ...
│   ├── concept_erasure/
│   │   ├── paper1.pdf
│   │   └── ...
│   └── attack_methods/
│       ├── paper1.pdf
│       └── ...
├── faiss_index.bin                        # FAISS vector index
└── paper_metadata.json                    # Paper metadata for citations

configs/
├── prompts/
│   ├── hypothesis_generator_data_based.yaml    # AgentPromptConfig
│   ├── hypothesis_generator_concept_erasure.yaml
│   ├── critic.yaml
│   ├── query_generator.yaml
│   ├── evaluator_data_based.yaml
│   ├── evaluator_concept_erasure.yaml
│   ├── reporter.yaml
│   └── judges.yaml
├── thresholds/
│   ├── data_based.yaml                    # Evaluation thresholds
│   └── concept_erasure.yaml
├── tasks/
│   ├── data_based_unlearning.yaml         # Task-specific config (seed templates)
│   └── concept_erasure.yaml
└── personas/
    └── judges.yaml                        # Judge persona definitions

memory/
└── [managed by CAMEL-AI memory system]    # MemoryEntry storage (implementation-specific)
```

## Schema Definitions (Pydantic Models)

All data models defined in the Data Models section are implemented as Pydantic models with schema validation:

```python
# Example: loop/models.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class LoopState(BaseModel):
    task_id: str = Field(..., description="Unique task identifier")
    task_type: str = Field(..., pattern="^(data_based_unlearning|concept_erasure)$")
    current_iteration: int = Field(ge=0, le=10)
    max_iterations: int = Field(default=10, ge=1, le=20)
    current_state: str = Field(..., pattern="^(HYPOTHESIS_GENERATION|CRITIC_DEBATE|RAG_RETRIEVAL|EXPERIMENT_EXECUTION|EVALUATION|FEEDBACK|COMPLETED|FAILED)$")
    vulnerability_found: bool = False
    started_at: datetime
    updated_at: datetime
    metadata: dict = Field(default_factory=dict)
```

**Schema Validation:** All JSON files are validated against Pydantic models on read/write, ensuring data integrity and preventing corruption.

**Migration Path:** Repository pattern abstractions enable future migration to PostgreSQL or MongoDB without changing component interfaces. Database schema would directly map to Pydantic models.
