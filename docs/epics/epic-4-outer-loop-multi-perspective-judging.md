# Epic 4: Outer Loop & Multi-Perspective Judging

**Expanded Goal**: Implement the Reporter agent that generates academic-format papers (Introduction, Methods, Experiments, Results, Discussion, Conclusion) with citation integration, and the multi-perspective LLM Judge system with 3-5 personas evaluating findings from different angles. By the end of this epic, AUST completes the full end-to-end autonomous research workflow, generating publishable reports with multi-faceted evaluation.

---

## Integration Architecture

### Data Flow Overview

```
Inner Loop (Existing)                    Outer Loop (New - Epic 4)
─────────────────────                    ──────────────────────────
InnerLoopOrchestrator                    OuterLoopOrchestrator
  ↓                                        ↓
InnerLoopState                           ReporterAgent
  ├─ iterations[]                          ├─ Reads: InnerLoopState
  ├─ task_spec                             ├─ Reads: attack_trace.json/md
  ├─ exit_condition                        ├─ Reads: paper_metadata (RAG)
  ├─ attack_trace_file                     ↓
  └─ output_dir/                         Generated Report
                                           ├─ Introduction
                                           ├─ Methods
                                           ├─ Experiments
                                           ├─ Results
                                           ├─ Discussion
                                           └─ Conclusion
                                             ↓
                                         JudgeAgent (3-5 personas)
                                           ├─ Security Expert
                                           ├─ ML Researcher
                                           ├─ Privacy Advocate
                                           ├─ Skeptical Reviewer
                                           └─ Industry Practitioner
                                             ↓
                                         Aggregated Judgment Summary
```

### Required New Components

**Agents** (`aust/src/agents/`)
- `reporter.py` - Academic report generation with section-wise content
- `judge.py` - Multi-perspective evaluation with persona loading

**Orchestration** (`aust/src/loop/`)
- `outer_loop_orchestrator.py` - Coordinates Reporter → Judges workflow
- Integration hook in `inner_loop_orchestrator.py` to trigger outer loop

**Data Models** (`aust/src/data_models/`)
- `report.py` - Report sections, metadata, citation tracking
- `judgment.py` - Judge evaluation, persona, scoring dimensions
- `outer_loop_state.py` - Outer loop state tracking

**Configuration Files** (`aust/configs/`)
```yaml
prompts/
  ├─ reporter.yaml              # Report generation prompts per section
  └─ judge_personas.yaml        # 3-5 judge persona definitions

models/
  ├─ reporter.yaml              # Model config for Reporter
  └─ judge.yaml                 # Model config for Judge

templates/
  └─ report_template.md         # Academic paper structure template
```

### Output Directory Structure

```
outputs/{task_id}/
  ├─ loop_state.json           # Existing: InnerLoopState
  ├─ attack_trace.json         # Existing: JSON format
  ├─ attack_trace.md           # NEW (Story 4.3): Human-readable
  ├─ debates/                  # Existing: Debate sessions
  ├─ experiments/              # Existing: Experiment results
  ├─ reports/                  # NEW (Story 4.1): Reporter output
  │   └─ report_{run_id}.md
  ├─ judgments/                # NEW (Story 4.5): Judge evaluations
  │   ├─ judge_security_{run_id}.md
  │   ├─ judge_ml_researcher_{run_id}.md
  │   └─ summary_{run_id}.md
  └─ final_{run_id}/           # NEW (Story 4.6): Delivery package
      ├─ report.md
      ├─ attack_trace.json
      ├─ attack_trace.md
      └─ judgments_summary.md
```

### Entry Point Integration

**CLI Arguments** (add to `aust/scripts/main.py`):
```bash
--enable-outer-loop          # Trigger outer loop after inner loop
--skip-judges                # Generate report only, skip judging
--judge-personas PERSONAS    # Select specific personas (default: all)
--report-format FORMAT       # markdown, latex, pdf (MVP: markdown)
```

**Invocation Example**:
```bash
# Full system (inner + outer loop):
python aust/scripts/main.py --task-type concept_erasure \
    --enable-outer-loop \
    --max-iterations 10 \
    --judge-personas "security,ml_researcher,privacy"
```

### Data Model Requirements

**Report Model** (Story 4.1, 4.2):
```python
class ReportSection(BaseModel):
    section_name: str  # Introduction, Methods, etc.
    content: str
    citations: list[str]  # BibTeX keys

class AcademicReport(BaseModel):
    report_id: str
    task_id: str
    generated_at: datetime
    sections: dict[str, ReportSection]
    metadata: dict  # date, task_type, method, iterations
```

**Judgment Model** (Story 4.4, 4.5):
```python
class JudgePersona(BaseModel):
    name: str  # "Security Expert"
    role: str
    evaluation_criteria: list[str]
    scoring_dimensions: dict[str, str]  # {novelty: "1-5 scale"}
    prompt_tone: str

class JudgmentEvaluation(BaseModel):
    judge_persona: str
    summary_assessment: str
    strengths: list[str]
    weaknesses: list[str]
    scores: dict[str, float]  # {novelty: 4.0, rigor: 3.5}
    recommendations: list[str]
```

**Outer Loop State** (Story 4.6):
```python
class OuterLoopState(BaseModel):
    task_id: str
    inner_loop_state: InnerLoopState
    report_generated: bool
    report_path: Optional[Path]
    judgments: list[JudgmentEvaluation]
    completed_at: Optional[datetime]
```

### Implementation Sequence

1. **Story 4.3** (Attack Trace Enhancement) - Foundation layer
2. **Story 4.1** (Reporter Structure) - Report skeleton
3. **Story 4.2** (Reporter Content) - Content generation with citations
4. **Story 4.4** (Judge Personas) - Persona definitions
5. **Story 4.5** (Judge Implementation) - Judge agent
6. **Story 4.6** (Outer Loop Orchestrator) - Workflow coordination
7. **Story 4.7** (End-to-End Test) - Full system validation

### Critical Integration Considerations

**RAG Integration**:
- Reporter needs access to `rag/paper_metadata.json` for citations
- BibTeX extraction from paper cards stored during RAG retrieval

**State Management**:
- Inner loop saves state → Outer loop reads state (read-only)
- Outer loop maintains separate `outer_loop_state.json`
- No modification of inner loop state by outer loop

**Error Handling**:
- Reporter fails → Save partial report + error log, continue to judges if possible
- Judge fails → Continue with remaining judges, flag missing perspective
- Outer loop errors should NOT invalidate inner loop results

**Performance Targets**:
- Reporter: ~6 LLM calls (one per section)
- Judges: Parallel execution (5 judges → ~1 round if parallel)
- Estimated outer loop time: 15-30 minutes
- Total system (inner + outer): <5 hours (NFR7)

---

## Story 4.1: Reporter Agent - Report Structure & Template

As a **researcher**,
I want **a Reporter agent that generates academic paper structure with proper sections**,
so that **the system produces publication-ready reports automatically**.

### Acceptance Criteria

1. Reporter agent implemented in `agents/reporter.py`
2. Report template defined in `configs/report_template.md` with sections: Introduction, Methods, Experiments, Results, Discussion, Conclusion
3. Reporter accepts inputs: attack traces (all iterations), successful experiments, evaluation results, retrieved paper references
4. Reporter generates section outlines based on inputs (e.g., Methods describes hypothesis generation + experiment execution workflow)
5. Generated report saved to `aust/outputs/reports/report_{run_id}.md` in Markdown format
6. Report includes metadata: date, task type, target unlearning method, number of iterations

## Story 4.2: Reporter Agent - Content Generation with Citations

As a **researcher**,
I want **the Reporter to populate report sections with detailed content and cite retrieved papers**,
so that **the generated report is comprehensive and properly attributed**.

### Acceptance Criteria

1. Reporter generates Introduction: problem statement, why unlearning security matters, objectives of the stress test
2. Reporter generates Methods: describes AUST architecture, hypothesis generation process (including RAG and critic debate), experiment execution, evaluation criteria
3. Reporter generates Experiments: details the specific unlearning method tested, datasets used, experiment parameters
4. Reporter generates Results: presents vulnerability discovered (or lack thereof), attack trace summary, evaluation metrics, key observations
5. Reporter generates Discussion: interprets findings, compares to static benchmarks (if applicable), discusses implications for unlearning robustness
6. Reporter generates Conclusion: summarizes contribution, limitations, future work
7. Citations integrated using BibTeX format references from `rag/paper_metadata.json`

## Story 4.3: Attack Trace Enhancement for Report Integration

As a **developer**,
I want **attack traces to include sufficient detail for direct inclusion in academic reports**,
so that **the Reporter can use traces as primary evidence**.

### Acceptance Criteria

1. Attack trace format enhanced to include: iteration number, hypothesis rationale, experiment design justification, quantitative results, qualitative observations
2. Traces document hypothesis evolution showing how critic feedback improved proposals
3. Traces include failure analysis: why certain hypotheses didn't lead to vulnerabilities
4. Dual format: JSON (machine-readable) saved to `aust/outputs/attack_traces/trace_{run_id}.json` + Markdown (human-readable) saved to `aust/outputs/attack_traces/trace_{run_id}.md`
5. Reporter can parse and extract key sections from traces for Results and Discussion sections

## Story 4.4: Judge Persona Definitions

As a **researcher**,
I want **to define 3-5 LLM judge personas with specific evaluation criteria**,
so that **the judging system provides diverse, meaningful perspectives**.

### Acceptance Criteria

1. Judge persona definitions created in `configs/judge_personas.yaml` with 3-5 personas:
   - Security Expert: evaluates practical exploitability, real-world attack feasibility
   - ML Researcher: evaluates novelty, methodological rigor, scientific contribution
   - Privacy Advocate: evaluates privacy implications, compliance relevance
   - Skeptical Reviewer: challenges claims, identifies weaknesses, suggests improvements
   - Industry Practitioner: evaluates deployment relevance, risk assessment utility
2. Each persona has: name, role description, evaluation criteria (bullet list), scoring dimensions (e.g., novelty 1-5, rigor 1-5)
3. Persona prompts specify tone and focus area (e.g., Security Expert is pragmatic and risk-focused)

## Story 4.5: Judge Agent Implementation

As a **researcher**,
I want **Judge agents that evaluate generated reports from their persona's perspective**,
so that **the system provides multi-faceted evaluation of findings**.

### Acceptance Criteria

1. Judge agent implemented in `agents/judge.py` that can instantiate any persona from `judge_personas.yaml`
2. Judge accepts inputs: generated report, attack traces, experiment results
3. Judge generates structured evaluation: summary assessment, strengths, weaknesses, scoring (per persona's dimensions), recommendations
4. Each judge evaluation saved to `aust/outputs/judgments/judge_{persona_name}_{run_id}.md`
5. Judges run independently (can be parallelized if time permits)
6. All judge outputs aggregated into `aust/outputs/judgments/summary_{run_id}.md`

## Story 4.6: Outer Loop Orchestrator

As a **researcher**,
I want **an Outer Loop Orchestrator that coordinates Reporter → Judges workflow**,
so that **the system automatically generates and evaluates reports after vulnerability discovery**.

### Acceptance Criteria

1. Outer Loop Orchestrator implemented in `loop/outer_loop.py`
2. Orchestrator triggered when inner loop exits with VULNERABILITY_FOUND or max iterations
3. Orchestrator workflow: call Reporter → wait for report generation → call all Judges → aggregate judgments
4. Orchestrator logs all outputs and timestamps for reproducibility
5. Default outer loop iteration budget is 5 (configurable via `configs/loop_config.yaml`); see Story 4.6
6. Final output package saved to `aust/outputs/final_{run_id}/` containing: report, attack traces (JSON + MD), all judge evaluations, aggregated summary

## Story 4.7: End-to-End Full System Test

As a **researcher**,
I want **to run the complete AUST system end-to-end (inner loop + outer loop)**,
so that **we validate autonomous research workflow from hypothesis to judged report**.

### Acceptance Criteria

1. Execute full system on both tasks: data-based unlearning and concept-erasure
2. Inner loop discovers at least one vulnerability per task (or reaches max iterations with meaningful attempt)
3. Reporter generates comprehensive academic-format report for each task
4. All 3-5 judges evaluate each report and provide diverse perspectives
5. Attack traces are reproducible (verified with manual re-execution of key steps)
6. System autonomy validated: ≥ 90% of workflow runs without human intervention (NFR10)
7. Performance: full system (inner + outer loop) completes within < 5 hours per task (NFR7)
8. All outputs (reports, traces, judgments) meet quality standards for paper integration
