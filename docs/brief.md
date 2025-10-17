# Project Brief: AUST

## Executive Summary

AUST (AI Scientist for Autonomous Unlearning Security Testing) is an autonomous AI system that discovers and explains vulnerabilities in machine unlearning methods through an adaptive, hypothesis-driven research loop. The system addresses the critical gap between static benchmark testing and real-world adversarial threats in privacy-preserving machine learning, where current evaluation methods fail to uncover failure modes that motivated adversaries could exploit.

AUST targets the research community and ML practitioners working on data-based machine unlearning and concept-based erasure (e.g., in generative models). The key value proposition is **autonomous end-to-end scientific discovery**: AUST proposes targeted stress tests using paper retrieval and past results, designs and executes experiments, interprets outcomes, and generates academic reports with multi-perspective LLM judge evaluation—demonstrating an AI system performing the complete scientific workflow on a societally relevant topic (privacy, compliance, model safety).

**Impact:**
- **For research:** Uncovers failure modes missed by fixed test suites; creates reusable, adaptive stress tests
- **For deployment:** Provides risk profiles and attack feasibility assessments that inform policy, procurement, and safety cases

## Problem Statement

**Current State and Pain Points:**

Machine unlearning methods promise to remove specific data or concepts from trained models—a critical capability for GDPR compliance, bias mitigation, and preventing misuse of generative AI. However, current evaluation approaches rely on static benchmarks that test whether unlearning succeeds under benign conditions. These fixed test suites cannot anticipate the creative, adaptive attacks that motivated adversaries would employ in the real world.

Existing evaluation has three critical gaps:

1. **Static vs. Adaptive Threats**: Benchmarks use predefined tests that don't evolve based on model behavior or discovered weaknesses
2. **Limited Attack Diversity**: Test suites focus on known attack patterns (e.g., membership inference) but miss novel attack vectors
3. **No Feedback Loop**: When vulnerabilities are found, insights aren't systematically used to discover related failure modes

**Impact of the Problem:**

- **For researchers**: Time-consuming manual experimentation to find edge cases; published methods may have undiscovered vulnerabilities
- **For practitioners**: Deploying unlearning methods without confidence in their robustness against adversarial probing
- **For society**: Privacy regulations (GDPR "right to be forgotten") rely on unlearning methods that may not actually work under attack

**Why Existing Solutions Fall Short:**

Static benchmark suites (e.g., DeepUnlearn) provide reproducibility but not adversarial robustness testing. Manual red-teaming by researchers is thorough but doesn't scale and lacks systematic coverage. There is no existing system that performs the full hypothesis → experiment → interpretation loop autonomously.

**Urgency and Importance:**

As generative AI deployment accelerates and privacy regulations tighten, the stakes for robust unlearning are increasing. The research community needs tools that can proactively discover vulnerabilities before adversaries do, and this capability needs to be demonstrated on concrete examples (data-based and concept-erasure unlearning) by Nov 14th to contribute to the field.

## Proposed Solution

**Core Concept and Approach:**

AUST implements an autonomous research scientist that operates through two nested loops:

1. **Inner Research Loop** (iterative discovery): Hypothesis Generator proposes targeted stress tests → Query Generator retrieves relevant papers/methods → Experiment Executor runs tests on unlearned models → Evaluator assesses results (multi-modal: metrics + VLM) → feedback returns to improve hypotheses. This loop continues until a vulnerability is discovered or iteration limit is reached.

2. **Outer Loop** (reporting and validation): Once vulnerabilities are found, Reporter generates an academic-format paper (Introduction, Methods, Experiments, Results, Discussion, Conclusion), and multiple LLM Judges evaluate the findings from different perspectives (novelty, rigor, reproducibility, impact).

**Key Differentiators:**

- **Adaptive vs. Static**: Unlike fixed benchmarks, AUST learns from failed experiments and adjusts hypothesis generation using RAG-based paper retrieval and memory of successful attacks
- **Multi-Agent Debate**: A Critic Agent challenges the Hypothesis Generator after initial iterations, forcing more novel and rigorous test proposals
- **End-to-End Automation**: From hypothesis to published report with human roles clearly documented—demonstrates AI performing the complete scientific workflow
- **Multi-Modal Evaluation**: Combines quantitative metrics (threshold-based forget accuracy for data-based unlearning) with qualitative VLM analysis (generation-based concept leakage detection)

**Why This Solution Will Succeed:**

- **Built on Proven Infrastructure**: Integrates DeepUnlearn (data-based) and existing concept-erasure methods (from GitHub) as MCP tools within CAMEL-AI's multi-agent framework
- **Pragmatic MVP Strategy**: Starts with single critic-generator pair and basic RAG, scales to orchestrator-worker architecture only if needed
- **Knowledge Integration**: Paper retrieval (task-specific + shared attack corpus) combined with long-term memory ensures hypothesis quality improves over time

**High-Level Vision:**

AUST demonstrates that AI systems can conduct rigorous, adaptive security research autonomously. Beyond the immediate contribution (discovering unlearning vulnerabilities), this establishes a template for AI-driven red-teaming in other ML safety domains.

**Key Outputs:**

- **Attack Traces**: Step-by-step records from the inner research loop showing the path from initial hypothesis → experiment design → execution → evaluation → refinement. These traces document exactly how vulnerabilities were discovered, providing both researchers and practitioners with actionable insights and reproducible attack paths. Users can follow these traces to discover new knowledge and understand vulnerability mechanisms.
- **Multi-Perspective Judging**: LLM judges evaluate findings from multiple angles (novelty, rigor, reproducibility, impact, practical exploitability), providing decision-makers with diverse facets to inform risk assessment and procurement decisions. This multi-faceted evaluation supports informed decision-making across technical and non-technical stakeholders.

## Target Users

### Primary User Segment: Machine Learning Researchers (ML Safety & Privacy Focus)

**Demographic/Firmographic Profile:**
- Academic researchers and PhD students working on machine unlearning, privacy-preserving ML, and AI safety
- Located at universities, research labs (e.g., OpenAI, Anthropic, Google DeepMind), or independent research organizations
- Publishing in top-tier venues (NeurIPS, ICML, ICLR, USENIX Security)

**Current Behaviors and Workflows:**
- Develop new unlearning methods and test them against static benchmarks (e.g., DeepUnlearn)
- Manually design adversarial tests to probe their methods' robustness
- Spend significant time on experimental iteration: hypothesis → implementation → analysis
- Write academic papers documenting methods, experiments, and results

**Specific Needs and Pain Points:**
- **Need comprehensive adversarial evaluation** but lack systematic tools beyond static benchmarks
- **Time-consuming manual red-teaming** limits coverage of the attack space
- **Uncertain about real-world robustness**: static benchmarks don't reflect motivated adversaries
- **Want reproducible, reusable stress tests** that can be shared with the community
- **Need to observe drawbacks** of their proposed methods before publication

**Goals They're Trying to Achieve:**
- Publish novel research demonstrating robust unlearning methods
- Discover and patch vulnerabilities before publication
- Contribute evaluation tools/datasets to the research community
- Advance the state-of-the-art in privacy-preserving ML
- **Follow attack traces to discover new knowledge** about failure modes and vulnerability mechanisms
- **Use multi-perspective judging** to validate their findings from different research angles

---

### Secondary User Segment: Enterprise Buyers & Compliance Officers

**Demographic/Firmographic Profile:**
- Decision-makers at companies evaluating machine unlearning software for procurement
- Compliance officers, legal teams, and CISOs responsible for GDPR/CCPA adherence
- Operating in regulated industries (tech platforms, healthcare, finance, e-commerce)
- Ranging from mid-size companies to large enterprises with significant privacy obligations

**Current Behaviors and Workflows:**
- Evaluate vendor claims about unlearning capabilities before purchase
- Conduct due diligence and security assessments on ML software
- Document compliance procedures for auditors, regulators, and internal stakeholders
- Make build-vs-buy decisions for privacy-preserving ML infrastructure

**Specific Needs and Pain Points:**
- **Cannot independently verify vendor security claims** without deep ML expertise
- **Need objective evidence of robustness** to justify procurement decisions to stakeholders
- **Face regulatory risk** if deployed unlearning solutions fail under adversarial conditions
- **Lack tools to assess "how exploitable"** a given unlearning method is in practice
- **Need clear, interpretable evidence** for non-technical stakeholders (legal, executive leadership)

**Goals They're Trying to Achieve:**
- Make informed procurement decisions based on security assessments
- Demonstrate due diligence to regulators and auditors
- Minimize organizational liability from privacy failures
- **Use attack traces as decision-making evidence**: step-by-step vulnerability exposures show exactly where and how methods fail, enabling informed discussions with vendors
- **Leverage multi-perspective judging**: judges provide diverse facets (security risk, practical exploitability, compliance implications) that support decision-making across technical and business stakeholders
- Negotiate with vendors from a position of knowledge about method limitations

**Value Proposition for This Segment:**

AUST provides two critical decision-making tools:
1. **Attack Traces**: Transparent, reproducible paths showing vulnerability discovery process—actionable intelligence for vendor negotiations and risk assessment
2. **Multi-Perspective Judging**: Evaluations from multiple angles (technical rigor, practical exploitability, compliance risk) translate findings into language appropriate for different stakeholders

---

### Tertiary User Segment: ML Practitioners & Method Developers

**Demographic/Firmographic Profile:**
- ML engineers and data scientists developing or deploying unlearning solutions
- Researchers at companies building privacy-preserving ML products
- Open-source contributors to unlearning libraries and tools

**Current Behaviors and Workflows:**
- Implement unlearning methods for internal use or as product features
- Evaluate third-party or open-source unlearning methods for deployment
- Conduct security testing and hardening before production deployment
- Document technical specifications and limitations

**Specific Needs and Pain Points:**
- **Limited security/privacy expertise** to conduct thorough red-teaming themselves
- **Want to understand failure modes** to prioritize hardening efforts
- **Need reproducible test suites** that go beyond static benchmarks
- **Struggle to anticipate novel attacks** that adversaries might use

**Goals They're Trying to Achieve:**
- Deploy robust unlearning solutions with confidence
- **Follow attack traces to observe and patch vulnerabilities** revealed by AUST's inner loop
- **Use attack paths as templates** for their own security testing
- Improve method design based on systematic adversarial feedback
- Contribute more secure implementations to the community

## Goals & Success Metrics

### Business Objectives

- **Demonstrate autonomous end-to-end AI scientific workflow** by Nov 14th: Complete system performing hypothesis generation → experimentation → interpretation → reporting → judging on both data-based and concept-erasure unlearning tasks
- **Publish research contribution**: Submit paper demonstrating AUST's ability to discover vulnerabilities missed by static benchmarks, with at least one novel vulnerability per task type
- **Establish evaluation methodology**: Create reusable attack traces and stress tests that the research community can adopt for adversarial robustness testing
- **Enable practical deployment decisions**: Provide enterprise buyers with actionable risk assessments based on multi-perspective judging outputs

### User Success Metrics

- **For Researchers**: Time to discover vulnerabilities reduced by 50%+ compared to manual red-teaming; attack traces enable reproduction of findings in < 1 hour
- **For Enterprise Buyers**: Decision confidence score increases (measured via user feedback); procurement decisions completed 30% faster with AUST-generated reports
- **For Practitioners**: Vulnerability patching success rate improves; attack paths successfully adapted to their own testing workflows

### Key Performance Indicators (KPIs)

- **Vulnerability Discovery Rate**: At least 1 exploitable vulnerability discovered per unlearning method tested (target: 2+ per method)
- **Hypothesis Quality**: Critic agent challenges result in 30%+ improvement in hypothesis novelty/rigor (measured by judge scores)
- **Loop Efficiency**: Average iterations to vulnerability discovery ≤ 10 loops (inner research loop)
- **Attack Trace Usability**: 80%+ of generated traces are reproducible by independent users without additional clarification
- **Multi-Perspective Coverage**: Judge evaluation covers ≥ 4 distinct perspectives (novelty, rigor, reproducibility, impact, exploitability)
- **System Autonomy**: ≥ 90% of research workflow completed without human intervention (excluding initial setup and final paper integration)
- **Timeline Adherence**: Full system operational by Nov 7th (Day 21), paper draft complete by Nov 14th

## MVP Scope

### Core Features (Must Have)

- **Single Critic-Generator Agent Pair**: Basic hypothesis generator with one critic agent that debates after first iteration to improve hypothesis quality. Rationale: Proves the adaptive feedback concept without complex orchestration overhead.

- **Inner Research Loop**: Complete cycle of Hypothesis Generation → Query Generation → RAG Retrieval → Experiment Execution → Multi-Modal Evaluation → Feedback Integration. Loop exits when vulnerability found or max iterations (10) reached. Rationale: Core of autonomous research capability.

- **DeepUnlearn Integration (Data-Based Unlearning)**: MCP FunctionTool integration enabling programmatic model unlearning and evaluation for data-based tasks. Rationale: Provides proven benchmark infrastructure for first task type.

- **Concept-Erasure Integration**: GitHub-sourced concept-erasure methods integrated as MCP FunctionTool. Rationale: Demonstrates generality across both major unlearning paradigms.

- **Basic RAG System**: 10-20 key papers per task domain with simple semantic search (no sophisticated card extraction). Rationale: Sufficient for hypothesis quality improvement; advanced indexing is infrastructure, not contribution.

- **Multi-Modal Evaluation**: Threshold-based metrics (forget accuracy for data-based) + VLM analysis (generation-based leakage for concept-erasure). Rationale: Demonstrates qualitative + quantitative assessment capability.

- **Memory System**: CAMEL-AI long-term storage capturing successful vulnerability discoveries for future reference. Rationale: Enables cumulative learning and attack trace reuse.

- **Academic Report Generator**: Automatic generation of paper-format reports (Intro, Method, Experiments, Results, Discussion, Conclusion) with citation integration. Rationale: Demonstrates end-to-end scientific workflow.

- **Multi-Perspective LLM Judges**: 3-5 judge personas evaluating findings from different angles (novelty, rigor, reproducibility, impact, exploitability). Rationale: Provides decision-making support for diverse stakeholders.

- **Attack Trace Generation**: Step-by-step documentation of inner loop iterations showing hypothesis evolution and vulnerability discovery path. Rationale: Key output for reproducibility and knowledge transfer.

### Out of Scope for MVP

- Orchestrator + worker architecture (start with single critic-generator pair)
- Sophisticated paper indexing with structured cards (use simple semantic search)
- Parallel hypothesis generation (run sequentially first)
- Extensive ablation studies (focus on end-to-end demonstration)
- Multiple unlearning methods per task type (1-2 methods per type sufficient)
- Real-time dashboard or web UI (command-line execution acceptable)
- Automated hyperparameter optimization (manual/default settings)
- Integration with additional unlearning benchmarks beyond DeepUnlearn

### MVP Success Criteria

**The MVP is successful if:**
1. System discovers at least **1 exploitable vulnerability** in data-based unlearning
2. System discovers at least **1 exploitable vulnerability** in concept-erasure
3. Attack traces are **reproducible** by independent users (tested with one external validator)
4. Generated academic report is **coherent and cite-worthy** (human-readable, properly structured)
5. Judge outputs provide **diverse perspectives** (≥3 distinct evaluation angles)
6. **90%+ of workflow runs autonomously** without human intervention (after initial setup)
7. **Completed by Nov 7th** with buffer time for paper writing

## Post-MVP Vision

### Phase 2 Features

If time permits after MVP completion (between Nov 7-14), consider these enhancements for the paper:

- **Parallel Hypothesis Generation**: After critic debate, generate multiple hypothesis variants concurrently to explore diverse attack vectors simultaneously. Strengthens the "adaptive exploration" narrative.

- **Ablation Studies**: Quick comparison of system components (with/without critic, with/without RAG) to quantify contribution of each element. Adds rigor to the paper's claims about multi-agent debate and knowledge retrieval.

- **Additional Vulnerability Examples**: Test 1-2 more unlearning methods per task type if the first discoveries come quickly. Demonstrates generality and robustness of the approach.

- **Attack Trace Analysis**: Qualitative analysis of generated traces showing patterns in successful vs. failed hypothesis paths. Provides insights for future work section.

### Long-term Research Vision

**Future Work (for paper's Discussion/Conclusion):**

- **Cross-Domain Transfer**: Apply autonomous red-teaming methodology to other ML safety domains (fairness auditing, robustness testing, backdoor discovery)
- **Community Resource**: Publish attack traces as reproducible datasets that other researchers can build upon
- **Meta-Learning**: Investigate how system could improve hypothesis generation by learning from patterns across multiple vulnerability discoveries
- **Standards Development**: Explore how attack traces and judging criteria could inform industry standards for unlearning robustness evaluation

### Expansion Opportunities

**Potential Impact Areas (for paper's Broader Impact section):**

- **Research Community**: Reusable stress tests and evaluation methodology for adversarial robustness testing
- **Industry Deployment**: Framework for security assessment of unlearning methods before production deployment
- **Regulatory Compliance**: Evidence-based approach for validating GDPR/CCPA data deletion claims
- **ML Safety**: Template for AI-driven adversarial evaluation in other safety-critical domains

## Technical Considerations

### Platform Requirements

- **Target Platforms:** Docker + Kubernetes environment with H200 GPU access
- **Compute Resources:** NVIDIA H200 GPUs via Kubernetes job scheduling (job.yaml configuration)
- **Container Environment:** Conda-based Docker image (conda_Dockerfile)
- **Performance Requirements:**
  - Inner research loop iteration time: < 30 minutes per cycle (hypothesis → experiment → evaluation)
  - Full vulnerability discovery: < 5 hours for one complete research loop (initial hypothesis to final report)
  - RAG retrieval latency: < 5 seconds per query

### Technology Stack

**Frontend:** Command-line interface

**Backend:**
- **CAMEL-AI**: Multi-agent orchestration, society/workforce APIs, long-term memory storage, MCP tool integration
  - **Installation:** Dev mode (editable install) to allow source code tweaking as needed
- **Python 3.11+**: Primary implementation language

**Database/Storage:**
- **Vector Database**: FAISS, Chroma, or similar for semantic search
- **Embedding Model**: Via OpenRouter or Sentence-Transformers
- **File Storage**: Persistent volumes for paper corpus, experiment results, generated outputs

**Hosting/Infrastructure:**
- **Container Orchestration**: Kubernetes with H200 GPU job scheduling
- **Docker**: Conda-based containerization
- **GPU Access**: H200 GPUs available via existing infrastructure

### Technology Preferences

**Agent Components:**
- **LLM/VLM Backend**: OpenRouter API for flexible model access (GPT-4o, Claude 3.5 Sonnet, and other options)
- **Multi-Modal Support**: VLM capabilities via OpenRouter for generation-based evaluation

**Unlearning Infrastructure:**
- **DeepUnlearn**: Git submodule in repository for data-based unlearning (enables version control and local modifications)
- **Concept-Erasure Methods**: GitHub repositories (e.g., EraseDiff, concept ablation tools) integrated as MCP FunctionTools or submodules

**Experiment Execution:**
- **PyTorch**: Model training/evaluation framework
- **Container Isolation**: Docker-based experiment execution for reproducibility

**Output Generation:**
- **Markdown/LaTeX**: Report generation format
- **Citation Management**: BibTeX integration for academic references

### Architecture Considerations

**Repository Structure:**
```
aust/
├── agents/              # Hypothesis Generator, Critic, Query Generator, Evaluator, Reporter, Judges
├── tools/               # MCP FunctionTool wrappers for concept-erasure methods
├── rag/                 # Paper corpus, embedding, retrieval logic
├── memory/              # CAMEL-AI long-term storage integration
├── loop/                # Inner/outer loop orchestration
├── outputs/             # Generated reports, attack traces, judge evaluations
├── configs/             # Agent prompts, evaluation thresholds, system parameters
├── experiments/         # Experiment execution sandbox
├── submodules/
│   └── DeepUnlearn/     # Git submodule for data-based unlearning
├── external/
│   └── camel/           # CAMEL-AI in dev/editable mode
├── docker/
│   ├── conda_Dockerfile # Conda-based container definition
│   └── job.yaml         # Kubernetes job specification
└── requirements.txt     # Python dependencies
```

**Service Architecture:**
- **Containerized Application**: Single Python application running in Docker with Kubernetes orchestration
- **Stateful Execution**: Memory persists across loop iterations via mounted volumes; attack traces saved incrementally
- **Dev Mode Dependencies**: CAMEL-AI installed with `pip install -e` for source modifications
- **Submodule Management**: DeepUnlearn as git submodule for version tracking and local modifications

**Integration Requirements:**
- **OpenRouter API**: Single API endpoint for multiple LLM/VLM model access
- **MCP Tool Protocol**: CAMEL-AI's FunctionTool interface for calling DeepUnlearn and concept-erasure methods
- **Kubernetes Resources**: GPU allocation, persistent volumes for outputs/memory
- **Container Registry**: Docker image storage and versioning

**Security/Compliance:**
- **Container Isolation**: Experiment execution sandboxed within Docker containers
- **API Rate Limiting**: Handle OpenRouter quotas gracefully with retry logic
- **Data Privacy**: No sensitive data in training/evaluation (use public datasets only)
- **Reproducibility**:
  - Pinned dependency versions in requirements.txt
  - Git submodule commits for DeepUnlearn version control
  - Random seed control and deterministic execution
  - Complete attack trace logging for reproducibility

## Constraints & Assumptions

### Constraints

**Budget:**
- OpenRouter API usage costs (LLM/VLM API calls for agents, embeddings)
- Budget constraints favor efficient prompt design and caching strategies
- No budget for proprietary datasets or commercial software licenses

**Timeline:**
- **Hard deadline: November 14th, 2025** for paper completion and thesis integration
- 3-week implementation sprint (Oct 24 - Nov 13)
- Week 4 reserved exclusively for paper writing
- No buffer for major architectural pivots after Nov 7th

**Resources:**
- **Single developer** (you) with concurrent thesis/coursework obligations
- Limited availability for debugging/iteration during implementation
- H200 GPU access available but shared resource (potential job queue delays)

**Technical:**
- CAMEL-AI framework maturity: may require source code modifications for edge cases
- DeepUnlearn benchmark: limited to supported unlearning methods and datasets
- Concept-erasure methods: dependency on third-party GitHub repositories (maintenance/compatibility risks)
- OpenRouter API: rate limits and model availability may constrain throughput

### Key Assumptions

- **Hypothesis quality is addressable**: Basic RAG + critic agent debate will produce hypotheses good enough to discover at least 1 vulnerability per task
- **DeepUnlearn is sufficient**: Existing DeepUnlearn methods and datasets cover representative data-based unlearning scenarios
- **Concept-erasure methods are accessible**: GitHub repositories for concept-erasure are well-documented and integrable within 2-3 days
- **CAMEL-AI MCP tools work**: FunctionTool interface can wrap unlearning methods without major refactoring
- **VLM evaluation is reliable**: Vision models can meaningfully assess concept leakage in generated images
- **10 iterations is enough**: Inner research loop will discover vulnerabilities within 10 cycles on average
- **Attack traces are interpretable**: Generated step-by-step records will be understandable by researchers without extensive ML background
- **Judge diversity matters**: Multi-perspective evaluation (3-5 personas) provides meaningful signal for decision-making
- **Static benchmark comparison exists**: Can reference DeepUnlearn results to demonstrate adaptive > static evaluation
- **Paper corpus is obtainable**: 10-20 key papers per task are publicly accessible and sufficient for RAG

## Risks & Open Questions

### Key Risks

- **Hypothesis Generation Quality (HIGH)**: If basic RAG + critic agent produces weak/repetitive hypotheses, the system may fail to discover vulnerabilities. Mitigation: Pre-load seed templates from literature; implement minimal human-in-the-loop feedback after first 2-3 loops.

- **Integration Complexity (MEDIUM)**: Wrapping DeepUnlearn and concept-erasure methods as CAMEL-AI MCP FunctionTools may require unexpected refactoring. Mitigation: Budget 3-4 days for integration in Week 1; test with simplest method first.

- **Vulnerability Discovery Failure (HIGH)**: System may not discover exploitable vulnerabilities within 10 iterations or deadline. Mitigation: Start with known-fragile unlearning methods; use progressive difficulty (easy → hard targets).

- **Timeline Slippage (HIGH)**: Single-developer constraint + concurrent obligations may cause delays. Mitigation: Strict prioritization (P0 only until Nov 7); daily progress tracking; cut scope aggressively if behind.

- **VLM Evaluation Reliability (MEDIUM)**: Vision models may produce inconsistent or misleading assessments of concept leakage. Mitigation: Combine VLM with quantitative metrics; validate VLM outputs against ground truth on test cases.

- **Third-Party Dependency Breakage (MEDIUM)**: CAMEL-AI or concept-erasure repos may have bugs/compatibility issues. Mitigation: CAMEL-AI in dev mode (can patch); fork concept-erasure repos if needed; pin all dependency versions.

- **Attack Trace Interpretability (LOW)**: Generated traces may be too technical or verbose for target users. Mitigation: Test with one external user; iterate on format during Week 3.

### Open Questions

- What specific unlearning methods should we target for MVP? (Need to select 1-2 per task type)
- Which papers should be included in the RAG corpus? (Need literature review for each task)
- What are the exact evaluation metric thresholds for "vulnerability discovered"? (Forget accuracy delta? Concept leakage probability?)
- How should we define judge personas? (Security expert, ML researcher, privacy advocate, skeptical reviewer, industry practitioner?)
- What format should attack traces use? (Markdown report? JSON log? Both?)
- Should we implement attack trace visualization (e.g., hypothesis tree diagram)?
- How do we validate that discovered vulnerabilities are "novel" vs "known"?
- What baseline should we compare against to show adaptive > static evaluation?
- Should we test on public datasets only, or can we use synthetic data?
- How granular should the academic report be? (Enough detail for reproduction vs. readability?)

### Areas Needing Further Research

- **Concept-Erasure Method Selection**: Survey recent papers (EraseDiff, Concept Ablation, etc.) to identify best candidates for integration
- **Judge Persona Design**: Research effective multi-perspective evaluation frameworks in ML security literature
- **Attack Pattern Taxonomy**: Review privacy attack literature (membership inference, model inversion, data extraction) to inform seed templates
- **Evaluation Metrics**: Identify standard metrics for both data-based unlearning (forget accuracy, membership inference AUC) and concept-erasure (CLIP score, FID, human evaluation proxies)
- **Reproducibility Standards**: Investigate what information is needed in attack traces for independent reproduction

## Next Steps

### Immediate Actions

1. **Week 1 Preparation (Before Oct 24)**:
   - Set up CAMEL-AI in dev mode and test basic agent functionality
   - Add DeepUnlearn as git submodule and verify installation
   - Identify 1-2 target unlearning methods per task (data-based + concept-erasure)
   - Collect initial paper corpus (10-20 papers per task)
   - Create seed hypothesis templates from literature (3-5 attack patterns)

2. **Week 1 Execution (Oct 24-30)**:
   - Implement MVP inner loop for data-based unlearning only
   - Test DeepUnlearn integration as MCP FunctionTool
   - Run first vulnerability discovery attempt (3-5 manual iterations)

3. **Week 2 Execution (Oct 31-Nov 6)**:
   - Add RAG system and integrate with hypothesis generator
   - Implement concept-erasure task with VLM evaluation
   - Run multiple loops on both tasks to discover vulnerabilities

4. **Week 3 Execution (Nov 7-13)**:
   - Implement reporter agent and judge system
   - Run end-to-end on both tasks, collect results
   - Generate attack traces and reports

5. **Week 4 Execution (Nov 14+)**:
   - Write academic paper using AUST-generated reports as foundation
   - Integrate with thesis

---

## Summary

This Project Brief documents the complete vision, scope, and implementation plan for AUST (AI Scientist for Autonomous Unlearning Security Testing). The system will demonstrate autonomous end-to-end scientific research—from hypothesis generation through experimentation to academic reporting and multi-perspective judging—on the societally relevant problem of machine unlearning vulnerabilities.

**Core Deliverables:**
- Autonomous research system (inner + outer loops)
- Attack traces showing step-by-step vulnerability discovery
- Multi-perspective judge evaluations
- At least 1 exploitable vulnerability per task (data-based + concept-erasure)
- Academic paper ready by Nov 14th

**Success Depends On:**
- Aggressive prioritization (P0 features only)
- Pragmatic MVP approach (basic RAG, single critic-generator)
- Risk mitigation (seed templates, progressive difficulty, dev mode dependencies)
- Timeline discipline (3-week sprint, no major pivots after Nov 7)

With the brainstorming session results and this project brief, you now have comprehensive documentation to guide AUST development from concept to completion.

