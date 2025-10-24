# Brainstorming Session Results

**Session Date:** 2025-10-16
**Facilitator:** Business Analyst Mary 📊
**Participant:** AUST Research Team

---

## Executive Summary

**Topic:** AUST - AI Scientist for Autonomous Unlearning Security Testing

**Session Goals:** Design the full system architecture for an autonomous AI scientist that discovers and explains vulnerabilities in machine unlearning methods (data-based unlearning and concept-based erasure) through an adaptive, hypothesis-driven research loop.

**Techniques Used:**
- First Principles Thinking (20 min)
- Morphological Analysis (15 min)
- SCAMPER Method - Combine (10 min)
- Yes, And... Building (10 min)
- Mind Mapping (15 min)
- Prioritization & Timeline Planning (20 min)
- Convergent Synthesis (10 min)

**Total Ideas Generated:** 50+ architectural decisions, components, and implementation strategies

### Key Themes Identified:
- **Adaptive Research Loop**: Inner loop (hypothesis → experiment → evaluation → feedback) that iterates until vulnerability discovered, then outer loop (reporting → judging)
- **Multi-Agent Architecture**: Orchestrator managing specialized agents (Hypothesis Generator, Critic, Query Generator, Evaluator, Reporter, Judges)
- **Knowledge Integration**: RAG system with paper retrieval + long-term memory of successful experiments
- **Pragmatic MVP Strategy**: Start with single critic-generator pair, scale to orchestrator-worker model later
- **Aggressive 3-Week Timeline**: Foundation (Week 1) → Enhancement (Week 2) → Automation (Week 3) → Paper Writing (Week 4)
- **Dual Task Unification**: Single workflow with prompt-based differentiation for data-based vs concept-erasure unlearning

---

## Technique Sessions

### First Principles Thinking - 20 min
**Description:** Breaking down AUST into fundamental, irreducible components to understand what makes the system truly autonomous and scientific.

#### Ideas Generated:
1. **Autonomy via Feedback Loops**: Agent interaction + evaluation metrics + VLM-based vision evaluation enables self-directed research
2. **Scientific Rigor via Hypothesis Generation**: Novel attack/stress test method discovery through paper retrieval, ensemble methods, and learning from past experiments
3. **Vulnerability Discovery via Iteration**: Breaking unlearned models through repeated hypothesis-test cycles until leakage detected
4. **Critical Agent Debate**: Force hypothesis generator to produce more novel ideas through adversarial questioning
5. **Parallel Hypothesis Generation**: After debate session, generate multiple hypotheses asynchronously
6. **Multi-Modal Learning Sources**: Paper reading, ensemble methods, hyperparameter tuning, conversation memory, success records
7. **Task-Specific Success Criteria**:
   - Data-based: Forget accuracy threshold changes
   - Concept-erasure: Generation-based leakage or metric-based resurgence

#### Insights Discovered:
- The "autonomy" emerges from the feedback loop, not just automation
- Scientific rigor requires both generation (creativity) and critique (rigor)
- Step-by-step approach (debate first, parallelize later) balances quality and efficiency

#### Notable Connections:
- Debate mechanism similar to peer review in human scientific process
- Memory system acts like "lab notebook" for the AI scientist
- VLM evaluation bridges qualitative (visual) and quantitative (metrics) assessment

---

### Morphological Analysis - 15 min
**Description:** Systematically exploring parameter space and design dimensions for AUST system components.

#### Ideas Generated:
1. **Agent Architecture Progression**: MVP uses single critic-generator pair → scale to orchestrator + workers later
2. **Hypothesis Source Strategy**: Hybrid ensemble combining paper-retrieval + memory-based methods (MUST HAVE)
3. **Experiment Execution**: Start sequential, leverage CAMEL-AI interpreters for execution
4. **Evaluation Strategy**: Multi-modal combination (metrics + VLM) required for both tasks
5. **Workflow Unification**: Two tasks (data-based vs concept-erasure) share single pipeline with prompt-based differentiation
6. **Paper Corpus Design**:
   - Task-specific collections (data-based papers, concept-erasure papers)
   - Shared "attack methods" corpus for cross-pollination
7. **Query Generator Agent**: Converts feedback (initial guesses, experiment results, useful papers) into RAG queries
8. **Method Extraction Focus**: Parse papers for Methods sections to implement/extrapolate new hypotheses
9. **Memory Solution**: Use CAMEL-AI's long-term memory storage (RAG-like)

#### Insights Discovered:
- Unified workflow is more maintainable than separate pipelines
- Paper corpus structure enables both specialization and creative cross-pollination
- Query Generator as separate agent makes retrieval adaptive rather than static

#### Notable Connections:
- MVP → Ideal progression mirrors scientific research (prototype → production)
- Cross-pollination between task-specific corpora mimics interdisciplinary research

---

### SCAMPER Method (Combine) - 10 min
**Description:** Exploring what can be combined or integrated to create better components, focusing on workflow overlaps.

#### Ideas Generated:
1. **Unified Prompt-Based Task Differentiation**: Same agents handle both tasks via specialized prompting
2. **Shared Core with Domain-Specific Modules**: Architecture approach B/C (configurable branches or plugins)
3. **Academic Paper Format for Reporting**: Introduction → Method → Experiments → Results → Discussion → Conclusion
4. **Single Critic Agent for MVP**: Can expand to multiple specialized critics later
5. **Shared vs Specialized Paper Collections**:
   - Data-based unlearning corpus (SISA, membership inference, benchmarks)
   - Concept-erasure corpus (EraseDiff, Stable Diffusion editing, concept ablation)
   - Shared attack methods corpus (adversarial examples, privacy attacks, red-teaming)

#### Insights Discovered:
- Prompt engineering is more flexible than code duplication for task differentiation
- Report format matching academic papers makes output immediately useful
- Different paper collections with shared attack corpus balances specialization and generalization

#### Notable Connections:
- Academic format report enables direct integration with human paper writing
- Shared attack corpus enables transfer learning between unlearning domains

---

### Yes, And... Building - 10 min
**Description:** Collaborative idea expansion through building on each other's concepts.

#### Ideas Generated:
1. **Query Generator Intelligence**: Takes feedback from evaluator + hypothesis generator needs + can cross-reference other methods
2. **Method Extraction Strategy**: Focus on Methods sections from papers to implement or extrapolate ideas
3. **Adaptive RAG Queries**: System learns what queries produce useful results over time
4. **Paper Processing Pipeline**: Raw paper → Section extraction → Structured cards (Method Cards + Experiment Cards)
5. **Structured Card Schema**:
   - Method Cards: Principles, Assumptions, Variables/Hyperparameters, Original text anchors
   - Experiment Cards: Dataset/Task, Controls/Ablations, Metrics, Threat Model, Findings, Figure summaries, Text anchors
6. **Dual Storage Format**: JSON for indexing/search + Markdown for human debugging
7. **Multimodal Paper Understanding**: Use GPT-5/Claude 3.5 to read text + figures natively
8. **Section Whitelisting/Blacklisting**:
   - Whitelist: Abstract, Intro (task/threat only), Method, Experiments, Results, Ablation, Limitations
   - Blacklist: Related Work (2-3 sentences only), Appendix (abstract only)
9. **One Card Per Paper**: Multiple method sections within single card

#### Insights Discovered:
- Paper processing quality directly impacts hypothesis quality
- Structured extraction prevents noise from captions, tables, appendices
- Dual format (JSON + MD) balances machine efficiency and human interpretability

#### Notable Connections:
- Structured cards act like "research protocols" that can be replicated
- Figure purpose summaries preserve visual information without pixel-level storage

---

### Mind Mapping - Agent Architecture - 15 min
**Description:** Visually mapping the agent ecosystem and their interactions, particularly the critical inner research loop.

#### Ideas Generated:
1. **Inner Research Loop Discovery**: Evaluator → Query Generator → RAG → Hypothesis Generator → Experiment → Evaluator (repeat)
2. **Loop Exit Conditions**:
   - Vulnerability found (success) → exit to Reporter
   - Max iterations reached → exit to Reporter
3. **Critic Timing**: Engages after first iteration with feedback + hypothesis + experiment results
4. **Query Generator Dual Input**: Receives both Evaluator feedback AND Hypothesis Generator requests
5. **Experience Compaction**: Successful vulnerability discoveries → compacted → long-term memory for future reference
6. **Decision Point Architecture**: Orchestrator decides whether to continue inner loop or exit to reporting
7. **Judge Personas**: Multiple LLM perspectives evaluate reports from different angles (novelty, rigor, impact, reproducibility)
8. **DeepUnlearn Integration**: Integrated as MCP FunctionTool within CAMEL-AI framework
9. **Concept-Erasure Tool Integration**: GitHub code converted to MCP FunctionTool
10. **Multi-Modal Evaluation Pipeline**: Metrics (threshold-based) + VLM (vision-based) combined for robust assessment

#### Insights Discovered:
- Inner vs outer loop separation clarifies when system is "researching" vs "reporting"
- Query Generator as bridge between evaluation and knowledge retrieval makes system adaptive
- Experience compaction creates institutional memory for the AI scientist

#### Notable Connections:
- Inner loop mirrors hypothesis-experiment cycle in human research
- Outer loop (reporting + judging) mirrors publication + peer review
- Memory system enables cumulative learning like human expertise development

---

### Prioritization & Timeline Planning - 20 min
**Description:** Identifying critical path and creating aggressive 3-week implementation timeline for Nov 14th deadline.

#### Ideas Generated:

**P0 - MUST HAVE (Core Contribution):**
1. Inner research loop (Hypothesis → Experiment → Evaluation → Feedback)
2. Critic agent challenging hypothesis generator
3. Multi-modal evaluation (metrics + VLM)
4. Automated reporting in academic format
5. LLM judges with multiple personas
6. DeepUnlearn integration as MCP tool
7. Query Generator + RAG system (basic but functional)
8. At least ONE vulnerability discovery per task (data-based + concept-erasure)

**P1 - SHOULD HAVE (Strengthens Story):**
1. Memory system for successful experiments
2. Parallel hypothesis generation (after debate)
3. Comparison against static benchmarks
4. Multiple vulnerability discoveries

**P2 - NICE TO HAVE (Polish):**
1. Sophisticated paper indexing (structured cards)
2. Orchestrator → worker architecture
3. Extensive ablation studies

**Week 1 Timeline (Oct 24-30): Foundation + MVP Loop**
- Days 1-2: CAMEL-AI setup + DeepUnlearn integration + GPU sandbox
- Days 3-5: MVP inner loop (single task: data-based unlearning only)
- Days 6-7: First vulnerability hunt (3-5 manual loop runs)

**Week 2 Timeline (Oct 31-Nov 6): Enhancement + Second Task**
- Days 8-9: Basic RAG system (10-20 papers, simple semantic search)
- Days 10-11: Memory integration + loop exit conditions (5-10 full loops on data-based)
- Days 12-14: Concept-erasure task integration + VLM evaluation (5-10 loops)

**Week 3 Timeline (Nov 7-13): Automation + Judging + Polish**
- Days 15-16: Reporter agent (academic paper template + citation integration)
- Days 17-18: Judge system (3-5 personas + evaluation criteria)
- Days 19-20: End-to-end runs on both tasks + optional ablations
- Day 21: Buffer / emergency fixes

**Week 4 Timeline (Nov 14+):**
- Days 22-28: Paper writing + thesis integration

**Risk Mitigation Strategies:**
1. **Seed Hypothesis Generator**: Pre-load 3-5 known attack patterns as templates
2. **Minimal Human-in-the-Loop**: One round of feedback after first 2-3 loops
3. **Progressive Difficulty**: Start with fragile unlearning methods, move to harder targets

#### Insights Discovered:
- Query + RAG promoted from P1 to P0 - essential for hypothesis quality
- Biggest risks: implementation time + hypothesis generation quality
- Paper indexing demoted to P2 - infrastructure, not core contribution
- 3-week sprint is aggressive but achievable with clear priorities

#### Notable Connections:
- Week 1 focuses on vertical slice (one complete loop)
- Week 2 adds breadth (second task + enhancement)
- Week 3 focuses on automation and evaluation (judge system)
- Risk mitigation strategies provide fallbacks without compromising core goals

---

## Idea Categorization

### Immediate Opportunities
*Ideas ready to implement now*

1. **MVP Inner Loop with Data-Based Unlearning**
   - Description: Build single complete research loop using DeepUnlearn benchmark for data-based unlearning only
   - Why immediate: Vertical slice proves core concept; DeepUnlearn provides ready-to-use infrastructure
   - Resources needed: CAMEL-AI setup, GPU access, DeepUnlearn repo, basic prompting for hypothesis/critic agents

2. **Basic RAG with Manual Paper Collection**
   - Description: 10-20 key papers per task with simple semantic search (no structured cards initially)
   - Why immediate: Good enough for MVP hypothesis generation; can upgrade later
   - Resources needed: Paper PDFs/URLs, embedding model, vector database (FAISS or similar)

3. **Threshold-Based Evaluation for Data-Based Task**
   - Description: Forget accuracy monitoring with configurable thresholds to detect unlearning failure
   - Why immediate: Clear, objective metric; well-defined in literature
   - Resources needed: DeepUnlearn evaluation functions, threshold configuration

4. **Academic Paper Template for Reporter**
   - Description: Structured template (Intro/Method/Exp/Results/Discussion/Conclusion) with variable slots
   - Why immediate: Template is straightforward; enables end-to-end testing quickly
   - Resources needed: Markdown template, citation format specification

5. **Seed Hypothesis Templates**
   - Description: Pre-load 3-5 known attack patterns (membership inference, model inversion, data extraction)
   - Why immediate: Reduces cold-start problem; provides baseline for critic to improve upon
   - Resources needed: Literature review for attack templates, prompt engineering

---

### Future Innovations
*Ideas requiring development/research*

1. **LLM-Based Paper Indexing with Structured Cards**
   - Description: Multimodal LLM (GPT-5/Claude 3.5) extracts Method + Experiment cards in JSON/MD format
   - Development needed: Card schema design, extraction prompts, quality validation pipeline
   - Timeline estimate: Week 2-3 or post-paper

2. **Orchestrator + Worker Architecture**
   - Description: Central orchestrator manages multiple specialized worker agents (hypothesis workers, evaluation workers, etc.)
   - Development needed: CAMEL-AI workforce API integration, task distribution logic, result aggregation
   - Timeline estimate: Post-MVP, if scaling is needed

3. **Adaptive Query Learning**
   - Description: Query Generator learns which query formulations produce useful papers over time
   - Development needed: Query effectiveness scoring, query refinement model, feedback incorporation
   - Timeline estimate: Post-paper, research contribution itself

4. **Progressive Difficulty Curriculum**
   - Description: Start with known-fragile unlearning methods, progressively test harder targets as system improves
   - Development needed: Difficulty taxonomy, success rate tracking, automatic progression logic
   - Timeline estimate: Week 2-3, if time permits

5. **Extensive Ablation Studies**
   - Description: With/without critic, with/without RAG, with/without memory, different judge configurations
   - Development needed: Experimental harness for controlled comparisons, statistical analysis
   - Timeline estimate: Week 3 or post-paper

---

### Moonshots
*Ambitious, transformative concepts*

1. **Fully Autonomous Research Loop**
   - Description: System runs continuously, discovers multiple vulnerabilities, writes papers, submits to conferences autonomously
   - Transformative potential: Demonstrates AI conducting end-to-end scientific research without human intervention
   - Challenges to overcome: Long-term stability, resource management, publication-quality writing, ethical review integration

2. **Cross-Domain Vulnerability Transfer**
   - Description: System discovers that attacks from data-based unlearning work on concept-erasure (or vice versa)
   - Transformative potential: Reveals fundamental weaknesses in unlearning paradigm; produces novel attack classes
   - Challenges to overcome: Sufficient diversity in hypothesis generation, abstraction capability, validation rigor

3. **Self-Improving Hypothesis Generator**
   - Description: Meta-learning from successful vs failed hypotheses to improve generation strategy over time
   - Transformative potential: System becomes increasingly effective scientist; demonstrates AI capability growth
   - Challenges to overcome: Meta-learning framework, sufficient data for learning, avoiding overfitting to specific vulnerabilities

4. **Multi-Modal Evidence Synthesis**
   - Description: System combines quantitative metrics, qualitative VLM observations, and literature citations into coherent arguments
   - Transformative potential: Produces human-competitive scientific reasoning and argumentation
   - Challenges to overcome: Evidence weighing, coherence across modalities, argumentation structure

---

### Insights & Learnings
*Key realizations from the session*

- **Autonomy Emerges from Feedback, Not Just Automation**: The inner research loop's ability to interpret results and adapt hypotheses is what makes AUST truly autonomous, not merely executing predefined tests.

- **Two-Loop Architecture Mirrors Human Research Process**: Inner loop (hypothesis-experiment iteration) mirrors lab work; outer loop (reporting-judging) mirrors publication and peer review - natural separation of concerns.

- **Query Generator as Adaptive Bridge**: Placing a dedicated Query Generator between evaluation feedback and RAG retrieval makes knowledge integration adaptive rather than static, critical for hypothesis quality.

- **Prompt-Based Task Unification Beats Code Duplication**: Using prompts to differentiate data-based vs concept-erasure tasks provides flexibility and maintainability vs building separate pipelines.

- **Paper Indexing is Infrastructure, Not Contribution**: While structured card extraction would improve retrieval, it's not the core scientific contribution - don't over-invest in infrastructure.

- **MVP Strategy Enables Rapid Validation**: Single critic-generator pair with basic RAG proves core concept quickly; can scale to orchestrator-workers later if needed.

- **Debate Mechanism Provides Scientific Rigor**: Critic agent challenging hypothesis generator after first iteration mirrors peer review and forces higher-quality hypothesis generation.

- **Memory Compaction Creates Institutional Knowledge**: Successfully discovered vulnerabilities stored in long-term memory enable cumulative learning, like human expertise development.

- **Multi-Modal Evaluation Bridges Qualitative and Quantitative**: Combining metrics (objective thresholds) with VLM analysis (subjective visual assessment) provides robust vulnerability detection.

- **Timeline Risk Requires Strategic De-Scoping**: With aggressive 3-week deadline, clear P0/P1/P2 prioritization and risk mitigation strategies (seed templates, progressive difficulty) are essential.

---

## Action Planning

### Top 3 Priority Ideas

#### #1 Priority: Build MVP Inner Research Loop (Week 1)
- **Rationale**: Proves core concept end-to-end; provides foundation for all other features; fastest path to first vulnerability discovery
- **Next steps**:
  1. Set up CAMEL-AI environment + DeepUnlearn integration as MCP FunctionTool
  2. Implement basic Hypothesis Generator + Critic Agent (simple prompting)
  3. Implement Experiment Executor (calls DeepUnlearn tools)
  4. Implement Evaluator (threshold-based forget accuracy)
  5. Wire loop together and test with manual iterations
- **Resources needed**:
  - CAMEL-AI documentation + examples
  - DeepUnlearn repository + API
  - GPU sandbox access (for experiment execution)
  - 3-5 seed hypothesis templates from literature
- **Timeline**: Days 1-7 (Week 1 complete)

#### #2 Priority: Integrate Query Generator + Basic RAG (Week 2, Days 8-9)
- **Rationale**: Essential for hypothesis quality (promoted to P0); addresses primary risk of weak hypothesis generation
- **Next steps**:
  1. Collect 10-20 key papers per task (data-based unlearning, concept-erasure, attack methods)
  2. Set up simple vector database (FAISS or similar) with embedding model
  3. Implement Query Generator agent (converts feedback + needs → search queries)
  4. Integrate RAG results into Hypothesis Generator context
  5. Test that hypotheses now reference papers/methods
- **Resources needed**:
  - Paper corpus (PDFs or URLs)
  - Embedding model (OpenAI, Sentence-Transformers, or similar)
  - Vector database library
  - Query Generator prompts
- **Timeline**: Days 8-9 (2 days)

#### #3 Priority: Add Concept-Erasure Task + Multi-Modal Evaluation (Week 2, Days 12-14)
- **Rationale**: Second task required for paper's core claim (both data-based AND concept-erasure); demonstrates generality
- **Next steps**:
  1. Integrate concept-erasure unlearning methods from GitHub as MCP FunctionTool
  2. Add VLM-based evaluation (generation-based leakage detection)
  3. Adapt prompts for concept-erasure domain
  4. Run 5-10 loops to discover at least one vulnerability
  5. Validate multi-modal evaluation (metrics + VLM) working correctly
- **Resources needed**:
  - Concept-erasure method repositories (EraseDiff, etc.)
  - VLM API access (GPT-4V, Claude 3.5, or CAMEL-AI's vision interface)
  - Concept-erasure paper corpus (for RAG)
  - Evaluation metric thresholds for concept leakage
- **Timeline**: Days 12-14 (3 days)

---

## Reflection & Follow-up

### What Worked Well
- **First Principles breakdown**: Quickly identified the three core fundamentals (autonomy, scientific rigor, vulnerability discovery) that anchor the entire system
- **Morphological Analysis**: Systematically explored parameter space and avoided missing critical dimensions
- **Mind Mapping**: Revealed the inner vs outer loop distinction, which clarified the architecture significantly
- **Prioritization exercise**: P0/P1/P2 tiers and 3-week timeline forced concrete decision-making and de-scoping
- **Iterative refinement**: Building on each technique's output led to progressively more concrete and actionable design

### Areas for Further Exploration
- **Hypothesis quality assurance**: What specific criteria define a "good" hypothesis? How does critic agent evaluate quality beyond novelty?
- **Loop exit criteria refinement**: Beyond "vulnerability found" or "max iterations", are there intermediate stopping conditions (e.g., diminishing returns, hypothesis space exhausted)?
- **Judge persona design**: What specific roles/perspectives should the LLM judges take? (e.g., security expert, ML researcher, privacy advocate, skeptical reviewer)
- **Evaluation metric thresholds**: What are the specific thresholds for forget accuracy (data-based) and concept leakage (concept-erasure)?
- **Failure mode handling**: What happens when experiments crash, tools fail, or GPU resources are unavailable?
- **Experiment reproducibility**: How to ensure generated experiments are reproducible by humans?
- **Dataset selection strategy**: Which specific datasets for data-based unlearning? Which generative models for concept-erasure?
- **Human role documentation**: How to clearly attribute human contributions vs AI contributions in reports?

### Recommended Follow-up Techniques
- **Five Whys**: Dig deeper into "why might hypothesis generation fail?" to proactively address quality issues
- **Assumption Reversal**: Challenge assumption that single critic is sufficient - what if we need multiple specialized critics?
- **Role Playing**: Brainstorm from different stakeholder perspectives (security researcher, ML practitioner, policy maker) to refine judge personas
- **Provocation Technique**: "What if AUST finds NO vulnerabilities?" - plan for null result scenario
- **Time Shifting**: "How would you demo this in 1 week with minimal functionality?" - emergency MVP plan

### Questions That Emerged
- How do we measure "novelty" of discovered vulnerabilities objectively?
- What constitutes a "successful" vulnerability discovery (severity threshold)?
- Should hypothesis generator be allowed to propose unimplementable experiments (then critic rejects)?
- How to balance exploration (diverse hypotheses) vs exploitation (refining promising leads)?
- Should the system prioritize finding many weak vulnerabilities or one strong vulnerability?
- How to prevent the system from rediscovering the same vulnerability repeatedly?
- What level of detail should auto-generated reports include (enough for reproducibility vs readable)?
- How to handle conflicting judge opinions (e.g., one judge says novel, another says incremental)?

### Next Session Planning
- **Suggested topics**:
  1. Deep dive into Hypothesis Generator prompting and seed templates
  2. Judge persona design and evaluation criteria specification
  3. Failure mode handling and error recovery strategies
  4. Paper corpus curation and quality assessment
- **Recommended timeframe**: After Week 1 MVP is functional (around Oct 30-31) - reconvene to refine based on real system behavior
- **Preparation needed**:
  - Have Week 1 MVP running (even if buggy)
  - Collect examples of good/bad hypotheses from initial runs
  - Document failure modes encountered
  - Identify specific pain points in implementation

---

*Session facilitated using the BMAD-METHOD™ brainstorming framework*
