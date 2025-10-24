# CAMEL-AI Resources Index for Story 1.5

**All files located in:** `/data/users/yyx/onProject/CAUST/`

## Quick Navigation

### For First-Time Users
1. Start with `README_CAMEL_RESOURCES.md` (9.1KB) - Overview
2. Then read `CAMEL_QUICK_REFERENCE.md` (6.5KB) - Fast lookup
3. Deep dive with `CAMEL_PATTERNS_GUIDE.md` (24KB) - Technical details

### For Implementation
1. Follow `STORY_1_5_IMPLEMENTATION_TEMPLATE.md` (18KB) - Working code
2. Reference `CAMEL_QUICK_REFERENCE.md` while coding
3. Use `CAMEL_PATTERNS_GUIDE.md` for technical clarification

---

## Document Details

### 1. README_CAMEL_RESOURCES.md
**Purpose:** Master index and orientation guide
**Size:** 9.1KB
**Contents:**
- Overview of all three patterns
- Document descriptions
- Source code references
- Quick start guide
- Pattern comparison matrix
- Troubleshooting guide
- File structure
**When to use:** First resource to understand what you have

### 2. CAMEL_QUICK_REFERENCE.md
**Purpose:** Fast lookup while coding
**Size:** 6.5KB
**Contents:**
- Import statements
- Constructor signatures
- Key methods and parameters
- Common patterns and configurations
- File locations
- Implementation checklist
- Debugging tips
**When to use:** While actively developing - quick syntax reference

### 3. CAMEL_PATTERNS_GUIDE.md
**Purpose:** Comprehensive technical reference
**Size:** 24KB
**Contents:**
- CriticAgent detailed explanation
- RolePlaying integration
- Tree search workflows
- SelfImprovingCoTPipeline details
- Task decomposition/composition
- Data structures
- Code examples with explanations
- Integration strategies
**When to use:** Understanding technical details and architecture

### 4. STORY_1_5_IMPLEMENTATION_TEMPLATE.md
**Purpose:** Ready-to-use implementation code
**Size:** 18KB
**Contents:**
- Configuration setup
- Phase 1: Task setup with hierarchies
- Phase 2: Multi-agent reasoning with CriticAgent
- Phase 3: Self-improving CoT pipeline
- Phase 4: Finalization and results
- Monitoring and debugging examples
- Output file handling
- Testing checklist
**When to use:** Ready to implement - follow the phases sequentially

---

## The Three CAMEL Patterns

### Pattern 1: CriticAgent & Tree Search
- Generates multiple proposals from multiple agents
- CriticAgent selects best proposal
- Enables collaborative multi-path reasoning
- Implementation: RolePlaying + CriticAgent

### Pattern 2: Self-Improving CoT Generation
- Generates initial reasoning trace
- Evaluates quality (correctness, clarity, completeness)
- Iteratively refines based on feedback
- Stops when threshold met or max iterations reached
- Implementation: SelfImprovingCoTPipeline

### Pattern 3: Task Module
- Hierarchical task management
- Task decomposition and composition
- State tracking (OPEN -> RUNNING -> DONE)
- Workflow orchestration
- Implementation: Task + TaskManager

---

## Source Code Reference

**Implementation Files:**
- `/external/camel/camel/agents/critic_agent.py` - CriticAgent implementation
- `/external/camel/camel/agents/chat_agent.py` - Base ChatAgent
- `/external/camel/camel/societies/role_playing.py` - RolePlaying orchestration
- `/external/camel/camel/tasks/task.py` - Task and TaskManager
- `/external/camel/camel/datagen/self_improving_cot.py` - SelfImprovingCoTPipeline

**Documentation:**
- `/external/camel/docs/key_modules/tasks.md` - Task documentation
- `/external/camel/docs/cookbooks/advanced_features/critic_agents_and_tree_search.ipynb`
- `/external/camel/docs/cookbooks/data_generation/self_improving_cot_generation.md`

**Examples:**
- `/external/camel/examples/tasks/task_generation.py`
- `/external/camel/docs/cookbooks/multi_agent_society/task_generation.ipynb`

---

## Critical Configuration Points

### CriticAgent
```python
# MUST have n > 1 for critic to work
ChatGPTConfig(n=3, temperature=0.8)

# MUST enable critic in loop
RolePlaying(with_critic_in_the_loop=True, ...)
```

### Self-Improving CoT
```python
# Quality threshold target (0-1)
score_threshold=0.80

# Maximum refinement iterations
max_iterations=3

# Evaluation dimensions
correctness, clarity, completeness
```

### Task Module
```python
# Create hierarchical structure
parent.add_subtask(child)

# Decompose complex tasks
subtasks = task.decompose(agent=agent)

# Manage execution
TaskManager.set_tasks_dependence(root, others, type="serial")
```

---

## Implementation Phases

1. **Phase 1: Task Setup** - Define hierarchical task structure
2. **Phase 2: Tree Search** - Multi-agent reasoning with critic
3. **Phase 3: Self-Improving CoT** - Iterative refinement
4. **Phase 4: Finalization** - Results compilation

Total estimated implementation time: 1-2 hours

---

## Quick Start (5 Minutes)

```bash
# 1. Read overview
less README_CAMEL_RESOURCES.md

# 2. Check syntax quickly
less CAMEL_QUICK_REFERENCE.md

# 3. Start implementation
less STORY_1_5_IMPLEMENTATION_TEMPLATE.md
```

---

## Recommended Reading Order

**Understanding Only:**
1. README_CAMEL_RESOURCES.md
2. CAMEL_QUICK_REFERENCE.md (sections only)
3. CAMEL_PATTERNS_GUIDE.md (sections of interest)

**Implementation:**
1. README_CAMEL_RESOURCES.md (skim)
2. STORY_1_5_IMPLEMENTATION_TEMPLATE.md (read fully)
3. CAMEL_PATTERNS_GUIDE.md (as needed for clarification)
4. CAMEL_QUICK_REFERENCE.md (while coding)

**Reference While Coding:**
- Keep CAMEL_QUICK_REFERENCE.md open
- Refer to CAMEL_PATTERNS_GUIDE.md for technical questions
- Follow STORY_1_5_IMPLEMENTATION_TEMPLATE.md structure

---

## File Statistics

- Total documentation: ~57KB
- Number of documents: 4
- Code examples: 50+
- Diagrams/workflows: 5+
- Configuration examples: 20+
- Troubleshooting entries: 10+

---

## Support Resources

**In Repository:**
- GitHub: https://github.com/camel-ai/camel
- Discord: https://discord.camel-ai.org
- X/Twitter: https://x.com/camelaiorg

**In Codebase:**
- API Reference: `/external/camel/docs/reference/`
- Cookbooks: `/external/camel/docs/cookbooks/`
- Examples: `/external/camel/examples/`

---

## Version Information

- Created: October 2024
- CAMEL Version: 0.2.16+
- Python: 3.9+
- Key Dependencies: camel-ai, openai, pydantic

---

## Summary

You have everything needed to implement Story 1.5:

✓ 4 complementary documents covering 3 CAMEL patterns
✓ Quick reference guide for development
✓ Complete working code template
✓ Comprehensive technical documentation
✓ 50+ code examples
✓ Troubleshooting guide
✓ Source code references

**Next Step:** Read `README_CAMEL_RESOURCES.md` to get oriented.

