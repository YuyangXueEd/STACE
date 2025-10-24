# CAMEL-AI Framework Resources for Story 1.5

## Overview

This directory contains comprehensive documentation and implementation guides for using CAMEL-AI's advanced patterns to implement Story 1.5. The focus is on three core patterns:

1. **CriticAgent & Tree Search** - Multi-agent collaborative reasoning with critic feedback
2. **Self-Improving CoT Generation** - Iterative refinement of reasoning traces
3. **Task Module** - Hierarchical task management and orchestration

---

## Documents Created

### 1. CAMEL_PATTERNS_GUIDE.md
**Complete technical reference (24KB)**

Comprehensive documentation covering:
- CriticAgent architecture and methods
- System message generation for CriticAgent
- RolePlaying integration with CriticAgent
- Tree search workflow patterns
- SelfImprovingCoTPipeline components and usage
- Task decomposition, composition, and evolution
- Data structures and evaluation types
- Integration strategies

**Use when:** You need detailed technical understanding of each pattern.

### 2. CAMEL_QUICK_REFERENCE.md
**Quick lookup guide (6.5KB)**

Fast reference with:
- Import statements
- Constructor signatures with parameters
- Key methods and their usage
- Configuration options
- Common patterns
- File locations
- Implementation checklist
- Debugging tips

**Use when:** You need to quickly look up syntax or configuration options.

### 3. STORY_1_5_IMPLEMENTATION_TEMPLATE.md
**Complete implementation example (6.5KB)**

Full working code template with:
- Phase 1: Setup and configuration
- Phase 2: Task setup with hierarchies
- Phase 3: Multi-agent reasoning with CriticAgent
- Phase 4: Self-improving CoT pipeline
- Phase 5: Finalization and results
- Monitoring and debugging examples
- Output files handling
- Testing checklist

**Use when:** You're ready to implement Story 1.5 and want a proven template.

---

## Source Code References

All source code is in `/data/users/yyx/onProject/CAUST/external/camel/`

### Core Implementation Files

| Component | Location |
|-----------|----------|
| CriticAgent | `/camel/agents/critic_agent.py` |
| ChatAgent | `/camel/agents/chat_agent.py` |
| RolePlaying | `/camel/societies/role_playing.py` |
| Task | `/camel/tasks/task.py` |
| TaskManager | `/camel/tasks/task.py` |
| SelfImprovingCoTPipeline | `/camel/datagen/self_improving_cot.py` |

### Documentation & Examples

| Resource | Location |
|----------|----------|
| Tasks Documentation | `/docs/key_modules/tasks.md` |
| Critic & Tree Search Cookbook | `/docs/cookbooks/advanced_features/critic_agents_and_tree_search.ipynb` |
| Self-Improving CoT Guide | `/docs/cookbooks/data_generation/self_improving_cot_generation.md` |
| Task Generation Example | `/examples/tasks/task_generation.py` |
| Task Generation Notebook | `/docs/cookbooks/multi_agent_society/task_generation.ipynb` |

---

## Quick Start Guide

### For Story 1.5 Implementation:

**Step 1: Understand the Patterns**
- Start with CAMEL_QUICK_REFERENCE.md for overview
- Read CAMEL_PATTERNS_GUIDE.md for technical details

**Step 2: Review Implementation Template**
- Study STORY_1_5_IMPLEMENTATION_TEMPLATE.md structure
- Note the 4 phases of execution

**Step 3: Implement Phase by Phase**
- Phase 1: Task setup with TaskManager
- Phase 2: RolePlaying with CriticAgent for tree search
- Phase 3: SelfImprovingCoTPipeline for refinement
- Phase 4: Results compilation and verification

**Step 4: Key Configuration Points**
```python
# CRITICAL: Enable critic selection
ChatGPTConfig(n=3, temperature=0.8)  # n must be > 1

# CRITICAL: Enable critic in loop
RolePlaying(with_critic_in_the_loop=True, ...)

# CRITICAL: Set quality threshold
SelfImprovingCoTPipeline(score_threshold=0.80, ...)
```

---

## Pattern Comparison

### When to Use Each Pattern

**CriticAgent & Tree Search:**
- Multiple solution proposals to evaluate
- Need to explore different reasoning paths
- Want collaborative selection from multiple options
- Have diverse expert agents providing different perspectives

**Self-Improving CoT:**
- Generated solutions need quality refinement
- Want iterative improvement with feedback
- Require tracking of improvement history
- Need to meet quality thresholds

**Task Module:**
- Complex workflows with multiple steps
- Need hierarchical task decomposition
- Require explicit state tracking
- Want to compose results from subtasks

---

## Key Implementation Insights

### 1. CriticAgent Essentials
```python
# Three critical pieces:
# 1. Model must generate multiple outputs
ChatGPTConfig(n=3, temperature=0.8)

# 2. RolePlaying must enable critic
RolePlaying(with_critic_in_the_loop=True)

# 3. Critic selects from proposals via reduce_step()
response = critic.reduce_step(list_of_messages)
```

### 2. Self-Improving CoT Workflow
```python
# The iterative loop:
# 1. Generate trace
# 2. Evaluate trace (get correctness, clarity, completeness scores)
# 3. If score < threshold and iterations < max_iterations:
#    - Get feedback
#    - Regenerate with feedback
#    - Go to step 2
# 4. Return best trace
```

### 3. Task Management
```python
# Hierarchy pattern:
# - Main task (root)
#   - Subtask 1 (generation)
#   - Subtask 2 (evaluation)
#   - Subtask 3 (refinement)
#   - Subtask 4 (verification)
#
# TaskManager orchestrates execution order
```

---

## Common Configurations

### For High-Quality Reasoning
```python
SelfImprovingCoTPipeline(
    max_iterations=5,                    # More refinement rounds
    score_threshold={'correctness': 0.9, # Per-dimension thresholds
                     'clarity': 0.85,
                     'completeness': 0.85},
)
```

### For Diverse Exploration
```python
ChatGPTConfig(
    temperature=0.9,    # Higher for more diversity
    n=5,                # More options for critic to choose from
)
```

### For Parallel Processing
```python
SelfImprovingCoTPipeline(
    batch_size=10,      # Process 10 at a time
    max_workers=8,      # Use 8 threads
)
```

---

## Troubleshooting Guide

### Issue: CriticAgent doesn't select anything
**Causes:** Missing `n>1` config, missing `with_critic_in_the_loop=True`
**Solution:** Check both configurations are set

### Issue: Quality scores don't improve
**Causes:** Weak evaluate_agent, low max_iterations, high threshold
**Solution:** 
- Improve evaluate_agent system message
- Increase max_iterations to 5
- Lower score_threshold

### Issue: Task decomposition fails
**Causes:** Invalid task content, agent can't parse response
**Solution:**
- Ensure task.content is > 5 characters
- Verify agent wraps subtasks in `<task>...</task>` tags
- Check validate_task_content() passes

---

## File Structure

```
CAUST/
├── README_CAMEL_RESOURCES.md          (this file)
├── CAMEL_PATTERNS_GUIDE.md            (comprehensive reference)
├── CAMEL_QUICK_REFERENCE.md           (quick lookup)
├── STORY_1_5_IMPLEMENTATION_TEMPLATE.md (working code template)
│
└── external/camel/
    ├── camel/
    │   ├── agents/
    │   │   ├── critic_agent.py
    │   │   ├── chat_agent.py
    │   │   └── ...
    │   ├── societies/
    │   │   └── role_playing.py
    │   ├── tasks/
    │   │   └── task.py
    │   ├── datagen/
    │   │   └── self_improving_cot.py
    │   └── ...
    │
    └── docs/
        ├── key_modules/
        │   └── tasks.md
        ├── cookbooks/
        │   ├── advanced_features/
        │   │   └── critic_agents_and_tree_search.ipynb
        │   ├── data_generation/
        │   │   └── self_improving_cot_generation.md
        │   └── multi_agent_society/
        │       └── task_generation.ipynb
        └── ...
```

---

## Next Steps

1. **Read** CAMEL_QUICK_REFERENCE.md for 10-minute overview
2. **Study** CAMEL_PATTERNS_GUIDE.md for deep understanding
3. **Review** STORY_1_5_IMPLEMENTATION_TEMPLATE.md code structure
4. **Implement** your Story 1.5 following the template
5. **Test** using the provided checklist
6. **Reference** CAMEL_QUICK_REFERENCE.md while coding

---

## Support Resources

### Documentation in Repository
- Main README: `/external/camel/README.md`
- API Reference: `/external/camel/docs/reference/`
- Cookbooks: `/external/camel/docs/cookbooks/`
- Examples: `/external/camel/examples/`

### Community
- GitHub: https://github.com/camel-ai/camel
- Discord: https://discord.camel-ai.org
- X/Twitter: https://x.com/camelaiorg

---

## Document Versions

- Created: October 2024
- CAMEL Version: 0.2.16+
- Python: 3.9+
- Dependencies: camel-ai, openai, pydantic

---

## Summary

You now have three complementary resources for implementing Story 1.5:

1. **CAMEL_PATTERNS_GUIDE.md** - Deep technical understanding
2. **CAMEL_QUICK_REFERENCE.md** - Fast lookup while coding
3. **STORY_1_5_IMPLEMENTATION_TEMPLATE.md** - Working code template

Together, these provide everything needed to successfully implement Story 1.5 using CAMEL's CriticAgent, Self-Improving CoT, and Task Module patterns.

Start with the Quick Reference, dive into the Patterns Guide as needed, and follow the Implementation Template for your solution.

Happy coding!
