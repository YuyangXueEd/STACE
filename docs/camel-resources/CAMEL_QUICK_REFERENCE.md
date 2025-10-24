# CAMEL-AI Quick Reference for Story 1.5

## CriticAgent Essentials

### Import
```python
from camel.agents import CriticAgent
from camel.types import RoleType
from camel.generators import SystemMessageGenerator
```

### Create
```python
sys_msg = SystemMessageGenerator().from_dict(
    meta_dict={'critic_role': 'expert', 'criteria': 'select best option'},
    role_tuple=('expert', RoleType.CRITIC)
)
critic = CriticAgent(system_message=sys_msg, verbose=True, retry_attempts=2)
```

### Use
```python
response = critic.reduce_step(list_of_messages)  # Selects from options
choice = response.msg.content
```

### Key Parameters
- `verbose=True` - Print critic's reasoning
- `retry_attempts=2` - Retry if invalid choice
- `message_window_size=6` - Context history

---

## Self-Improving CoT Essentials

### Import
```python
from camel.datagen import SelfImprovingCoTPipeline
from camel.agents import ChatAgent
```

### Create
```python
pipeline = SelfImprovingCoTPipeline(
    reason_agent=reason_agent,
    evaluate_agent=evaluate_agent,
    problems=[{"problem": "your question"}],
    max_iterations=3,
    score_threshold=0.8,
    output_path="results.json",
)
```

### Run
```python
results = pipeline.generate()
for result in results:
    print(result.final_trace)
    print(result.improvement_history)
```

### Key Parameters
- `max_iterations=3` - Max refinement rounds
- `score_threshold=0.8` - Quality target (0-1)
- `batch_size=5` - Parallel processing
- `max_workers=4` - Thread count

---

## Task Module Essentials

### Import
```python
from camel.tasks import Task, TaskManager
from camel.agents import ChatAgent
```

### Create Task
```python
task = Task(
    content="Your task description",
    id="task_1"
)
```

### Decompose
```python
subtasks = task.decompose(agent=agent)
# Returns list of Task objects
```

### Compose
```python
task.compose(agent=agent)
# Result stored in task.result
```

### Evolve
```python
task_manager = TaskManager(task)
new_task = task_manager.evolve(task, agent=agent)
# Task with increased complexity
```

### Hierarchical
```python
parent = Task(content="Parent", id="0")
child = Task(content="Child", id="1")
parent.add_subtask(child)
print(parent.to_string())
```

---

## RolePlaying + CriticAgent Pattern

```python
from camel.societies import RolePlaying
from camel.configs import ChatGPTConfig
from camel.models import ModelFactory

# Must have n>1 for critic to work
model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O_MINI,
    model_config_dict=ChatGPTConfig(
        temperature=0.8,
        n=3,  # CRITICAL: Generate multiple options
    ).as_dict()
)

society = RolePlaying(
    task_prompt="Your task",
    with_critic_in_the_loop=True,
    critic_criteria="selection criteria",
    assistant_agent_kwargs={'model': model},
    user_agent_kwargs={'model': model},
)

msg = society.init_chat()
for _ in range(10):
    asst_resp, user_resp = society.step(msg)
    if asst_resp.terminated or user_resp.terminated:
        break
    msg = asst_resp.msg
```

---

## Common Patterns

### 1. Generate Multiple Options
```python
# In model config, use n>1
ChatGPTConfig(n=3, temperature=0.8)
# This makes agents generate 3 options per request
```

### 2. Evaluate Traces
```python
# Agent-based
evaluator = ChatAgent("Rate correctness (0-1), clarity (0-1), completeness (0-1)")

# Reward model-based
from camel.models.reward import BaseRewardModel
# Use evaluator.evaluate(problem, trace)
```

### 3. Batch Processing
```python
# SelfImprovingCoTPipeline handles batching
# with ThreadPoolExecutor and retry logic
# Automatic resource monitoring and adjustment
```

### 4. Save Results
```python
# SelfImprovingCoTPipeline auto-saves to output_path
# Safe atomic writes with .tmp files
import json
with open('results.json', 'r') as f:
    data = json.load(f)
```

---

## File Locations (Absolute Paths)

**Source Code:**
- `/data/users/yyx/onProject/CAUST/external/camel/camel/agents/critic_agent.py`
- `/data/users/yyx/onProject/CAUST/external/camel/camel/societies/role_playing.py`
- `/data/users/yyx/onProject/CAUST/external/camel/camel/tasks/task.py`
- `/data/users/yyx/onProject/CAUST/external/camel/camel/datagen/self_improving_cot.py`

**Documentation:**
- `/data/users/yyx/onProject/CAUST/external/camel/docs/key_modules/tasks.md`
- `/data/users/yyx/onProject/CAUST/external/camel/docs/cookbooks/advanced_features/critic_agents_and_tree_search.ipynb`
- `/data/users/yyx/onProject/CAUST/external/camel/docs/cookbooks/data_generation/self_improving_cot_generation.md`

**Examples:**
- `/data/users/yyx/onProject/CAUST/external/camel/examples/tasks/task_generation.py`
- `/data/users/yyx/onProject/CAUST/external/camel/docs/cookbooks/multi_agent_society/task_generation.ipynb`

---

## Story 1.5 Implementation Checklist

- [ ] Set up Task with story content
- [ ] Create reason_agent (ChatAgent)
- [ ] Create evaluate_agent (ChatAgent)
- [ ] Configure model with n=3 for multiple options
- [ ] Set up RolePlaying with critic_in_the_loop=True
- [ ] Run multi-turn reasoning with critic selection
- [ ] Create SelfImprovingCoTPipeline for refinement
- [ ] Set score_threshold based on quality targets
- [ ] Run generate() and collect results
- [ ] Validate improvement_history shows progression
- [ ] Save final traces to output file

---

## Debugging Tips

### CriticAgent isn't selecting options?
- Check: Is `n>1` set in model config?
- Check: Is `with_critic_in_the_loop=True` in RolePlaying?
- Check: Increase `retry_attempts`

### CoT traces not improving?
- Check: Is evaluate_agent providing meaningful feedback?
- Check: Increase `max_iterations`
- Check: Lower `score_threshold` or use dimension-specific thresholds
- Check: Verify few_shot_examples if using them

### Task decomposition not working?
- Check: Agent can receive messages
- Check: Task content validates (not empty/too short)
- Check: Agent returns subtasks in `<task>...</task>` tags
- Check: Parser can extract the tags from response

### Batch processing too slow?
- Check: Increase `batch_size`
- Check: Increase `max_workers`
- Check: Check CPU/memory usage for constraints

---

## Key Differences: CriticAgent vs ChatAgent

| Feature | ChatAgent | CriticAgent |
|---------|-----------|-----------|
| Purpose | General conversation | Select from options |
| Main Method | step() | reduce_step() |
| Input | Single message | Multiple messages |
| Output | One response | Selected + reasoning |
| Retry Logic | None | 2 retries default |
| Use Case | Most tasks | Tree search/voting |

