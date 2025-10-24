# CAMEL-AI Framework Patterns for Story 1.5

## Executive Summary

This document provides a comprehensive guide to implementing Story 1.5 using CAMEL-AI's advanced patterns. The three core patterns are:

1. **CriticAgent & Tree Search** - Multi-agent collaborative reasoning with critic feedback
2. **Self-Improving CoT Generation** - Iterative refinement of reasoning traces through evaluation and feedback
3. **Task Module** - Structured task management with decomposition, composition, and hierarchical support

---

## 1. CriticAgent and Tree Search Pattern

### Purpose
The CriticAgent enables tree search for complex problem-solving by having multiple agents generate proposals and a critic agent selecting and providing feedback.

### Key Concepts

#### A. The CriticAgent Class
Located in: `/data/users/yyx/onProject/CAUST/external/camel/camel/agents/critic_agent.py`

**What Makes CriticAgent Different from ChatAgent:**
- Extends ChatAgent with specialized functionality for option reduction
- Has a `reduce_step()` method that:
  1. Flattens multiple message options
  2. Gets the critic's choice among options
  3. Returns the selected option with reasoning
- Handles retry logic if invalid selections are made
- Tracks an `options_dict` to map choice numbers to actual messages

#### B. Core CriticAgent Methods

```python
class CriticAgent(ChatAgent):
    def __init__(
        self,
        system_message: Optional[Union[BaseMessage, str]] = None,
        model: Optional[...] = None,
        memory: Optional[AgentMemory] = None,
        message_window_size: int = 6,
        retry_attempts: int = 2,  # Retries if invalid choice given
        verbose: bool = False,
        logger_color: Any = Fore.MAGENTA,
    ) -> None:
```

**Key Methods:**
1. `flatten_options(messages)` - Converts multiple proposals into numbered choices
2. `get_option(input_message)` - Gets critic's selection with retry logic
3. `parse_critic(critic_msg)` - Extracts the chosen option number
4. `reduce_step(input_messages)` - Main method for selecting from proposals
5. `clone(with_memory)` - Creates a new instance maintaining state if needed

#### C. System Message for CriticAgent

Generated using `SystemMessageGenerator` with role type `RoleType.CRITIC`:

```python
# Configuration
critic_role = 'a picky critic'
meta_dict = dict(
    critic_role=critic_role,
    criteria='Help better accomplish the task.'
)
role_tuple = (critic_role, RoleType.CRITIC)

# Generate system message
sys_msg = sys_msg_gen().from_dict(
    meta_dict=meta_dict,
    role_tuple=role_tuple
)

# Resulting system message:
"""
You are a a picky critic who teams up with a {user_role} and a {assistant_role} 
to solve a task: {task}.
Your job is to select an option from their proposals and provides your explanations.
Your selection criteria are Help better accomplish the task.
You always have to choose an option from the proposals.
"""

critic_agent = CriticAgent(system_message=sys_msg, verbose=True)
```

### 2. Integration with RolePlaying

The `RolePlaying` class integrates CriticAgent for tree search workflows:

**Key Parameters:**
```python
society = RolePlaying(
    task_prompt='Develop a plan to TRAVEL TO THE PAST and make changes.',
    with_task_specify=True,
    task_specify_agent_kwargs={'model': model},
    
    user_role_name='an ambitious aspiring TIME TRAVELER',
    user_agent_kwargs={'model': model},
    
    assistant_role_name='the best-ever experimental physicist',
    assistant_agent_kwargs={'model': model},
    
    # CRITIC CONFIGURATION
    with_critic_in_the_loop=True,
    critic_criteria='improve the task performance',
    critic_kwargs=dict(verbose=True),
)
```

**How RolePlaying Uses CriticAgent:**

1. Both user and assistant agents generate multiple responses (via `n=3` in model config)
2. `_reduce_message_options()` method is called with multiple messages
3. If `with_critic_in_the_loop=True`, CriticAgent.reduce_step() selects best option
4. Selected message is fed back to continue the conversation

**Relevant RolePlaying Code:**
```python
def _reduce_message_options(
    self,
    messages: Sequence[BaseMessage],
) -> BaseMessage:
    if len(messages) > 1 and not self.with_critic_in_the_loop:
        raise ValueError(
            "Got than one message to process. "
            f"Num of messages: {len(messages)}."
        )
    elif self.with_critic_in_the_loop and self.critic is not None:
        critic_response = self.critic.reduce_step(messages)
        processed_msg = critic_response.msg
    else:
        processed_msg = messages[0]
    
    return processed_msg
```

### 3. Tree Search Workflow

**The Complete Flow:**

```
Initial Task
    ↓
User Agent generates N options (n=3)
    ↓
CriticAgent evaluates all options
    ↓
CriticAgent selects best option with reasoning
    ↓
Selected message → Assistant Agent
    ↓
Assistant Agent generates N options
    ↓
CriticAgent evaluates all options
    ↓
[Loop continues until termination]
```

### 4. Implementation Example

```python
from camel.agents import CriticAgent
from camel.generators import SystemMessageGenerator as sys_msg_gen
from camel.messages import BaseMessage as bm
from camel.types import RoleType
from camel.societies import RolePlaying
from camel.configs import ChatGPTConfig
from camel.types import TaskType, ModelType, ModelPlatformType
from camel.models import ModelFactory

# Setup model with multiple samples
model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O_MINI,
    model_config_dict=ChatGPTConfig(
        temperature=0.8,  # Higher for diversity
        n=3,              # Generate 3 options per request
    ).as_dict()
)

# Create society with critic
society = RolePlaying(
    task_prompt='Design a time travel strategy',
    with_task_specify=True,
    task_specify_agent_kwargs={'model': model},
    
    user_role_name='time traveler',
    user_agent_kwargs={'model': model},
    
    assistant_role_name='physicist',
    assistant_agent_kwargs={'model': model},
    
    with_critic_in_the_loop=True,
    critic_criteria='select the most feasible and innovative option',
    critic_kwargs=dict(verbose=True),
)

# Run the conversation
input_msg = society.init_chat()

for round_num in range(10):
    assistant_response, user_response = society.step(input_msg)
    
    if assistant_response.terminated or user_response.terminated:
        break
    
    print(f'[AI User] {user_response.msg.content}')
    print(f'[AI Assistant] {assistant_response.msg.content}')
    
    if 'CAMEL_TASK_DONE' in user_response.msg.content:
        break
    
    input_msg = assistant_response.msg
```

---

## 2. Self-Improving CoT Generation Pattern

### Purpose
Generates and iteratively improves Chain-of-Thought reasoning traces through a cycle of generation → evaluation → refinement.

### Architecture Overview

```
Problem Input
    ↓
[ITERATION LOOP]
    ├─→ reason_agent generates/improves trace
    │
    ├─→ evaluate_agent OR reward_model evaluates trace
    │        ├─ Evaluates: correctness, clarity, completeness
    │        └─ Provides feedback
    │
    └─→ Check threshold: if met, exit loop; else continue
    
Final Trace → Output
```

### 1. Core Components

Located in: `/data/users/yyx/onProject/CAUST/external/camel/camel/datagen/self_improving_cot.py`

#### A. Pipeline Class
```python
class SelfImprovingCoTPipeline:
    def __init__(
        self,
        reason_agent: ChatAgent,           # Generates/improves reasoning
        problems: List[Dict],
        max_iterations: int = 3,           # Maximum refinement iterations
        score_threshold: Union[float, Dict[str, float]] = 0.7,
        evaluate_agent: Optional[ChatAgent] = None,  # Evaluates traces
        reward_model: Optional[BaseRewardModel] = None,
        output_path: Optional[str] = None,
        few_shot_examples: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_workers: Optional[int] = None,
        solution_pattern: str = r'\boxed{(.*?)}',
        trace_pattern: Optional[str] = None,
    ):
```

#### B. Evaluation Types

**Agent-Based Evaluation:**
```python
evaluate_agent = ChatAgent(
    "You are a highly critical teacher who evaluates the student's answers "
    "with a meticulous and demanding approach."
)

# Evaluates for:
# - Correctness: Does the trace logically solve the problem?
# - Clarity: Is the reasoning easy to follow?
# - Completeness: Are all necessary steps included?
```

**Reward Model Evaluation:**
```python
# Returns scores on predefined dimensions:
# - Correctness
# - Coherence
# - Complexity
# - Verbosity
```

### 2. Step-by-Step Workflow

**Step 1: Initial Trace Generation**
```python
reason_agent = ChatAgent(
    """Answer my question and give your 
    final answer within \boxed{}."""
)

# Generates initial reasoning trace
# Optionally guided by few-shot examples
```

**Step 2: Evaluation**
```python
# Option 1: Agent-based
evaluation = evaluate_agent.step(
    f"Problem: {problem}\nTrace: {trace}"
)
# Returns: AgentTraceEvaluation with correctness, clarity, 
#          completeness scores (0-1) and feedback

# Option 2: Reward Model
evaluation = reward_model.evaluate(
    problem=problem,
    trace=trace
)
# Returns: RewardTraceEvaluation with dimension scores
```

**Step 3: Iterative Refinement**
```python
# If evaluation doesn't meet threshold
feedback = evaluation.feedback

# Generate improved trace using feedback
improved_trace = reason_agent.step(
    f"""Previous trace: {trace}
    Feedback: {feedback}
    
    Please improve the trace addressing the feedback."""
)

# Re-evaluate improved trace
# Repeat until threshold met or max_iterations reached
```

### 3. Implementation Example

```python
from camel.agents import ChatAgent
from camel.datagen import SelfImprovingCoTPipeline
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

# Initialize agents
reason_agent = ChatAgent(
    """Answer my question and provide your final answer 
    within \\boxed{}. Think step-by-step."""
)

evaluate_agent = ChatAgent(
    "You are a highly critical evaluator. Rate the reasoning for: "
    "1) Correctness (0-1), 2) Clarity (0-1), 3) Completeness (0-1). "
    "Provide specific feedback."
)

# Prepare problems
problems = [
    {"problem": "A train leaves station A at 60 mph..."},
    {"problem": "Calculate the derivative of..."},
]

# Create pipeline
pipeline = SelfImprovingCoTPipeline(
    reason_agent=reason_agent,
    evaluate_agent=evaluate_agent,
    problems=problems,
    max_iterations=3,
    score_threshold=0.7,  # Average score threshold
    output_path="cot_traces.json",
    batch_size=5,
    max_workers=4,
)

# Generate traces
results = pipeline.generate()

# Access results
for result in results:
    print(f"Problem: {result.problem}")
    print(f"Final Trace: {result.final_trace}")
    print(f"Improvement History: {result.improvement_history}")
```

### 4. Advanced Configuration: Thresholds

```python
# Single threshold applied to average score
pipeline = SelfImprovingCoTPipeline(
    ...,
    score_threshold=0.8,  # Average must be >= 0.8
)

# Per-dimension thresholds
pipeline = SelfImprovingCoTPipeline(
    ...,
    score_threshold={
        "correctness": 0.9,
        "clarity": 0.7,
        "completeness": 0.8,
    }
)
```

### 5. Batch Processing with Error Handling

```python
# The pipeline uses BatchProcessor for parallel processing
# with automatic retry and dynamic batch sizing

# Handles:
# - ThreadPoolExecutor for parallel trace generation
# - Exponential backoff retry on API errors
# - Dynamic batch size adjustment based on success rates
# - Safe JSON file writing (atomic writes via .tmp files)

# Progress tracking:
# - Logs CPU/memory usage
# - Reports processing metrics
# - Updates output file in real-time
```

### 6. Key Data Structures

```python
class TraceIteration(BaseModel):
    iteration: int
    trace: str
    evaluation: Union[AgentTraceEvaluation, RewardTraceEvaluation]

class ProblemResult(BaseModel):
    id: Optional[str]
    problem: str
    solution: Optional[str]
    final_trace: str
    boxed_answer_success: bool
    improvement_history: List[TraceIteration]

class AgentTraceEvaluation(BaseModel):
    correctness: float      # 0-1
    clarity: float          # 0-1
    completeness: float     # 0-1
    feedback: str
```

---

## 3. Task Module Pattern

### Purpose
Provides structured, hierarchical task management with decomposition, composition, and evolution capabilities.

### Key Concepts

Located in: `/data/users/yyx/onProject/CAUST/external/camel/camel/tasks/task.py`

#### A. Task Attributes

```python
class Task(BaseModel):
    content: str                    # Task description
    id: str                         # Unique identifier
    state: TaskState               # OPEN, RUNNING, DONE, FAILED, DELETED
    type: Optional[str]            # Task type
    parent: Optional[Task]         # Parent task for hierarchies
    subtasks: List[Task]           # Child tasks
    result: Optional[str]          # Task result/output
    failure_count: int             # Tracks failures
    assigned_worker_id: Optional[str]  # For workforce systems
    dependencies: List[Task]       # Task dependencies
    additional_info: Optional[Dict]    # Metadata
    image_list: Optional[List[...]]    # Multimodal support
    video_bytes: Optional[bytes]   # Video content
```

#### B. Task States

```python
class TaskState(str, Enum):
    OPEN = "OPEN"          # Ready to be executed
    RUNNING = "RUNNING"    # Currently being executed
    DONE = "DONE"          # Successfully completed
    FAILED = "FAILED"      # Failed to complete
    DELETED = "DELETED"    # Deleted/cancelled
```

### 1. Creating Basic Tasks

```python
from camel.tasks import Task

# Simple task
task = Task(
    content="Weng earns $12 an hour for babysitting. How much for 51 minutes?",
    id="0",
)

# Task with metadata
task = Task(
    content="Design a time travel experiment",
    id="physics_1",
    type="physics",
    additional_info={"difficulty": "hard", "domain": "theoretical physics"}
)
```

### 2. Hierarchical Tasks

```python
# Create root task
root_task = Task(content="Prepare a meal", id="0")

# Create subtasks
sub_task_1 = Task(content="Shop for ingredients", id="1")
sub_task_2 = Task(content="Cook the meal", id="2")
sub_task_3 = Task(content="Set the table", id="3")

# Create sub-subtasks
sub_task_2_1 = Task(content="Chop vegetables", id="2.1")
sub_task_2_2 = Task(content="Cook rice", id="2.2")

# Build hierarchy
root_task.add_subtask(sub_task_1)
root_task.add_subtask(sub_task_2)
root_task.add_subtask(sub_task_3)

sub_task_2.add_subtask(sub_task_2_1)
sub_task_2.add_subtask(sub_task_2_2)

# Print structure
print(root_task.to_string())
# Output:
# Task 0: Prepare a meal
# Task 1: Shop for ingredients
# Task 2: Cook the meal
#     Task 2.1: Chop vegetables
#     Task 2.2: Cook rice
# Task 3: Set the table
```

### 3. Task Decomposition

Breaking down a complex task into subtasks:

```python
from camel.agents import ChatAgent
from camel.tasks import Task

agent = ChatAgent("You're a helpful assistant")

task = Task(
    content="Weng earns $12 an hour. She worked 51 minutes. How much did she earn?",
    id="0",
)

# Decompose task
new_tasks = task.decompose(agent=agent)

for t in new_tasks:
    print(t.to_string())

# Output:
# Task 0.0: Convert 51 minutes into hours.
# Task 0.1: Calculate Weng's earnings for the converted hours at $12/hour.
# Task 0.2: Provide the final earnings amount.
```

**How Decomposition Works:**
1. Agent receives task with decomposition prompt
2. Agent generates subtasks wrapped in `<task>...</task>` tags
3. Parser extracts subtasks and creates Task objects
4. Subtasks are automatically linked to parent

### 4. Task Composition

Combining subtask results into a final answer:

```python
# After subtasks are completed and have results set
task.compose(agent=agent)

# This calls the compose agent to synthesize results
print(task.result)  # Final composed result
```

### 5. Task Evolution

Modifying a task while preserving intent:

```python
from camel.tasks import Task, TaskManager

task = Task(
    content="Weng earns $12/hour for babysitting. 51 minutes?",
    id="0",
)

task_manager = TaskManager(task)

# Evolve the task (makes it more complex/detailed)
evolved_task = task_manager.evolve(task, agent=agent)

print(evolved_task.to_string())
# Output might be:
# Task 0.0: Weng earns $12/hour. She worked 1 hr 45 min.
#          Plus $5 bonus. How much total?
```

### 6. TaskManager for Workflow Orchestration

```python
from camel.tasks import Task, TaskManager

# Create root task
root_task = Task(
    content="Build a chatbot application",
    id="0"
)

# Create subtasks
task1 = Task(content="Design architecture", id="1")
task2 = Task(content="Implement models", id="2")
task3 = Task(content="Create UI", id="3")

# Create task manager
task_manager = TaskManager(root_task)

# Add tasks to manager
task_manager.add_tasks([task1, task2, task3])

# Set task dependencies (serial or parallel)
TaskManager.set_tasks_dependence(
    root=root_task,
    others=[task1, task2, task3],
    type="serial"  # task1 → task2 → task3
)

# Or parallel execution
TaskManager.set_tasks_dependence(
    root=root_task,
    others=[task1, task2, task3],
    type="parallel"  # root → task1, task2, task3
)

# Topological sort for execution order
sorted_tasks = TaskManager.topological_sort([task1, task2, task3])
```

### 7. Task Validation

```python
from camel.tasks.task import validate_task_content, TaskValidationMode

# Validate input task content
is_valid = validate_task_content(
    content="Build a time travel device",
    task_id="physics_1",
    min_length=5,
    mode=TaskValidationMode.INPUT
)

# Validate output/result
is_valid = validate_task_content(
    content="Prototype completed with 95% accuracy",
    task_id="physics_1",
    mode=TaskValidationMode.OUTPUT,
    check_failure_patterns=True  # Checks for failure indicators
)
```

### 8. Task Result Management

```python
task = Task(content="Solve the equation", id="0")

# Update task result and mark as DONE
task.update_result("x = 5, verified through substitution")

# Check task state
print(task.state)  # TaskState.DONE
print(task.result)  # "x = 5, verified through substitution"

# Get results including subtasks
result_str = task.get_result()
```

### 9. Integration with ChatAgent

```python
from camel.agents import ChatAgent
from camel.tasks import Task

# Create agent
agent = ChatAgent("You're a helpful task solver")

# Create task
task = Task(
    content="Analyze the impact of climate change on coastal cities",
    id="analysis_1"
)

# Decompose
subtasks = task.decompose(agent=agent)

# For each subtask, get solution from agent
for subtask in subtasks:
    user_msg = BaseMessage.make_user_message(
        role_name="Analyst",
        content=subtask.content
    )
    response = agent.step(user_msg)
    subtask.update_result(response.msg.content)

# Compose final result
task.compose(agent=agent)
print(f"Final result: {task.result}")
```

---

## Integration Strategy for Story 1.5

### Recommended Architecture

```
Story 1.5 Implementation
│
├─ LAYER 1: Task Management
│  ├─ Main task: "Generate improved reasoning with critic feedback"
│  ├─ Decompose into subtasks:
│  │  ├─ Generate initial solution
│  │  ├─ Evaluate solution
│  │  ├─ Refine based on feedback
│  │  └─ Verify final result
│  └─ Use TaskManager for orchestration
│
├─ LAYER 2: Multi-Agent Reasoning (RolePlaying + CriticAgent)
│  ├─ Set up RolePlaying society
│  ├─ Enable CriticAgent for tree search
│  ├─ Configure model for multiple outputs (n=3)
│  ├─ Run iterative conversation with critic feedback
│  └─ Extract best reasoning path
│
└─ LAYER 3: Self-Improving CoT (Optional Enhancement)
   ├─ Use reasoning outputs as initial traces
   ├─ Run SelfImprovingCoTPipeline for refinement
   ├─ Leverage evaluate_agent for quality assessment
   ├─ Batch process multiple cases
   └─ Output improved traces
```

### Example Integration Code

```python
from camel.agents import ChatAgent, CriticAgent
from camel.societies import RolePlaying
from camel.datagen import SelfImprovingCoTPipeline
from camel.tasks import Task, TaskManager
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig

# Setup
model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O_MINI,
    model_config_dict=ChatGPTConfig(
        temperature=0.8,
        n=3,  # Multiple options for critic
    ).as_dict()
)

# PHASE 1: Task Definition
root_task = Task(
    content="Develop a time travel strategy with iterative improvement",
    id="story_1_5"
)

# PHASE 2: RolePlaying with CriticAgent (Tree Search)
society = RolePlaying(
    task_prompt=root_task.content,
    with_task_specify=True,
    task_specify_agent_kwargs={'model': model},
    
    user_role_name="curious scientist",
    user_agent_kwargs={'model': model},
    
    assistant_role_name="brilliant physicist",
    assistant_agent_kwargs={'model': model},
    
    with_critic_in_the_loop=True,
    critic_criteria="select most feasible and innovative options",
    critic_kwargs=dict(verbose=True),
)

# Collect reasoning traces
reasoning_traces = []
input_msg = society.init_chat()

for round_num in range(5):
    asst_resp, user_resp = society.step(input_msg)
    reasoning_traces.append(asst_resp.msg.content)
    
    if asst_resp.terminated or user_resp.terminated:
        break
    
    input_msg = asst_resp.msg

# PHASE 3: Self-Improving CoT (Refinement)
problems = [{"problem": root_task.content}]

reason_agent = ChatAgent(
    "You are an expert at refining reasoning traces"
)

evaluate_agent = ChatAgent(
    "You are a critical evaluator. Rate: 1) Correctness, "
    "2) Clarity, 3) Completeness. Provide feedback."
)

pipeline = SelfImprovingCoTPipeline(
    reason_agent=reason_agent,
    evaluate_agent=evaluate_agent,
    problems=problems,
    max_iterations=3,
    score_threshold=0.8,
)

results = pipeline.generate()

# PHASE 4: Store Results in Task
final_trace = results[0].final_trace
root_task.update_result(final_trace)

print(f"Final improved reasoning:\n{root_task.result}")
```

---

## Summary of Key Patterns

| Pattern | Purpose | Key Component | Main Advantage |
|---------|---------|----------------|-----------------|
| CriticAgent + Tree Search | Multi-option selection with feedback | `reduce_step()`, RolePlaying | Explores multiple solution paths |
| Self-Improving CoT | Iterative reasoning refinement | `SelfImprovingCoTPipeline` | Continuous quality improvement |
| Task Module | Structured task management | `Task`, `TaskManager` | Clear hierarchy and orchestration |

---

## File References

- CriticAgent: `/data/users/yyx/onProject/CAUST/external/camel/camel/agents/critic_agent.py`
- RolePlaying: `/data/users/yyx/onProject/CAUST/external/camel/camel/societies/role_playing.py`
- Task: `/data/users/yyx/onProject/CAUST/external/camel/camel/tasks/task.py`
- SelfImprovingCoTPipeline: `/data/users/yyx/onProject/CAUST/external/camel/camel/datagen/self_improving_cot.py`
- Cookbook - Critic & Tree Search: `/data/users/yyx/onProject/CAUST/external/camel/docs/cookbooks/advanced_features/critic_agents_and_tree_search.ipynb`
- Cookbook - Self-Improving CoT: `/data/users/yyx/onProject/CAUST/external/camel/docs/cookbooks/data_generation/self_improving_cot_generation.md`
- Documentation - Tasks: `/data/users/yyx/onProject/CAUST/external/camel/docs/key_modules/tasks.md`
