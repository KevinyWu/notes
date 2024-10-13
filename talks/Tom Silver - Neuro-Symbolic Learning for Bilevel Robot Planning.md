# Tom Silver - Neuro-Symbolic Learning for Bilevel Robot Planning

#tamp
#unsupervised
#planning

[[talks/README#[2024-04-06] Tom Silver - Neuro-Symbolic Learning for Bilevel Robot Planning|README]]

[Recording](https://www.youtube.com/watch?v=ikS2MN0HWxw)

## General-purpose Robots

- Want robots to become domain experts
    - Learning to specialize
    - Planning and reasoning in the moment
    - Needs: 1) time efficient, 2) data efficient, 3) human efficient, 4) solve hard, long-horizon tasks
        - **Task and motion planning (TAMP)** bad at 1, 2, 3; good at 4 (needs lots of human domain knowledge)
        - **RL** bad at at 2, 4, good at 1, 3 (simulation is good but data hungry)
        - **LLMs** good at 1, 2, bad at 3, 4 (cannot solve hard tasks)
        - Tom's approach: **Learning to plan and planning to learn with abstractions**

## Learning to Plan and Planning to Learn with Abstractions

- Use computer vision to construct object-centered states
- **Predicates**: relational classifiers over object states
    - Boolean function, i.e. evaluates to true if block is on the table
    - Evaluate predicates for all objects in environment: **abstract state space**
- **Skill operators**
    - Arguments: placeholders for objects
    - Preconditions: what must be true in order to use this operator
    - Add/delete effects: how is the abstract state space changed
- Why predicates and operators?
    - Powerful off-the-shelf symbolic planners exist
- **Bilevel planning: view abstractions as constraints**
    - Find trajectory of **low-level** states and actions so that **high-level** abstract states are followed, transitions are valid
- Still needs humans to design domain specific predicates and operators!

## Learning Operators Given Predicates

- Given full-task demonstrations, segment them into subtasks
- Use the predicates to get the abstract state of each subtask
- Look at differences between subtasks: **transitions**
    - Two transitions are in the same set if they are equivalent up to object substitutions
- Problem: we don't know predicates in any unsupervised settings
    - Want the robot to synthesize its own predicates

## Predicate Learning as Program Synthesis

- Generate a pool of candidate predicates from a grammar
- Perform a hill-climbing search to add predicates to set one at a time
    - Objective is to find predicates effective for bilevel planning
    - **Real objective function**: how long symbolic planner takes to solve a task
        - Intractable! optimization problem too slow
        - **Surrogate objective function**: estimate the probability that the optimization succeeds
- Learned predicates are task-aware, can increase planning speed

## Turning Abstract Plan into Low-Level Action

- A **skill** has an operator (symbolic high-level) and policy (neural low-level)
- Planner may be insufficient - abstractions might be liars!
    - [Example](https://youtu.be/ikS2MN0HWxw?t=1850)
    - Be less trusting of abstractions
- **Parameterized skills**
    - Ex. with a hook tool, have different parameters, being the location on the tool we grasp
    - Use a **skill sampler**
- Want to learn parameterized skills to move between abstract states

## Planning to Learn

- After learning to plan, collect online data to improve learning
- Learning with practice: learn low-level concepts from active data collection
    - Active learning in ML, exploration in RL
- Learning skill samplers (for parameterized skills) through practice
    - Choose skills to practice to maximize future task success
    - Skill competence: probability a skill with achieve its intended abstract transition
        - Practice skill with lowest competence may not be improvable or unnecessary for task distribution
        - Better approach: practice skill whose improvement would lead to greatest expected improvement for overall task distribution
            - 1. **Estimate** the competence
            - 2. **Extrapolate** the competence
            - 3. **Situate** the skill in the task distribution
            - Unsupervised: robot learns without human intervention
