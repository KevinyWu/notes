# Peter Stone - Practical Reinforcement Learning - Lessons from 30 Years of Research

[[talks/README#[2024-10-01] Peter Stone - Practical Reinforcement Learning - Lessons from 30 Years of Research|README]]

[Recording](https://youtu.be/5Dw4OoJK9Qw?feature=shared)

## Motivation

- **Question**: To what degree can autonomous intelligent agents learn in the presence of teammates and/or adversaries in real-time, dynamic domains?
- RL as a tool
	- Does RL work?
	- If not, how do we make it work?

## Practical RL

- Representation
	- Choose the algorithm to suit the problem
	- Learn the representation to suit the problem
- Interaction
	- Multiagent interaction is complicated
	- Interaction with people can be simplifying
- Synthesis
	- Decompose problems to use multiple algorithms
	- Synthesize concepts
- Mortality
	- Leverage the past
	- Acknowledge a finite future (you can't try everything before making a decision)

## Representation

- **Choosing the algorithm to suit the problem**
- Parametrized learning problem
	- Size of state space
	- Stochasticity in transitions
	- Expressiveness of function approximator
	- Generalization width
	- State noise
- Real world RL without simulation
	- Low dimensional state space crucial
- Hybrid action spaces and partial observability
	- Learn the action class and its parameters together
- NEAT+Q: first example of neural network function approximator for Mountain Car
- Causal dynamics learning
	- Learn task-independent state abstraction
	- Leverage for downstream tasks
- Causal policy gradient
	- [Disentangled Unsupervised Skill Discovery for Efficient Hierarchical Reinforcement Learning](https://jiahenghu.github.io/DUSDi-site/)
	- Factored action space
	- Whole-body mobile manipulation
- ELDEN
	- Exploration to discover local dependencies among state variables

## Interactions

- **Multiagent interaction is complicated**
- RoboCup soccer keepaway
- CMLES: Convergence with Model Learning and Safety
	- For optimal multiagent learning
	- Actions serve dual purpose: yield immediate payoff, train other agents
	- Optimal policy lures agents to learn exploitable behavior
- Ad hoc teamwork
	- Humans are good at this! (pickup soccer, etc)
	- Unknown teammates (programmed by others)
	- Challenge: create a good team player
- **Interaction with people can be simplifying**
- RLHF
	- [[Interactively shaping agents via human reinforcement - The TAMER framework]]
	- First work on RLHF
	- Deep TAMER: extension to pixels
- EMPATHIC: Learning from implicit human feedback
- BCO: Behavior Cloning from Observation

## Synthesis

- **Decompose problems to use multiple algorithms**
- Layered learning
	- Decompose problem into many subtasks
	- Use learning of one subtask as input to the learning of the next subtask
- **Synthesize concepts**
- Continuous state function approximation
- Sample efficiency: model-based learning e.g. R-Max
- Hierarchy e.g. MAXQ

## Mortality

- **Leverage the past**
- Transfer learning
- Automated curriculum learning
	- People learn via curricula
		- Task creation, task sequencing, transfer learning
	- Learn a teacher agent
- Grounded simulation learning for Sim2Real
	- We will never have perfect simulators
	- Simulator grounding -> policy improvement -> real world execution -> etc.
- Distribution matching for Sim2Real and RL
	- Ground simulator by matching policy's state visitation distributions in sim and real
- **Acknowledge a finite future**
- In reality, we can't explore everywhere
- Texplore for targeted exploration: choosing where **not** to explore
	- Only explore states that are uncertain and promising
	- Real-world autonomous car
- Lifelong learning
	- Continual learning and private unlearning
	- LIBERO: benchmarking knowledge transfer for lifelong robot learning

## Build Agents

- RoboCup
- Developmental robots
- General purpose service robots
- [Deep Reinforcement Learning for Robotics: A Survey of Real-World Successes](https://arxiv.org/abs/2408.03539)
- GT Sophy
	- Gran Turismo: very physically accurate racing simulator
	- Challenges
		- Race car control
		- Multi-agent tactics
		- Etiquette (no cheating)
	- Turns out end-to-end deep RL is better than MPC and even beats best human drivers
	- Quantile regression SAC
	- Huber loss (loss function used in robust regression, less sensitive to outliers)
	- Model designed to fit on PlayStation
	- Adding reward term for decreasing distance from car in front or increasing from car behind to incentivize passing/blocking
	- Exposure problem: curriculum of racing situations that don't happen often but are very important
	- Curating experience replay buffer
	- Etiquette: additional reward components to discourage cheating such as pushing other cars off the track
