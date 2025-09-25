# Introduction

[[courses/deep-reinforcement-learning/README#[1] Introduction|README]]

Lecture [1.1](https://youtu.be/SupFHGbytvA?feature=shared), [1.2](https://youtu.be/BYh36cb92JQ?feature=shared), [1.3](https://youtu.be/Ufww5pzc_N0?feature=shared)

## Reinforcement Learning

- Reinforcement learning: let the robot learn from its own experience
- Recent advances in AI
	- Diffusion models to generate images
	- LLMs
	- Data driven AI: data distribution drives the model
- Reinforcement learning
	- **A mathematical formalism for learning-based decision making**
	- **An approach for learning-based decision making and control from experience**
	- Classical RL: agent interacts with the environment
	- Modern methods: large scale optimization, evolutionary algorithms
	- Modern RL (deep RL): combines large scale optimization with classical RL
- What does RL do differently?
	- Supervised ML: learn a function that maps inputs to outputs
		- Assumes i.i.d. data
		- Knows ground truth labels
	- Reinforcement learning: learn a policy that maps states to actions
		- Does not assume i.i.d. data, previous outputs affect future inputs
		- Ground truth labels for each action are not available, only success/failure

## Problem Formulation

- Problem formulation ![[rl_cycle.png]]
	- Input: $s_t$ (state) at each time step
	- Output: $a_t$ (action) at each time step
	- Data: $(s_t, a_t, r_t)_{t=1}^{T}$
	- Goal: learn policy $\pi_{\theta}: s_t \rightarrow a_t$
- Purposes of RL
	- Learning complex physical tasks (like a robot dog jumping over obstacles)
		- Often difficult to code explicitly
	- Unexpected solutions: RL can find solutions that are not obvious to humans
	- Applied at scale in real world: ex. many robots sorting trash
	- RL with LLMs: training models with human scoring
	- RL with image generation: reward function is the similarity between the language prompt and the generated image, to find most relevant image
	- **Data-driven methods don't try to do better than the data, RL figures out new, better solutions**
- Some philosophy
	- Richard Sutton: "The two methods that seem to scale arbitrarily … are *learning* and *search*"
		- **Learning**: extract patterns from data (the "Deep" part)
		- **Search**: use computation to extract inferences (the "RL" part)
	- Why do we need ML?
	    - One reason: to produce adaptable and complex decisions
	- Why do we need RL?
	    - Learing-based control in real-world settings is a major open problem
- Where do rewards come from?
	- Copying observed behavior
	- Inferring rewards from observations: inverse RL
		- Inferring intentions?
	- Learning from other tasks: transfer learning, Meta learning
- Modern RL
	- Advances in pretrained models
		- Vision-language-action models for robotics (RT-2, Octo)
	- How to build intelligent machines?
		- Hypothesis: learning is the basis of intelligence
		- Further hypothesis: there is a single learning algorithm that can learn everything
	- Challenges
		- Transfer learning
		- Too data intensive, whereas humans learn quickly
		- No amazing methods for using both data and RL
		- Role of prediction in RL
