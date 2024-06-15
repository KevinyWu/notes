# Deep Reinforcement Learning

[Lectures (fall 2023)](https://youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&feature=shared)

[Course website](https://rail.eecs.berkeley.edu/deeprlcourse/)

Notes from the course "CS 285: Deep RL" taught by Sergey Levine at Berkeley.

- [Deep Reinforcement Learning](#deep-reinforcement-learning)
  - [1 Introduction](#1-introduction)
  - [2 Supervised Learning of Behaviors](#2-supervised-learning-of-behaviors)

## 1 Introduction

Lecture [1.1](https://youtu.be/SupFHGbytvA?feature=shared), [1.2](https://youtu.be/BYh36cb92JQ?feature=shared), [1.3](https://youtu.be/Ufww5pzc_N0?feature=shared)

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
- Problem formulation
  - <img src="figures/rl_cycle.png" width="400" alt="rl_cycle">
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
  - Richard Sutton: "The two methods that seem to scale arbitrarily ... are *learning* and *search*"
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

## 2 Supervised Learning of Behaviors

Lecture [2.1](https://youtu.be/tbLaFtYpWWU?feature=shared), [2.2](https://youtu.be/YivJ9KDjn-o?feature=shared), [2.3](https://youtu.be/ppN5ORNrMos?feature=shared)

- Notation
  - <img src="figures/notation.png" width="600" alt="notation">
  - State $s_t$ is different from observation $o_t$
    - State is a complete and concise representation of state of the world (**fully observed**)
    - Observation is what the agent sees (**partially observed**)
    - State can sometimes be inferred from observation
  - **Markov assumption**: $s_t$ contains all relevant information from the past (don't need $s_{t-1}, s_{t-2}, \ldots$ to predict $s_{t+1}$)
- Imitation learning
  - Learn policies using supervised learning
- **Behavioral cloning**: learn a policy that mimics an expert's behavior
  - Collect data from expert
  - Train a policy to predict the expert's actions
  - Problems
    - Distribution mismatch between training and test data
    - Violates i.i.d. assumption: small errors lead to larger and larger errors over time