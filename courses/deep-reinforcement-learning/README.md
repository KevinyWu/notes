# Deep Reinforcement Learning

Notes from the course "CS 285: Deep RL" taught by Sergey Levine at Berkeley (Fall 2023).

[Lectures (fall 2023)](https://youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&feature=shared)
[Course website](https://rail.eecs.berkeley.edu/deeprlcourse/)

## Contents

- [[courses/deep-reinforcement-learning/README#[1] Introduction|1 Introduction]]
- [[courses/deep-reinforcement-learning/README#[2] Supervised Learning of Behaviors|2 Supervised Learning of Behaviors]]
- [[courses/deep-reinforcement-learning/README#[3] Pytorch Tutorial|3 Pytorch Tutorial]]
- [[courses/deep-reinforcement-learning/README#[4] Introduction to Reinforcement Learning|4 Introduction to Reinforcement Learning]]
- [[courses/deep-reinforcement-learning/README#[5] Policy Gradients|5 Policy Gradients]]
- [[courses/deep-reinforcement-learning/README#[6] Actor-Critic Algorithms|6 Actor-Critic Algorithms]]

## [1] Introduction

[[courses/deep-reinforcement-learning/1 Introduction|1 Introduction]]
- Reinforcement learning (RL) focuses on decision-making and control through experience, differing from supervised learning by learning policies without ground truth labels
- The RL problem is formulated around mapping states to actions, learning from rewards, and finding novel solutions for complex tasks (e.g., robot behavior, image generation)
- Modern RL combines classical methods with large-scale optimization and explores challenges like transfer learning, data efficiency, and integrating prediction into RL

## [2] Supervised Learning of Behaviors

[[2 Supervised Learning of Behaviors]]
- Imitation learning uses supervised learning to mimic expert behavior, but issues like distribution mismatch and error accumulation over time complicate this approach
- Behavioral cloning, while effective, faces challenges such as non-Markovian and multimodal behavior, leading to solutions like sequence models, Gaussian mixture models, and latent variable models
- The DAgger algorithm improves imitation learning by continuously aggregating training data from the model's own actions, but it still relies heavily on human-provided data, which poses limitations

## [3] Pytorch Tutorial

See lecture 3 slides.

## [4] Introduction to Reinforcement Learning

[[4 Introduction to Reinforcement Learning]]
- A Markov Decision Process (MDP) formalizes decision-making, where actions influence state transitions and rewards, while partially observed MDPs add complexity by incorporating observations
- Q-learning aims to maximize expected rewards by optimizing a policy using Q-functions and value functions, which estimate the future reward of actions in specific states
- RL algorithms vary in approach: policy gradient methods optimize policies directly, value-based methods estimate value functions, and actor-critic models combine both, with trade-offs in sample efficiency and convergence stability

## [5] Policy Gradients

[[5 Policy Gradients]]
- Notes

## [6] Actor-Critic Algorithms

[[6 Actor-Critic Algorithms]]
- Notes
