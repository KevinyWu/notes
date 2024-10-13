# Learning Vision-based Pursuit-Evasion Robot Policies

**Authors**: Andrea Bajcsy, Antonio Loquercio, Ashish Kumar, Jitendra Malik

#supervised
#multi-agent
#planning

[[papers/multi-agent/README#[2023-08] Learning Vision-based Pursuit-Evasion Robot Policies|README]]

[Paper](http://arxiv.org/abs/2308.16185)
[Code](https://github.com/abajcsy/vision-based-pursuit/)
[Website](https://abajcsy.github.io/vision-based-pursuit/)

## Abstract

> Learning strategic robot behavior – like that required in pursuit-evasion interactions – under real-world constraints is extremely challenging. It requires exploiting the dynamics of the interaction, and planning through both physical state and latent intent uncertainty. In this paper, we transform this intractable problem into a supervised learning problem, where a fully-observable robot policy generates supervision for a partially-observable one. We find that the quality of the supervision signal for the partially-observable pursuer policy depends on two key factors: the balance of diversity and optimality of the evader's behavior and the strength of the modeling assumptions in the fully-observable policy. We deploy our policy on a physical quadruped robot with an RGB-D camera on pursuit-evasion interactions in the wild. Despite all the challenges, the sensing constraints bring about creativity: the robot is pushed to gather information when uncertain, predict intent from noisy measurements, and anticipate in order to intercept. Project webpage: <https://abajcsy.github.io/vision-based-pursuit/>

## Summary

- While there has been progress for robots acting in the wild, work has mostly been robots in isolation - they need to interact with one another
    - Must account for **uncertainty in other agents' future behavior**
    - Pursuit-evasion interactions: the pursuer gathers information about the hidden evader by turning and scanning the environment; upon detection, the pursuer has to continuously strategize about its next move without perfect knowledge of how the evader will react, all from onboard sensors.
- Privileged information depends not only on the robot, but also on the other agent's behavior, dictated by its intent
- Create a policy that automatically takes actions to resolve physical state uncertainty (e.g., looking around to see detect where the other agent is) while also generating predictions about other agents' intent to yield strategic behavior
- **First work to demonstrate autonomous interaction between a quadruped and another robotic or human agent truly in the wild**

## Background

- Partially observable stochastic games impossible to solve with game theory, too computationally intensive
- **Multi-agent reinforcement learning (MARL)** algorithms exploit large-scale simulation and neural network representations
- Assume evader's physical state, latent intent, and action are all hidden: **only observation is onboard RGB-D and proprioception**
- Use **privileged learning**: model has access to additional information during training but not deployment (i.e. different inputs)
    - Privileged data guides the model to focus on relevant patterns to learn faster and more effectively
    - RL was unsuccessful under partial observability

## Method

- **Privileged learning** ![[pursuit.png]]
- Given an evader policy $\pi^e$, we seek a pursuer policy $\pi^P$ which maximizes expected cumulative reward over the set of trajectories $\tau = \{(x_o^p, x_0^e,u_0^e,o_0^p,o_0^e,r_0)\dots (x_T^p, x_T^e,u_T^e,o_T^p,o_T^e,r_T)\}$
    - $x,u,o,r$ are states, actions, observations, and rewards
    - Reward is distance between two agents at each timestep, plus additional termination bonus upon capture
    - All agents reason about **relative state** of their own body frame
- Three models tested for evader policy: random motion, multi-agent RL (MARL), dynamic game theory
- **Teacher**
    - Fully observable policy $\pi^*$
    - Access to true pursuer relative state and future $N$ states of evader
    - Future trajectory encoded into low-dim latent $z_t$ that represents the **near-term behavior of the evader**
- **Student**
    - Partially observable policy $\pi^p$
    - Pursuer estimates relative state of the evader from the output of a 3D object detector using RGB camera
    - Use **Kalman Filter** to generate estimated relative state of the evader and uncertainty $\hat{\Sigma}_t$
    - **History of relative state estimates and pursuer actions are encoded into low-dim latent $\hat{z}_t$**
        - This is an "estimate" of the intent of the evader using only noisy, real-world data

## Results

- **Simulated results**
    - Experiment: make the evader policy highly predictable
        - The pursuer should be able to intercept evader if it has high-quality understanding of evader's latent intent
        - Supervision using privileged future evader trajectories converges 10x faster than policies that only look at the present or the past - hence the teacher policy architecture
    - Effect of evader model
        - Random and MARL similar capture time, game theory much slower
- **Real-world results**
    - Out-of-distribution because behavior of evader is unscripted
    - Deployed all three policies on real robot
    - MARL pursuer policy is fastest, while game-theory pursuer policy is more inefficient in information-seeking motions
