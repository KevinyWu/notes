# Real-World Robot Learning

Notes from the course "CS 7000: Real World Robot Learning" taught by Antonio Loquercio and Dinesh Jayaraman at the University of Pennsylvania (Spring 2025).

[Course website](https://antonilo.github.io/real_world_robot_learning_sp25/)

## Contents

- [[courses/real-world-robot-learning/README#[1] Introduction|1 Introduction]]
- [[courses/real-world-robot-learning/README#[2] Intro To Imitation Learning & Reinforcement Learning Part I|2 Intro To Imitation Learning & Reinforcement Learning Part I]]
- [[courses/real-world-robot-learning/README#[3] Intro To Imitation Learning & Reinforcement Learning Part II|3 Intro To Imitation Learning & Reinforcement Learning Part II]]
- [[courses/real-world-robot-learning/README#[4] Intro To Imitation Learning & Reinforcement Learning Part III|4 Intro To Imitation Learning & Reinforcement Learning Part III]]
- [[courses/real-world-robot-learning/README#[5] Hands-on Tutorial on Policy Learning (Lab)|5 Hands-on Tutorial on Policy Learning (Lab)]]
- [[courses/real-world-robot-learning/README#[6] Robot Perception I|6 Robot Perception I]]

## [1] Introduction

#robotics
[[courses/real-world-robot-learning/1 Introduction|1 Introduction]]
- Moravec's Paradox: things that are easy for computers are hard for humans and vice versa
- We don't have the right data, enough data, or the right objectives
- Challenges in real-world robot learning: safety, computational efficiency, stuff breaks

## [2] Intro To Imitation Learning & Reinforcement Learning Part I

#imitation-learning
#reinforcement-learning
[[2 Intro To Imitation Learning & Reinforcement Learning Part I]]
- Robots learn through perception-action loops, acquiring policies, dynamics models, reward functions, and state representations to interact with their environment
- Reinforcement learning maximizes expected rewards in an unknown environment through exploration, while imitation learning optimizes policies by mimicking expert demonstrations
- Behavioral cloning trains policies via supervised learning but suffers from compounding errors, requiring techniques like DAGGER to iteratively refine the policy with expert corrections

## [3] Intro To Imitation Learning & Reinforcement Learning Part II

#imitation-learning
#reinforcement-learning
[[3 Intro To Imitation Learning & Reinforcement Learning Part II]]
- Inverse reinforcement learning improves upon behavioral cloning by explicitly learning the reward function, making policies more generalizable beyond expert demonstrations
- Reinforcement learning optimizes policies by balancing exploration and exploitation, solving problems where correct actions aren't explicitly labeled, unlike supervised learning
- Policy gradient methods optimize expected rewards by directly updating policies using sampled trajectories, with rewards often estimated via a Q-function

## [4] Intro To Imitation Learning & Reinforcement Learning Part III

#imitation-learning
#reinforcement-learning
[[4 Intro To Imitation Learning & Reinforcement Learning Part III]]

## [5] Hands-on Tutorial on Policy Learning (Lab)

[IL Tutorial Colab](https://colab.research.google.com/github/antonilo/real_world_robot_learning_sp25/blob/main/_tutorials/lerobot_tutorial/lerobot_tutorial.ipynb)
[RL Tutorial Colab](https://colab.research.google.com/drive/1p1DkWle2Iwcjnq2ClJp_bN1AzIGg4dBq?usp=sharing)

## [6] Robot Perception I

#tags
[[6 Robot Perception I]]
