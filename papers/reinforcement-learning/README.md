# Reinforcement Learning

## [2013-12] Playing Atari with Deep Reinforcement Learning

#reinforcement-learning
#q-learning
[[Playing Atari with Deep Reinforcement Learning]]
- Uses CNNs with Q-learning to approximate the optimal action-value function, learning control policies directly from raw pixel input in Atari games
- Stores agent experiences in a replay buffer and samples random minibatches for training, breaking the correlation between consecutive experiences and improving data efficiency
- Uses an $\epsilon$-greedy policy that randomly selects actions with probability $\epsilon$ and otherwise selects the action that maximizes the Q-value, balancing exploration and exploitation during training

## [2021-07] Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning

#reinforcement-learning
#learning-from-video
[[Mastering Visual Continuous Control - Improved Data-Augmented Reinforcement Learning]]
- DrQ-v2 improves upon DrQ, using data augmentation to enable model-free RL from pixels
- Techniques like random shift augmentation, double Q-learning, and scheduled exploration noise enhance performance and stability in visual continuous control tasks
- Key improvements include a larger replay buffer, smaller learning rate, and faster training times (3x faster than DrQ)

## [2022-09] VIP: Towards Universal Visual Reward and Representation via Value-Implicit Pre-Training

#reinforcement-learning
#inverse-reinforcement-learning
#learning-from-video
[[VIP - Towards Universal Visual Reward and Representation via Value-Implicit Pre-Training]]
- VIP leverages large-scale, offline human videos for pre-training a visual representation that generates dense rewards for unseen robotic tasks, without requiring task-specific robotic data
- Formulates reward learning as a goal-conditioned RL problem using a dual value-function approach, avoiding the need for action labels and enabling effective learning from action-free human videos

## [2022-11] Learning Reward Functions for Robotic Manipulation by Observing Humans

#reinforcement-learning
#inverse-reinforcement-learning
#learning-from-video
[[Learning Reward Functions for Robotic Manipulation by Observing Humans]]
- HOLD which learns task-agnostic reward functions for robotic manipulation by observing human videos, without needing robot-specific data or predefined human-robot correspondences
- Two methods for learning state distances relative to a goal image: direct temporal regression (HOLD-R) and time-contrastive learning (HOLD-C)

## [2023-12] Diffusion Reward: Learning Rewards via Conditional Video Diffusion

#reinforcement-learning
#inverse-reinforcement-learning
#learning-from-video
#diffusion
[[Diffusion Reward - Learning Rewards via Conditional Video Diffusion]]
- Learns dense rewards for RL from expert videos by leveraging conditional video diffusion models, capturing expert behavior through generative diversity
- The approach formalizes rewards as negative conditional entropy, where lower generative diversity is associated with expert-like trajectories, promoting better exploration and task performance

## [2024-04] Rank2Reward: Learning Shaped Reward Functions from Passive Video

#reinforcement-learning
#inverse-reinforcement-learning
#learning-from-video
[[Rank2Reward - Learning Shaped Reward Functions from Passive Video]]
- Learns reward functions from passive videos by ranking frames based on progress towards a task goal, enabling reinforcement learning without state-action data
- Rank2Reward learns an ordering space that both encodes progress towards a goal and is agnostic to time required to reach the state
