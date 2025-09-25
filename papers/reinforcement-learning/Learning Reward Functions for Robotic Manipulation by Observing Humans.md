# Learning Reward Functions for Robotic Manipulation by Observing Humans

**Authors**: Minttu Alakuijala, Gabriel Dulac-Arnold, Julien Mairal, Jean Ponce, Cordelia Schmid

[[papers/reinforcement-learning/README#[2022-11] Learning Reward Functions for Robotic Manipulation by Observing Humans|README]]

[Paper](http://arxiv.org/abs/2211.09019)
[Code](https://github.com/minttusofia/hold-rewards)
[Website](https://sites.google.com/view/hold-rewards)
[Video](https://www.youtube.com/watch?v=t6g3Em4HDwI)

## Abstract

> Observing a human demonstrator manipulate objects provides a rich, scalable and inexpensive source of data for learning robotic policies. However, transferring skills from human videos to a robotic manipulator poses several challenges, not least a difference in action and observation spaces. In this work, we use unlabeled videos of humans solving a wide range of manipulation tasks to learn a task-agnostic reward function for robotic manipulation policies. Thanks to the diversity of this training data, the learned reward function sufficiently generalizes to image observations from a previously unseen robot embodiment and environment to provide a meaningful prior for directed exploration in reinforcement learning. We propose two methods for scoring states relative to a goal image: through direct temporal regression, and through distances in an embedding space obtained with time-contrastive learning. By conditioning the function on a goal image, we are able to reuse one model across a variety of tasks. Unlike prior work on leveraging human videos to teach robots, our method, Human Offline Learned Distances (HOLD) requires neither a priori data from the robot environment, nor a set of task-specific human demonstrations, nor a predefined notion of correspondence across morphologies, yet it is able to accelerate training of several manipulation tasks on a simulated robot arm compared to using only a sparse reward obtained from task completion.

## Summary

- We propose two methods for scoring states relative to a goal image: through direct temporal regression, and through distances in an embedding space obtained with time-contrastive learning
- By conditioning the function on a goal image, we are able to reuse one model across a variety of tasks
- **Use of videos of people solving manipulation tasks to learn a notion of distance between images from the observation space of a task**
- The learned distance function captures roughly how long it takes for an expert to transition from one state to another
- Model-free RL typically requires extensive data collection in robot action space
    - Propose to learn a state-value function from observation-only data

## Background

- Functional distances from observation-only data
    - **Goal is to learn functional distance $d(s, g)$ between image $s$ of current state and goal image $g$**
    - This should correlate with $\delta (s, g)$, the number of time steps it takes for an expert $\pi^*$ to reach goal $g$ from the state $s$
- We assume access to a dataset of $N$ video demonstrations of humans executing a variety of manipulation tasks using approximately shortest paths
    - Although the absolute length of such time intervals may not be consistent across demonstrators, their relative durations provide a useful learning signal

## Method

- Two methods for learning $d$ from this data ![[hold.png]]
- **Direct regression (HOLD-R)**
    - $\theta^* = \arg\min \sum_{i=1}^N \sum_{t=1}^{T_i} \sum_{\delta = 1}^{T_i - t} \|d_{\theta}(s^i_t, s^i_{t+\delta}) - \delta \|^2_2$
    - $s_t^i$ is the image at time $t$ in video $i$
    - $T_i$ is the length of video $i$
    - $d_{\theta}$ is the learned distance function trained to predict $\delta$
    - The third summation corresponds to data augmentation allowing any future time step in the video to be considered the goal rather than only the last
- **Time-contrastive embeddings (HOLD-C)**
    - Directly predicting time intervals can be difficult and sensitive to noise
    - Use single-view time-contrastive objective like [TCN](#apr-2017-tcn-time-contrastive-networks-self-supervised-learning-from-video) to learn an embedding space where the distance between two images is proportional to the time it takes to transition between them
    - Advantages over TCN
        - HOLD enables the robot to outperform the human demonstrator while TCN tries to mimic the demonstrator 1:1
        - HOLD requires less supervision: only the goal image, not the full trajectory
        - HOLD uses simpler Euclidean distance in the embedding space rather than mixture of Euclidean and Huber style loss
- Policy learning
    - **Although the reward function is goal conditioned and shared across tasks, still learn a separate policy for each task**
    - Reward function: $r(s_t, a_t, s_{t+1}, g) = -\max (0, d(s_{t+1}, g) - d(g, g))/T$
    - Subtracting $d(g,g)$ ensures that the reward is zero when the goal is reached (this value may be positive due to untrimmed training videos) and no other state has higher reward

## Results

- Dataset: Something-Something V2 (220k video clips of 174 human action classes)
- Training details
    - Two sizes of network: ResNet-50 and Video Vision Transformer (ViViT)
    - For HOLD-C, pretrain on ImageNet-21k
- Policy learning
    - Pushing an drawer opening task from RLV
    - Close drawer, push cup forward, turn faucet right task from DVD
    - Soft Actor-Critic as the RL algorithm
    - **Augment learned reward with sparse task reward, 1 for success, 0 otherwise**
        - This improves the base learned reward function
- Outperforms TCN and R3M rewards on both RLV tasks
