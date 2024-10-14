# Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning

**Authors**: Denis Yarats, Rob Fergus, Alessandro Lazaric, Lerrel Pinto

#reinforcement-learning
#learning-from-video

[[papers/reinforcement-learning/README#[2021-07] Mastering Visual Continuous Control Improved Data-Augmented Reinforcement Learning|README]]

[Paper](http://arxiv.org/abs/2107.09645)
[Code](https://github.com/facebookresearch/drqv2)

## Abstract

> We present DrQ-v2, a model-free reinforcement learning (RL) algorithm for visual continuous control. DrQ-v2 builds on DrQ, an off-policy actor-critic approach that uses data augmentation to learn directly from pixels. We introduce several improvements that yield state-of-the-art results on the DeepMind Control Suite. Notably, DrQ-v2 is able to solve complex humanoid locomotion tasks directly from pixel observations, previously unattained by model-free RL. DrQ-v2 is conceptually simple, easy to implement, and provides significantly better computational footprint compared to prior work, with the majority of tasks taking just 8 hours to train on a single GPU. Finally, we publicly release DrQ-v2's implementation to provide RL practitioners with a strong and computationally efficient baseline.

## Summary

- DrQ-v2 builds on the idea of using data augmentations
- First model-free method to solve complex humanoid tasks directly from pixels
- More computationally efficient than previous methods like DreamerV2
    - Trains 3x faster than DrQ

## Background

- RL from images
    - Stack three consecutive prior observations to approximate the current state of the system
    - MDP can be described as a tuple $(\mathcal{X}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma, d_0)$
        - $\mathcal{X}$ is the state space
        - $\mathcal{A}$ is the action space
        - $\mathcal{P}:\mathcal{X}\times \mathcal{A} \rightarrow \Delta (\mathcal{X})$ is the transition function that defines a probability distribution over the next state given current state and action
        - $\mathcal{R}:\mathcal{X}\times \mathcal{A} \rightarrow [0,1]$ is the reward function
        - $\gamma \in [0,1)$ is the discount factor
        - $d_0 \in \Delta (\mathcal{X})$ is the initial state distribution
    - Goal: find a policy $\pi:\mathcal{X}\rightarrow \Delta(\mathcal{A})$ that maximizes the expected discounted sum of rewards $E_{\pi}[\sum_{t=0}^{\infty}]$
    - $x_0 \sim d_0$
    - $\forall t, a_t \sim \pi(\cdot|x_t), x_{t+1} \sim \mathcal{P}(\cdot|x_t, a_t), r_t = \mathcal{R}(x_t, a_t)$
- **Deep Deterministic Policy Gradient (DDPG)**
    - Actor-critic algorithm for continuous control that concurrently learns a Q-function $Q_{\theta}$ and a deterministic policy $\pi_{\phi}$
    - Uses Q-learning to learn the Q-function by minimizing the Bellman error
    - The policy is learned by employing Deterministic Policy Gradient (DPG) algorithm and maximizing the value of the Q-function

## Method

- Random shift augmentation
    - First pad each side of the $84\times 84$ image by 4 pixels, repeating the boundary pixel, then selecting a random $84\times 84$ crop
    - Apply bilinear interpolation on top of the shifted image (replace each pixel with a weighted average of the 4 nearest pixels)
- DrQ-v2 method ![[drqv2.png]]
- Image encoder
    - Embeds augmented image into low-dimensional latent vector
    - Convolutional encoder $f_{\zeta}$
    - $h_t = f_{\zeta}(\text{aug}(x_t))$
- Actor-Critic algorithm
    - DDPG as the actor-critic RL backbone
    - Augment it with $n$-step returns to estimate TD (temporal difference) error
    - Double Q-learning to reduce overestimation bias
        - Train two Q-functions $Q_{\theta_1} and Q_{\theta_2}$
    - **Loss function for training deterministic actor** $\pi_{\phi}$: $\mathcal{L}_{\phi}(\mathcal{D}) = -E_{x_t\sim \mathcal{D}}[\min_{k=1,2} Q_{\theta_k}(h_t, a_t)]$
        - $h_t = f(\text{aug}(x_t))$
        - $a_t = \pi_{\phi}(h_t) + \epsilon$
        - **Exploration noise** $\epsilon \sim \text{clip}(\mathcal{N}(0, \sigma), -c, c)$
        - Trained using DGP
        - Do not use actor's gradients to update encoder's parameters $\zeta$
- Scheduled exploration noise
    - At the beginning of training we want the agent to be more stochastic and explore the environment more effectively
    - At the later stages of training, when the agent has already identified promising behaviors, it is better to be more deterministic and master those behaviors
    - Linear decay $\sigma(t)$ for the variance of the exploration noise
    - $\sigma(t) = \sigma_{init} + (1 - \min(\frac{t}{T}, 1))(\sigma_{final} - \sigma_{init})$
        - $T$ is the decay horizon
- Hyperparameters
    - Three most important: size of replay buffer, minibatch size, learning rate
    - Compared to DrQ
        - Larger replay buffer
        - Smaller minibatch size
        - Smaller learning rate: $10^{-4}$ instead of $10^{-3}$

## Results

- DDPG demonstrates better exploration properties than SAC
- Larger replay buffer (1M) helps prevent the catastrophic forgetting problem, especially in tasks with more diverse initial state distributions
    - **Replay buffer (AKA experience replay) stores past experiences `(state, action, reward, next state, done)`**
        - During the update phase, a batch of experiences is sampled from the replay buffer and used to update the Q-function and policy
    - Benefits
        - Data efficiency: learn more from less interactions with the environment
        - Stability: provides set of relatively uncorrelated experiences rather than consecutive experiences
        - Off-policy learning: can learn from experiences generated by a different policy
