# Diffusion Reward: Learning Rewards via Conditional Video Diffusion

**Authors**: Tao Huang, Guangqi Jiang, Yanjie Ze, Huazhe Xu

#reinforcement-learning
#inverse-reinforcement-learning
#learning-from-video
#diffusion

[[papers/reinforcement-learning/README#[2023-12] Diffusion Reward Learning Rewards via Conditional Video Diffusion|README]]

[Paper](http://arxiv.org/abs/2312.14134)
[Code](https://github.com/TEA-Lab/diffusion_reward)
[Website](https://diffusion-reward.github.io/)

## Abstract

> Learning rewards from expert videos offers an affordable and effective solution to specify the intended behaviors for reinforcement learning (RL) tasks. In this work, we propose Diffusion Reward, a novel framework that learns rewards from expert videos via conditional video diffusion models for solving complex visual RL problems. Our key insight is that lower generative diversity is exhibited when conditioning diffusion on expert trajectories. Diffusion Reward is accordingly formalized by the negative of conditional entropy that encourages productive exploration of expert behaviors. We show the efficacy of our method over robotic manipulation tasks in both simulation platforms and the real world with visual input. Moreover, Diffusion Reward can even solve unseen tasks successfully and effectively, largely surpassing baseline methods. Project page and code: <https://diffusion-reward.github.io>.

## Summary

- Designing dense rewards for RL is hard
    - Sparse rewards less effort but worse performance
    - Solution: learn from expert (but unlabeled) videos
- Diffusion Reward leverages conditional video diffusion models to capture the expert video distribution and extract dense rewards
    - Our key insight is that higher generative diversity is observed when conditioned on expert-unlike videos, while lower given expert videos.
- **Generative models can extract rewards unsupervisedly without future information like goal frame**

## Background

- Finite-horizon MDP: $(\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma)$
    - $\mathcal{S}$ is the state space
    - $\mathcal{A}$ is the action space
    - $\mathcal{T}$ is the transition function
    - $\mathcal{R}$ is the reward function
    - $\gamma$ is the discount factor
- Goal: learn a policy $\pi$ that maximizes the expected return $J(\pi) = E_{\tau\sim\pi}[\sum_{t=0}^T\gamma^t r(s_t, a_t)]$
- Expert videos: $\mathcal{D} = \{\mathcal{D}^1, \mathcal{D}^2, \ldots, \mathcal{D}^N\}$
    - Each $\mathcal{D}^i$ is a set for demonstrated videos from task $i$, containing multiple expert trajectories $\tau^i$
- **Diffusion model**: probabilistic models that aim to model data distribution by gradually denoising a normal distribution through a reverse diffusion process
    - Latent diffusion process
        - Train unsupervised encoder from expert videos to compress high-dimensional observation with VQ-GAN
    - Condition on historical frames to utilize temporal information
        - Can be viewed as matching the distribution of expert and agent trajectories

## Method

- High level: leverage entropy information from video diffusion models pre-trained on expert videos to encourage RL agents to explore expert-like trajectories more ![[diffusion_reward.png]]
- Conditional entropy as rewards
    - Previous studies like VIPER use log-likelihood as rewards
        - This struggles with accurately modeling the expert distribution with intricate dynamics
        - Out-of distribution learned rewards drop
    - **Diffusion reward key idea: increased generation diversity with unseen historical observations, reduced with seen ones**
        - Diffusion conditioned on expert-like trajectories exhibits lower diversity where the agent ought to be rewarded more and vice versa
        - Estimate negative conditional entropy given historical frames
    - **Diffusion reward**: $r^{\text{diff}} = (1-\alpha)\bar{r}^{\text{ce}} + \alpha\bar{r}^{\text{rnd}} + r^{\text{spar}}$
        - $\bar{r}^{\text{ce}}$ is the conditional entropy reward
        - $\bar{r}^{\text{rnd}}$ is the random reward to encourage exploration
        - $r^{\text{spar}}$ is the raw sparse reward to guide the agent to the goal

## Results

- 7 tasks from MetaWorld, 3 from Adroit
- Baselines
    - Raw sparse reward: uses environment's sparse reward
    - [RND](https://arxiv.org/abs/1810.12894), [AMP](https://arxiv.org/abs/2104.02180), [VIPER](https://arxiv.org/abs/2305.14343)
- All methods use DrQv2 as the RL backbone
