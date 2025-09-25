# A Comparison of Imitation Learning Algorithms for Bimanual Manipulation

**Authors**: Michael Drolet, Simon Stepputtis, Siva Kailas, Ajinkya Jain, Jan Peters, Stefan Schaal, Heni Ben Amor

[[papers/imitation-learning/README#[2024-08] A Comparison of Imitation Learning Algorithms for Bimanual Manipulation|README]]

[Paper](http://arxiv.org/abs/2408.06536)
[Code](https://github.com/ir-lab/bimanual-imitation)
[Website](https://bimanual-imitation.github.io/)

## Abstract

> Amidst the wide popularity of imitation learning algorithms in robotics, their properties regarding hyperparameter sensitivity, ease of training, data efficiency, and performance have not been well-studied in high-precision industry-inspired environments. In this work, we demonstrate the limitations and benefits of prominent imitation learning approaches and analyze their capabilities regarding these properties. We evaluate each algorithm on a complex bimanual manipulation task involving an over-constrained dynamics system in a setting involving multiple contacts between the manipulated object and the environment. While we find that imitation learning is well suited to solve such complex tasks, not all algorithms are equal in terms of handling environmental and hyperparameter perturbations, training requirements, performance, and ease of use. We investigate the empirical influence of these key characteristics by employing a carefully designed experimental procedure and learning environment. Paper website: <https://bimanual-imitation.github.io/>

## Summary

- RL is hard for manipulation, need good rewards for all areas of environment
- Imitation learning (specifically BC) does not need an explicit reward function
    - Drawback is that IL cannot discover out-of-distribution solutions
- Diffusion Policy and ACT perform the best

## Background

- **Imitation learning is learning from experts**
- Early work in bimanual manipulation utilized classical control-based approaches
- More recently: learning-based approaches
    - Reinforcement learning
    - ALOHA with Action Chunking Transformer

## Method

- Algorithms
    - ** Vanilla Behavioral cloning** uses the following objective: $\hat{\theta} = \arg\max_{\theta}E_{(s,a)\sim \tau_E}[\log(\pi_{\theta}(a|s))]$
        - **Find a policy that maximizes the log probability of taking an expert action given a state, across all expert trajectories (action-state pairs)**
        - $\pi_{\theta}$ - policy
        - $\tau_E$ - expert trajectories
        - $a$ - action
        - $s$ - state
    - **Action Chunking Transformer (ACT)** performs behavioral cloning using a conditional variational autoencoder (CVAE) implemented as a multi-headed attention transformer
    - **Implicit Behavioral Cloning (IBC)** is is trained using the Negative Counter Example (NCE) loss function, such that negative counter-examples of the expert are generated to train the model
    - **Diffusion Policy** performs an iterative procedure to generate actions using a series of denoising steps
        - Refine noise into actions via a learned gradient field
    - **GAIL** formulates imitation learning as an inverse reinforcement learning (IRL) problem, wherein the reward function is learned based on the discriminator's scores
        - Interacts with environment
        - The generator network (i.e., the policy) tries to produce state-action pairs that match the expert's
        - Stop when discriminator cannot distinguish between policy and expert
    - **DAgger** addresses the covariate shift problem, where the distribution of observations the policy encounters differs from those in the expert dataset
        - Interacts with environment
- Environment
    - Two UR5 arms on a rotating torso
    - 18-dim action space
    - Environment reward used to measure algorithm's success - not used in training
    - Hyperparameter search to find best model

## Results

- Metrics
    - Hyperparameter tolerance
    - Noise tolerance (to adding noise to action)
    - Compute efficiency: time needed to train
    - Performance: success % in zero-noise scenario
- GAIL, Diffusion, BC are closest to the expert in Wasserstein distance
- Interaction with environment (GAIL, DAgger) and action chunking (ACT, Diffusion) more robust to noise
    - Action and observation horizons introduced by Diffusion and ACT help cope with potentially non-Markovian environments.
