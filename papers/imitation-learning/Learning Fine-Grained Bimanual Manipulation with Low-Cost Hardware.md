# Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware

**Authors**: Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn

[[papers/imitation-learning/README#[2023-04] Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware|README]]

[Paper](http://arxiv.org/abs/2304.13705)
[Code](https://github.com/tonyzhaozh/act)
[Website](https://tonyzhaozh.github.io/aloha/)

## Abstract

> Fine manipulation tasks, such as threading cable ties or slotting a battery, are notoriously difficult for robots because they require precision, careful coordination of contact forces, and closed-loop visual feedback. Performing these tasks typically requires high-end robots, accurate sensors, or careful calibration, which can be expensive and difficult to set up. Can learning enable low-cost and imprecise hardware to perform these fine manipulation tasks? We present a low-cost system that performs end-to-end imitation learning directly from real demonstrations, collected with a custom teleoperation interface. Imitation learning, however, presents its own challenges, particularly in high-precision domains: errors in the policy can compound over time, and human demonstrations can be non-stationary. To address these challenges, we develop a simple yet novel algorithm, Action Chunking with Transformers (ACT), which learns a generative model over action sequences. ACT allows the robot to learn 6 difficult tasks in the real world, such as opening a translucent condiment cup and slotting a battery with 80-90% success, with only 10 minutes worth of demonstrations. Project website: <https://tonyzhaozh.github.io/aloha/>

## Summary

- Behavioral cloning
    - Compounding error of imitation learning, even with high-quality demonstrations
    - Compounding error: errors from previous timesteps accumulate and cause robot to drift
- Solve this by "action chunking": seqeunces of actions grouped together and executed as a single unit
- **Action Chunking Transformer (ACT)**: a transformer-based model that learns to chunk actions
    - Trained as a conditional VAE

## Background

- Data collection
    - Collect human demonstrations, recording joint positions of leader robots as actions
    - Important to use leader joints because force applied is implicitly defined by the difference between them, through the PID controller inside the Dynamixel servos
- **Action chunking**: fix the chunk size to be $k$
    - Every $k$ steps, the agent receives an observation, generates the next $k$ actions, and executes them
    - Chunking helps model non-Markovian behavior (Markovian is when future state can only depend on current state) compared to single-step action prediction
    - Naive chunking can result in jerky motion
        - Solution: query the policy at every timestep (generating overlapping chunks)
        - Temporal ensemble: weighted average of the predictions from the overlapping chunks, where $w_i = \exp (-m * i)$, $w_0$ is the weight of the oldest action
        - Smaller $m$ means faster incorporation of new observations

## Method

- **Conditional variational autoencoder (CVAE)** generates action sequence conditioned on current observations
    - **CVAE encoder**
        - Only servers to train decoder, discarded at test time
        - Predicts mean and variance of the style variable $z$
        - Implemented as BERT-like transformer encoder
        - Input: current joint positions and target action sequence of length $k$ from the demonstration dataset, prepended with [CLS] token
        - Output: feature corresponding to [CLS] is used to predict mean and variance of style variable $z$
    - **CVAE decoder (the policy)**
        - Predicts the action sequence conditioned on both $z$ and the current observations (images + joint positions)
        - Implemented with ResNet image encoder, transformer encoder, and transformer decoder
        - ResNet processes images and flatens along spatial dimensions; we add position embeddings to preserve spatial information
        - Also add current joint positions and style variable $z$ to the input
        - Transformer encoder synthesizes information from this input
        - Transformer decoder generates coherent action sequence
        - **Action space is vector of joint angles, so total output (the "action sequence") is $k\times n$ tensor where $n$ is the number of joints**
- Maximize log-likelihood of the demonstration action chunks: $\min_{\theta} -\sum_{s_t, a_{t:t+k}\in D} \log \pi_{\theta}(a_{t:t+k}|s_t)$
- Standard VAE objective
- Train from scratch for each task
- Training architecture ![[act_train.png]]
- Rollout architecture ![[act_test.png]]

## Results

- Outperforms BC-ConvMLP, BeT, RT-1, and VINN on 6 tasks
- Collect 10-20 minutes of data (50 demonstrations) for each task
- Ablation study
    - Chunking around $k=100$ is best
    - Temporal ensemble improves performance only slightly, by 3.3%
    - Training with CVAE is crucial
        - Variational autoencoder is a generative model, it learns a compact representation of the input data and learns how to generate new data similar to the input data
        - Conditional VAE is a VAE that is conditioned on some input data
        - Predictions based on style variable $z$ and current observations, allowing the model to generate different actions depending on the environment
        - Human teleoperated demonstrations can be noisy and multi-modal (human might perform the same task in different ways), so style variable $z$ helps the model understand the context better
        - With scripted data, CVAE did not help, but with human demonstrations, CVAE success rate was 35% vs 2% without CVAE
