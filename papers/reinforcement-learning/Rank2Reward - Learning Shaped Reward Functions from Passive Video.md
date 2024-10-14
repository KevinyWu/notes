# Rank2Reward: Learning Shaped Reward Functions from Passive Video

**Authors**: Daniel Yang, Davin Tjia, Jacob Berg, Dima Damen, Pulkit Agrawal, Abhishek Gupta

#reinforcement-learning
#inverse-reinforcement-learning
#learning-from-video

[[papers/reinforcement-learning/README#[2024-04] Rank2Reward Learning Shaped Reward Functions from Passive Video|README]]

[Paper](http://arxiv.org/abs/2404.14735)
[Code](https://github.com/dxyang/rank2reward/)
[Website](https://rank2reward.github.io/)

## Abstract

> Teaching robots novel skills with demonstrations via human-in-the-loop data collection techniques like kinesthetic teaching or teleoperation puts a heavy burden on human supervisors. In contrast to this paradigm, it is often significantly easier to provide raw, action-free visual data of tasks being performed. Moreover, this data can even be mined from video datasets or the web. Ideally, this data can serve to guide robot learning for new tasks in novel environments, informing both "what" to do and "how" to do it. A powerful way to encode both the "what" and the "how" is to infer a well-shaped reward function for reinforcement learning. The challenge is determining how to ground visual demonstration inputs into a well-shaped and informative reward function. We propose a technique Rank2Reward for learning behaviors from videos of tasks being performed without access to any low-level states and actions. We do so by leveraging the videos to learn a reward function that measures incremental "progress" through a task by learning how to temporally rank the video frames in a demonstration. By inferring an appropriate ranking, the reward function is able to guide reinforcement learning by indicating when task progress is being made. This ranking function can be integrated into an adversarial imitation learning scheme resulting in an algorithm that can learn behaviors without exploiting the learned reward function. We demonstrate the effectiveness of Rank2Reward at learning behaviors from raw video on a number of tabletop manipulation tasks in both simulations and on a real-world robotic arm. We also demonstrate how Rank2Reward can be easily extended to be applicable to web-scale video datasets.

## Summary

- Challenges in inverse RL
    - Requires demonstration data with state-action tuples
    - Learned reward functions may explain expert data well but not be "well-shaped" for exploration
- **Key assumption: video demonstrations typically make monotonic progress towards a goal**
- Ranking function to predict the relative progress between two video frames
- Ranking function trained on only expert data
    - To account for out-of-distribution data, train a separate discriminator model during policy rollout to differentiate expert and non-expert data
- Limitations and future work
    - Embodiment shift between human demonstrations and robot manipulators
    - Rewards are trained on single-task, and it would be hard to have a different reward and agent for every task
    - Classifier $D_{\phi}$ is sensitive to changes in the background and dynamic scenes

## Background

- Inverse RL
    - Reward functions poorly shaped
    - Rank2Reward learns a well-shaped reward function
- Imitation from observation
    - Time-Contrastive Networks (TCN) learns representation from contrastive learning across time and viewpoints
        - Their embedding space does not contain a notion of progress towards a goal
        - They rely on tracking a specific expert trajectory which requires temporal alignment
    - **Rank2Reward learns an ordering space that both encodes progress towards a goal and is agnostic to time required to reach the state**

## Method

- Learning a measure of progress by ranking ![[rank2reward.png]]
    - True reward (unknown) $r(s, a) > \epsilon$ where $\epsilon > 0$ for all tasks
    - Progress along a trajectory can be measured by simply learning a function that can rank different image frames in a trajectory according to their temporal ordering
    - Learn utility function $\hat{u}(s)$ of states
        - Bradley-Terry model aims to learn this utility function suck that the likelihood of preferring a state $s^k_i$ over $s^k_j$ for some expert trajectory $\tau_k$ is given by $p(s_i^k > s_j^k) = \frac{\exp(\hat{u}(s_i^k))}{\exp(\hat{u}(s_i^k)) + \exp(\hat{u}(s_j^k))}$
        - In expert dataset $\mathcal{D}^e$, along $\tau_k$, $s_i^k$ is preferred to $s_j^k$ if it occurs later (i.e. $i > j$, so $\hat{u}(s_i^k) > \hat{u}(s_j^k)$)
    - Setting $\hat{u}(s_0) = 0$, $p(s>s_0) = \frac{1}{1 + \exp(-\hat{u}(s))}$
        - **Denote this "likelihood of making progress" as $p_{RF}(s)$**
    - Learns a monotonically-increasing utility function
- Incorporating learned rankings into policy optimization
    - Since the reward function $\hat{r}(s)$ has only been learned on expert dataset, it may overestimate rewards at other states leading to incorrect policies
    - **Pessimistic policy objective:** $\max_{\pi} E_{s\sim d^{\pi}, a\sim \pi(a|s)}[\log p_{RF}(s)] - \alpha D_{KL}(d^{\pi}(s), d^{e}(s))$
        - The state marginal distribution $d^{\pi}(s)$ gives the probability of being in state $s$ when following policy $\pi$
        - First part aims to maximize likelihood of progress
        - Second part is a KL divergence between the state distribution of the policy and the expert
        - $D_{KL}(d^{\pi}(s), d^{e}(s)) = E_{s\sim d^{\pi}}\left [\log \frac{d^{\pi}(s)}{d^{e}(s)}\right ]$
        - Objective to make this divergence small, which means the policy $d^{\pi}(s)$ is close to the expert $d^{e}(s)$
    - $d^e(s)$ and $d^{\pi}(s)$ are not known, so a classifier $D_{\phi}(s)$ is trained to distinguish between the expert and policy states, providing $\frac{d^{\pi}(s)}{d^{e}(s)} \approx \frac{D_{\phi}(s)}{1-D_{\phi}(s)}$
- **Thus, final objective is** $\max_{\pi} E_{s, a\sim d^{\pi}}\left [\log \left (p_{RF}(s)\left (\frac{D_{\phi}(s)}{1-D_{\phi}(s)} \right )^{\alpha} \right ) \right ]$
    - In training, alternate between:
        - Training classifier $D_{\phi}$ between states in expert video vs. on-policy data
        - Perform policy obtimization with the classifier $D_{\phi}$ and the learned ranking function $p_{RF}$ to get the reward $\hat{r}(s) = \log p_{RF}(s) + \alpha\left (\log (D_{\phi}(s)) - \log (1-D_{\phi}(s)) \right )$
    - Can learn the ranking component $p_{RF}(s)$ offline with only expert data, independent of the policy

## Results

- Simulated experiments
    - [Meta-world simulator](https://meta-world.github.io/)
    - Baselines
        - [GAIL](https://arxiv.org/abs/1606.03476), [AIRL](https://arxiv.org/abs/1710.11248), [VICE](https://arxiv.org/abs/1805.11686), [SOIL](https://arxiv.org/abs/2004.04650), [TCN](https://arxiv.org/abs/1704.06888), [ROT](https://arxiv.org/abs/2206.15469)
        - Ranking only: ablation study with only ranking function, no adversarial training
  - Real-world experiments
    - 5 DoF xArm5
    - End-effector position control where action space is normalized delta positions
    - Purely image based observations
    - 6 real-world tasks: reach, push, push with obstacles, drawer opening, sweeping, drawing
  - Ego4D experiments
    - [Ego4D](https://ego4d-data.org/)
    - Utilize the last frame as the goal frame and learn a ranking component conditioned on the goal frame
    - Discriminator training: positive example from same clip as goal, negative example from different clip
    - Evaluate with "true goal" from same clip and "counterfactual goal" from different clip
        - Reward function increases with true goal
        - Reward function is non-monotonic with counterfactual goal and has lower value
