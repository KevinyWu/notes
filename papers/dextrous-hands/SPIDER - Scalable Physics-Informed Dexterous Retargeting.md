# SPIDER - Scalable Physics-Informed Dexterous Retargeting

**Authors**: Chaoyi Pan, Changhao Wang, Haozhi Qi, Zixi Liu, Homanga Bharadhwaj, Akash Sharma, Tingfan Wu, Guanya Shi, Jitendra Malik, Francois Hogan

[[papers/dextrous-hands/README#[2025-11] SPIDER - Scalable Physics-Informed DExterous Retargeting|README]]

[Paper](http://arxiv.org/abs/2511.09484)
[Code](https://github.com/jc-bao/spider)
[Website](https://jc-bao.github.io/spider-project/)

## Abstract

> Learning dexterous and agile policy for humanoid and dexterous hand control requires large-scale demonstrations, but collecting robot-specific data is prohibitively expensive. In contrast, abundant human motion data is readily available from motion capture, videos, and virtual reality, which could help address the data scarcity problem. However, due to the embodiment gap and missing dynamic information like force and torque, these demonstrations cannot be directly executed on robots. To bridge this gap, we propose Scalable Physics-Informed DExterous Retargeting (SPIDER), a physics-based retargeting framework to transform and augment kinematic-only human demonstrations to dynamically feasible robot trajectories at scale. Our key insight is that human demonstrations should provide global task structure and objective, while large-scale physics-based sampling with curriculum-style virtual contact guidance should refine trajectories to ensure dynamical feasibility and correct contact sequences. SPIDER scales across diverse 9 humanoid/dexterous hand embodiments and 6 datasets, improving success rates by 18% compared to standard sampling, while being 10X faster than reinforcement learning (RL) baselines, and enabling the generation of a 2.4M frames dynamic-feasible robot dataset for policy learning. As a universal physics-based retargeting method, SPIDER can work with diverse quality data and generate diverse and high-quality data to enable efficient policy learning with methods like RL.

## Summary

- Notes from the introduction and conclusion sections

## Background

- Notes about the background information

## Method

- Notes about the method

## Results

- Notable results from the paper
