# Octo: An Open-Source Generalist Robot Policy

**Authors**: Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, Jianlan Luo, You Liang Tan, Lawrence Yunliang Chen, Pannag Sanketi, Quan Vuong, Ted Xiao, Dorsa Sadigh, Chelsea Finn, Sergey Levine

#foundation-models
#imitation-learning

[[papers/foundation-models/README#[2023-12] Octo An Open-Source Generalist Robot Policy|README]]

[Paper](http://arxiv.org/abs/2405.12213)
[Code](https://github.com/octo-models/octo)
[Website](https://octo-models.github.io/)

## Abstract

> Large policies pretrained on diverse robot datasets have the potential to transform robotic learning: instead of training new policies from scratch, such generalist robot policies may be finetuned with only a little in-domain data, yet generalize broadly. However, to be widely applicable across a range of robotic learning scenarios, environments, and tasks, such policies need to handle diverse sensors and action spaces, accommodate a variety of commonly used robotic platforms, and finetune readily and efficiently to new domains. In this work, we aim to lay the groundwork for developing open-source, widely applicable, generalist policies for robotic manipulation. As a first step, we introduce Octo, a large transformer-based policy trained on 800k trajectories from the Open X-Embodiment dataset, the largest robot manipulation dataset to date. It can be instructed via language commands or goal images and can be effectively finetuned to robot setups with new sensory inputs and action spaces within a few hours on standard consumer GPUs. In experiments across 9 robotic platforms, we demonstrate that Octo serves as a versatile policy initialization that can be effectively finetuned to new observation and action spaces. We also perform detailed ablations of design decisions for the Octo model, from architecture to training data, to guide future research on building generalist robot models.

## Summary

- Octo: pretrained on 800k robot trajectories from Open X-Embodiement
- **Supports both natural language and goal image conditioning**

## Background

- Generalist robot policy (GRP): model that directly maps robot observations to actions and provide zero-shor or few-shot generalization to new tasks
    - Ex. GNM, RoboCat, RT-X

## Method

- Architecture ![[octo.png]]
    - Input tokenizers transform language instructions $l$, goal images $g$, and robot observation sequences $o_1, o_2, \ldots, o_H$ into tokens $[\mathcal{T}_l, \mathcal{T}_g, \mathcal{T}_o]$
    - Transformer backbone processes these tokens and produces embeddings $e_l, e_g, e_o = T(\mathcal{T}_l, \mathcal{T}_g, \mathcal{T}_o)$
    - Readout heads $R$ produce action logits $a = R(e_l, e_g, e_o)$
    - Tasks and observation tokens
        - Language inputs tokenized and passed through pretrained t5-base (111M)
        - Image observations tokenized and passed through shallow convolutional stack, then split into sequence of flattened patches
    - Transformer backbone and readout heads
        - Observation tokens can only attend to tokens from the same or earlier time steps and task tokens (green)
        - Readout tokens $\mathcal{T}_{R, t}$ (purple) attend to observations and task tokens but is not attended to by any observation or task token, so they can only passively read and process internal embeddings without influencing them
        - This design is flexible to add new task and observations inputs
    - Design decisions
        - Transformer over ResNet for image encodings with shallow CNN
        - Early input fusion: transformer requires quadratic scaling with input length, so channel stack goal images with observation images
- **Training details**
    - Trained on mixture of 25 datasets from [Open X-Embodiement](https://robotics-transformer-x.github.io/)
    - Downweight larger datasets
    - Zero-pad any missing camera channels
    - Binary gripper command: 1 if gripper is open, 0 if gripper is closed
    - Training objective
        - Use conditional diffusion decoding to predict continuous, multi-modal action distributions
        - Only one forward pass of the transformer backbone is performed per action prediction, after which the multi-step denoising process is carried out within the small diffusion head
        - To generate action, sample a Gaussian noise vector $x^K \sim \mathcal{N}(0, I)$ and apply $K$ steps of denoising with a learned denoising network $\epsilon_{\theta}(x^k, e, k)$
        - $\epsilon_{\theta}(x^k, e, k)$ is conditoned on the output $x^k$ of the previous denoising step, the step index $k$, and the output embedding $e$ of the transformer action readout
            - $x^{k-1} = \alpha (x^k - \gamma \epsilon_{\theta}(x^k, e, k) + \mathcal{N}(0, \sigma^2 I))$
            - Parameters $\alpha, \gamma, \sigma$ correspond to cosine noise schedule

## Results

- Things that improved performance
    - Adding one frame of history as context
    - Using action chunking (no temporal ensemble needed)
    - Decreasing patch size
    - Increasing shuffle buffer size
- Things that did not work
    - MSE action head rather than diffusion decoding
    - Discrete action heads
    - ResNet Encoders
    - Pretrained Encoders
    - Relative gripper action representation
    - Adding proprioceptive observations
    - Fine-tuning language model (T5 encoder)
