# Supervised Policy Learning for Real Robots

[[tutorials/README#[2024-07-19] Supervised Policy Learning for Real Robots|README]]

[Website](https://supervised-robot-learning.github.io/#reading-list)
[Recording](https://www.youtube.com/watch?v=jIB_joS7ww8)
[Slides](https://drive.google.com/drive/folders/1wnGHYcKrc_3XbAhpmyG36d4PWkaWPOQ3?usp=sharing)

## Introduction

- Two main types of imitation learning
    - **Behavioral cloning (BC)** - observations to actions
    - Inverse reinforcement learning (IRL) - infers reward function given observations
- Why BC?
    - BC is working well, enabling robots to do tasks that were impossible a few years ago
        - But often < 90% success
    - Control from only RGB
        - No need for explicit state estimation
        - No explicit dynamics (intuitive physics)
        - No cost function tuning
        - Enables other sensing modalities (sound, tactile, etc.)
    - Emergence of GPT
        - LLMs -> VLMs -> LBMs (large behavior models like VLA)
        - Actions are different from language tokens
            - Continuous
            - Have to obey physics
            - Feedback/stability important
            - Success in single-task BC suggests these are not blockers!
- Why not BC?
    - Requires lots of data (but less than we thought!) and compute
    - Slow inference
    - "Transfer learning bet"
        - Best data: teleoperation
        - Worst getting actions from more common data like YouTube
        - In between: simulation data, UMI, cross-embodiment data (Open X)
    - Not yet robust
        - But if we develop this, we get a new type of robustness: **physical common sense**
        - Ability to adapt when very out of distribution
- Challenges
    - Distribution shift, compounding error
    - Multimodal demonstrations (many ways to solve a task)
    - Can we have both stability and multimodality?
- BC architectures
    - Most models today are **auto-regressive**: next token given history, then shift
        - As opposed to state-space models like LSTM
    - Output decoders
        - Discrete tokens (**RT-1, RT-2**): actions discretized into uniform bins
        - Discrete tokens + continuous offset (**BeT, VQ-BeT**)
        - Continuous output with CVAE encoder/decoder (**ACT**)
        - Continuous output sampled at inference time via denoising (**Diffusion Policy**)
            - More expressive but more expressive inference
    - Input encoders
        - ResNet
        - ViT
        - ImageNet
        - Proprioception: encoding joints/poses
        - VLM for multitask
    - Do we need new architecture?
        - First understand basics
            - Use of proprioception
            - Context length limitations
            - Even domain experts give different answers for basic explanations
        - **Need more real robot rollouts!**

## Hands-on Supervised Policy Learning (Part I)

- Data collection
    - Open-loop replays: replay demonstration to verify action space is correct
    - No timestep discrepancies, latencies in certain parts of the trajectory due to other systems
    - Open-loop replay may fail with variability in environment or stochasticity
- BC with MLP + MSE
    - Observation (image + proprioception) -> ConvNet/ViT -> MLP -> predicted action -> MSE against action label
    - Pre-trained visual encoders
        - CV: ImageNet, ResNet
        - Semi-supervised: CLIP, DINO
        - Robot data pre-trained: R3M, MVP, LIV, Dobb-E
        - **Important to fine-tune the visual encoder!**
    - Problem: multimodal data + least-squares regression = unimodal solution
    - What if we used nearest neighbors instead of MLP?
        - Works in some cases, but out-of-distribution failure more common
        - With images, learn nearest neighbors in latent space
- Many ways to fail
    - Bad featurization (especially if visual): visuomotor policies harder than state-based policies
    - Multi-modal demonstrations: demonstrated behavior can have multiple modes
    - Train-test discrepancy: cumulative error causes models to derail

## Hands-on Supervised Policy Learning (Part II)

- Learning behavior through tokenization
    - Difference: actions are continuous, how to tokenize?
    - Continuous action -> tokenizer -> discrete tokens -> de-tokenizer -> continuous action
    - **Binning actions**: uniform (equal width), quantile (equal actions in each bin), k-means
    - **Easier to learn multi-modal distributions over discrete action space**
    - Use transformer to process observation sequence
        - Flexible: can add image goals, language goals
        - Tokenize ground truth action and use categorical loss
- VQ-BeT
    - Three parts
        - Sequence modeling
        - Offset prediction (offset is difference between ground truth continuous action and descretized ground truth)
        - Loss calculation
    - Advice and factors to consider
        - Balance between action prediction loss and offset prediction loss
        - Predicting single action vs action chunk prediction
        - Temperature during rollout
            - Higher temperature: increase in entropy
            - Lower temperature: more deterministic
        - Codebook size (number of action bins)
            - VQ-BeT uses 16x16
        - Number of VQ layers
- Diffusion Policy
    - Diffusion model
        - 1. Add noise to training data until Gaussian noise
        - 2. Predict noise while denoising
    - Add noise to human trajectory to make them **random trajectories**
    - Denoising process denoises these random trajectories
    - Signs that it works well
        - Reactive to perturbations
        - Succeeds on long-horizon tasks
        - Discrete/branching logic
    - **Switched to relative actions (relative to current end-effector pose) rather than absolute action**
    - **Validation loss tends to go up during training! (not sure why - policy still works)**
- Standard training recipe at TRI
    - Fixed training steps (80k)
    - Take the last checkpoint
    - Good data collection
        - Initial data collection ~100 demonstrations
        - Keep in mind the environment, background, view
        - Intentionally fail and demonstrate the recovery
        - If obvious failures in rollout, collect more demonstrations showing recovery
        - Almost never more than 200 demonstrations!

## What Matters in Supervised Policy Learning

- Start in simulation
    - RoboMimic, LIBERO, LeRobot
    - Helps understand the mechanics of training
    - Helps debug and transfer to real world
- Visualize data
    - [rerun.io](https://rerun.io/)
- Look at training and validation losses
- Eval on robot! - this is the only real metric
- What if your policy fails?
    - Start simple: small dataset, reduce entropy
