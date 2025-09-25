# Imitation Learning

## [2024-08] A Comparison of Imitation Learning Algorithms for Bimanual Manipulation

[[A Comparison of Imitation Learning Algorithms for Bimanual Manipulation]]
- RL is hard for manipulation, need good rewards for all areas of environment
- Imitation learning (specifically BC) does not need an explicit reward function
- Diffusion Policy and ACT perform the best, but there are only simulated experiments (no real-robot)

## [2023-04] Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware

[[Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware]]
- ACT low-cost robots to perform fine-grained bimanual manipulation by learning action sequences from human demonstrations, addressing compounding errors in imitation learning
- CVAE-based model generates coherent action sequences based on both visual observations and joint positions
- Action chunking and temporal ensemble techniques improve policy stability and task performance
