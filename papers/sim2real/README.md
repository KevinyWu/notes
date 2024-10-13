# Sim2Real

## [2024-09] Hand-Object Interaction Pretraining from Videos

#sim2real
#learning-from-video
#reinforcement-learning
#behavioral-cloning
[[Hand-Object Interaction Pretraining from Videos]]
- Does not assume a strict alignment of the human's intent in the video and the downstream robot tasks: resulting dataset $\mathcal{T}$ contains knowledge that could be valuable to any manipulation tasks
- Simulator as intermediary between video and robot trajectory: can add physics lost in videos and increase dataset diversity by randomizing simulation environment
- Outperforms other video-pretrained models on BC task where one policy is responsible for manipulating multiple different tasks
