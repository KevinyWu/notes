# Dextrous Hands

## [2017-11] Embodied Hands - Modeling and Capturing Hands and Bodies Together

[[Embodied Hands - Modeling and Capturing Hands and Bodies Together]]
- MANO (hand Model with Articulated and Non-rigid defOrmations)
- In order to make the model practical for the purpose of scan registration, we will try to expose a set of parameters that efficiently explain the most common hand poses
- 6, 10, 15 PCA components explain 81%, 90%, 95% of the full space

## [2024-09] Hand-Object Interaction Pretraining from Videos

[[Hand-Object Interaction Pretraining from Videos]]
- Does not assume a strict alignment of the human's intent in the video and the downstream robot tasks: resulting dataset $\mathcal{T}$ contains knowledge that could be valuable to any manipulation tasks
- Simulator as intermediary between video and robot trajectory: can add physics lost in videos and increase dataset diversity by randomizing simulation environment
- Outperforms other video-pretrained models on BC task where one policy is responsible for manipulating multiple different tasks
