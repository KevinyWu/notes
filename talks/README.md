# Talks

## [2022-03-28] Antonio Loquercio - Learning Vision-Based High-Speed Flight

#tags
[[Antonio Loquercio - Learning Vision-Based High-Speed Flight]]

## [2024-04-06] Tom Silver - Neuro-Symbolic Learning for Bilevel Robot Planning

#robot-learning
#tamp
#unsupervised
[[Tom Silver - Neuro-Symbolic Learning for Bilevel Robot Planning]]
- Unified framework for learning abstractions and grounding mechanisms for task and motion planning (TAMP)
- Unsupervised learning of predicate-operator based state abstractions for planning
- Optimization problem: find trajectory of **low-level** states and actions so that **high-level** abstract states are followed, transitions are valid

## [2024-04-09] Lerrel Pinto - On Building General-Purpose Home Robots

#foundation-models
#robot-learning
#imitation-learning
[[Lerrel Pinto - On Building General-Purpose Home Robots]]
- **OK-Robot**: one-shot pick-and-place using existing pre-trained models: scan room with RGB-D, use **VoxelMap** algorithm to make a 3D point cloud, **OWL-ViT** to create object masks point cloud, **CLIP** image encoder gets image representation that maps to a text representation
- For each CLIP embedding, we can assign a XYZ coordinate, then use A* for pathfinding, **LangSAM** to segment the object from the image, **AnyGrasp** pretrained grasping model
- Versatile Imitation from One Minute of Demonstrations: real-robot RL using optimal transport matching to compute how similar is agent to expert trajectory as reward

## [2024-08-08] Yunzhu Li - Foundation Models for Robotic Manipulation

#robot-learning
#foundation-models
[[Yunzhu Li - Foundation Models for Robotic Manipulation]]
- **VoxPoser**: Code is multilingual interface between humans, foundation models, and robot
- Output may not always be perfect, but should always generate something reasonable!
