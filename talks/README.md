# Talks

## [2022-03-28] Antonio Loquercio - Learning Vision-Based High-Speed Flight

#imitation-learning
#autonomous-flight
#sim2real
[[Antonio Loquercio - Learning Vision-Based High-Speed Flight]]
- Imitation learning with simulation training is enough for zero-shot real world high-speed agile flight
- Need some abstraction function to reduce gap between sim and real

## [2024-04-06] Tom Silver - Neuro-Symbolic Learning for Bilevel Robot Planning

#tamp
#unsupervised
#planning
[[Tom Silver - Neuro-Symbolic Learning for Bilevel Robot Planning]]
- Unified framework for learning abstractions and grounding mechanisms for task and motion planning (TAMP)
- Unsupervised learning of predicate-operator based state abstractions for planning
- Optimization problem: find trajectory of low-level states and actions so that high-level abstract states are followed, transitions are valid

## [2024-04-09] Lerrel Pinto - On Building General-Purpose Home Robots

#foundation-models
#imitation-learning
[[Lerrel Pinto - On Building General-Purpose Home Robots]]
- OK-Robot: one-shot pick-and-place using existing pre-trained models: scan room with RGB-D, use VoxelMap algorithm to make a 3D point cloud, OWL-ViT to create object masks point cloud, CLIP image encoder gets image representation that maps to a text representation
- For each CLIP embedding, we can assign a XYZ coordinate, then use A* for pathfinding, LangSAM to segment the object from the image, AnyGrasp pretrained grasping model
- Versatile Imitation from One Minute of Demonstrations: real-robot RL using optimal transport matching to compute how similar is agent to expert trajectory as reward

## [2024-05-07] Saurabh Gupta - Robot Learning by Understanding Egocentric Videos

#learning-from-video
#visual-representations
#diffusion
[[Saurabh Gupta - Robot Learning by Understanding Egocentric Videos]]
- Learning factored representations: segment out agent, then Video Inpainting Diffusion Model (VIDM) recovers pixels behind agent; use this factored representation to learn affordances or reward functions
- WildHands for improved 3D hand pose estimation in the wild
- Diffusion Meets DAgger uses diffusion models to generate observations and action labels in out of distribution (OOD) states for imitation learning

## [2024-06-14] David Held - Spatially-aware Robot Learning for Deformable Object Manipulation

#unsupervised
#deformable-objects
#sim2real
[[David Held - Spatially-aware Robot Learning for Deformable Object Manipulation]]
- Self-supervised fine-tuning for deformable objects (MEDOR) uses Chamfer distance and mapping consistency loss at test time to refine cloth manipulation, improving model accuracy through occlusion-aware mesh reconstruction
- Action-conditioned cloth tracking predicts future cloth states based on current mesh and action input, leveraging real-world video data and using Chamfer and neighborhood consistency losses to mitigate prediction errors

## [2024-07-10] Andreea Bobu - Aligning Robot and Human Representations

#human-robot-interaction
#inverse-reinforcement-learning
[[Andreea Bobu - Aligning Robot and Human Representations]]
- Robots need to interact with humans to establish shared representations, focusing on learning these representations from human input before using them for tasks
- Proposes using relative human-labeled data and simulation to augment limited data, enabling zero-shot transfer and leveraging models for generalization
- Misaligned representations are detected through confidence models, improving robustness by avoiding unreliable human inputs

## [2024-08-08] Yunzhu Li - Foundation Models for Robotic Manipulation

#3d-scenes
#foundation-models
[[Yunzhu Li - Foundation Models for Robotic Manipulation]]
- VoxPoser: Code is multilingual interface between humans, foundation models, and robot
- Output may not always be perfect, but should always generate something reasonable!

## [2024-08-27] Yilun Du - Generalizing Outside The Data Distribution through Compositional Generation

#diffusion
#planning
#energy-based-models
[[Yilun Du - Generalizing Outside The Data Distribution through Compositional Generation]]
- Energy-based models (EBMs) enable compositional generation by representing distributions probabilistically as energy landscapes, allowing flexible combination of constraints for tasks like planning and multimodal reasoning
- Compositionality in EBMs facilitates generalization outside training data by combining independent factors, supporting tasks like generating compositional scenes, adapting styles, and solving robotic manipulation problems
- Applications span robotics, planning, and multimodal tasks, with EBMs combining diverse data sources (e.g., video, language, and visual models) to optimize long-horizon goals and tasks using unified energy landscapes

## [2024-09-13] Rachel Holladay - Leveraging Mechanics for Multi-step Robotic Manipulation Planning

#tamp
#planning
[[Rachel Holladay - Leveraging Mechanics for Multi-step Robotic Manipulation Planning]]
- Robots must to reason over geometry and physics together to accomplish long-horizon manipulation tasks
- Planning under uncertainty: dealing with dynamic non-prehensile actions (DNP)
- Philosophy: allow robot to keep acting my never doing anything catastrophic

## [2024-10-11] Antonio Loquercio - Simulation-What Made Us Intelligent Will Make Our Robots Intelligent

#sim2real
#3d-scenes
#learning-from-video
#tactile
[[Antonio Loquercio - Simulation-What Made Us Intelligent Will Make Our Robots Intelligent]]
- Sensorimotor simulation enhances both high-level skills (planning) and low-level skills (motor control)
- Move away from physics-based simulators because a perfect simulator is not feasible or scalable, and include senses beyond vision
- Robot simulation is a flexible medium between learning from human video and real robots

## [2024-08-25] Carl Vondrick - Making Sense of the Multimodal World

#unsupervised
#marine-robotics
#reinforcement-learning
#behavioral-cloning
[[Carl Vondrick - Making Sense of the Multimodal World]]
- Surrogate models that predict the performance of a predicted action can be used to guide online self-learning
- Underwater is a good environment for self-learning due to safety and environment reset (objects dropping to seafloor)
- GenAI can be used to generate videos of people doing a task, and robots can imitate these videos to generalize to unseen objects

## [2025-02-10] Peter Stone - Human-in-the-Loop Machine Learning for Robot Navigation and Manipulation

#tags
[[Peter Stone - Human-in-the-Loop Machine Learning for Robot Navigation and Manipulation]]
- Notes
