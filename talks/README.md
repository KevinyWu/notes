# Talks

## [2022-03-28] Antonio Loquercio - Learning Vision-Based High-Speed Flight

[[Antonio Loquercio - Learning Vision-Based High-Speed Flight]]
- Imitation learning with simulation training is enough for zero-shot real world high-speed agile flight
- Need some abstraction function to reduce gap between sim and real

## [2024-04-06] Tom Silver - Neuro-Symbolic Learning for Bilevel Robot Planning

[[Tom Silver - Neuro-Symbolic Learning for Bilevel Robot Planning]]
- Unified framework for learning abstractions and grounding mechanisms for task and motion planning (TAMP)
- Unsupervised learning of predicate-operator based state abstractions for planning
- Optimization problem: find trajectory of low-level states and actions so that high-level abstract states are followed, transitions are valid

## [2024-04-09] Lerrel Pinto - On Building General-Purpose Home Robots

[[Lerrel Pinto - On Building General-Purpose Home Robots]]
- OK-Robot: one-shot pick-and-place using existing pre-trained models: scan room with RGB-D, use VoxelMap algorithm to make a 3D point cloud, OWL-ViT to create object masks point cloud, CLIP image encoder gets image representation that maps to a text representation
- For each CLIP embedding, we can assign a XYZ coordinate, then use A* for pathfinding, LangSAM to segment the object from the image, AnyGrasp pretrained grasping model
- Versatile Imitation from One Minute of Demonstrations: real-robot RL using optimal transport matching to compute how similar is agent to expert trajectory as reward

## [2024-05-07] Saurabh Gupta - Robot Learning by Understanding Egocentric Videos

[[Saurabh Gupta - Robot Learning by Understanding Egocentric Videos]]
- Learning factored representations: segment out agent, then Video Inpainting Diffusion Model (VIDM) recovers pixels behind agent; use this factored representation to learn affordances or reward functions
- WildHands for improved 3D hand pose estimation in the wild
- Diffusion Meets DAgger uses diffusion models to generate observations and action labels in out of distribution (OOD) states for imitation learning

## [2024-06-14] David Held - Spatially-aware Robot Learning for Deformable Object Manipulation

[[David Held - Spatially-aware Robot Learning for Deformable Object Manipulation]]
- Self-supervised fine-tuning for deformable objects (MEDOR) uses Chamfer distance and mapping consistency loss at test time to refine cloth manipulation, improving model accuracy through occlusion-aware mesh reconstruction
- Action-conditioned cloth tracking predicts future cloth states based on current mesh and action input, leveraging real-world video data and using Chamfer and neighborhood consistency losses to mitigate prediction errors

## [2024-07-10] Andreea Bobu - Aligning Robot and Human Representations

[[Andreea Bobu - Aligning Robot and Human Representations]]
- Robots need to interact with humans to establish shared representations, focusing on learning these representations from human input before using them for tasks
- Proposes using relative human-labeled data and simulation to augment limited data, enabling zero-shot transfer and leveraging models for generalization
- Misaligned representations are detected through confidence models, improving robustness by avoiding unreliable human inputs

## [2024-08-08] Yunzhu Li - Foundation Models for Robotic Manipulation

[[Yunzhu Li - Foundation Models for Robotic Manipulation]]
- VoxPoser: Code is multilingual interface between humans, foundation models, and robot
- Output may not always be perfect, but should always generate something reasonable!

## [2024-08-27] Yilun Du - Generalizing Outside The Data Distribution through Compositional Generation

[[Yilun Du - Generalizing Outside The Data Distribution through Compositional Generation]]
- Energy-based models (EBMs) enable compositional generation by representing distributions probabilistically as energy landscapes, allowing flexible combination of constraints for tasks like planning and multimodal reasoning
- Compositionality in EBMs facilitates generalization outside training data by combining independent factors, supporting tasks like generating compositional scenes, adapting styles, and solving robotic manipulation problems
- Applications span robotics, planning, and multimodal tasks, with EBMs combining diverse data sources (e.g., video, language, and visual models) to optimize long-horizon goals and tasks using unified energy landscapes

## [2024-09-13] Rachel Holladay - Leveraging Mechanics for Multi-step Robotic Manipulation Planning

[[Rachel Holladay - Leveraging Mechanics for Multi-step Robotic Manipulation Planning]]
- Robots must to reason over geometry and physics together to accomplish long-horizon manipulation tasks
- Planning under uncertainty: dealing with dynamic non-prehensile actions (DNP)
- Philosophy: allow robot to keep acting my never doing anything catastrophic

## [2024-10-11] Antonio Loquercio - Simulation-What Made Us Intelligent Will Make Our Robots Intelligent

[[Antonio Loquercio - Simulation-What Made Us Intelligent Will Make Our Robots Intelligent]]
- Sensorimotor simulation enhances both high-level skills (planning) and low-level skills (motor control)
- Move away from physics-based simulators because a perfect simulator is not feasible or scalable, and include senses beyond vision
- Robot simulation is a flexible medium between learning from human video and real robots

## [2024-08-25] Carl Vondrick - Making Sense of the Multimodal World

[[Carl Vondrick - Making Sense of the Multimodal World]]
- Surrogate models that predict the performance of a predicted action can be used to guide online self-learning
- Underwater is a good environment for self-learning due to safety and environment reset (objects dropping to seafloor)
- GenAI can be used to generate videos of people doing a task, and robots can imitate these videos to generalize to unseen objects

## [2024-10-01] Peter Stone - Practical Reinforcement Learning - Lessons from 30 Years of Research

[[Peter Stone - Practical Reinforcement Learning - Lessons from 30 Years of Research]]
- Practical RL requires choosing the right algorithm, learning useful representations, and balancing exploration with decision-making
- Multiagent RL is complex, but interactions with humans can simplify learning through guidance and teamwork
- Decomposing problems, leveraging past experiences, and acknowledging finite time are key to efficient RL in real-world applications

## [2024-11-04] Yuke Zhu - Data Pyramid and Data Flywheel for Robotic Foundation Models

[[Yuke Zhu - Data Pyramid and Data Flywheel for Robotic Foundation Models]]
- First generalist, then better specialist: prioritize developing generalist robotic capabilities before refining specialized skills
- Need to learn to generalize across the data pyramid: effective robotic foundation models must leverage and generalize across web, synthetic, and real-robot data
- Data flywheel through trustworthy and safe deployment: continuous improvement comes from real-world deployment, where more data leads to better learning, more capable robots, and further data collection

## [2025-02-10] Peter Stone - Human-in-the-Loop Machine Learning for Robot Navigation and Manipulation

[[Peter Stone - Human-in-the-Loop Machine Learning for Robot Navigation and Manipulation]]
- Human-in-the-loop learning improves robot navigation and manipulation by integrating human feedback through various modalities such as demonstration, intervention, evaluative feedback, and reinforcement learning
- Adaptive Planner Parameter Learning (APPL) enhances navigation by learning local planner parameters from human input, reducing the need for expert tuning and adapting to different environments dynamically
- ORION enables vision-based manipulation from a single human video by constructing object-centric representations and open-world object graphs, allowing robots to generalize manipulation skills from minimal demonstrations

## [2025-04-23] Mahi Shafiullah - Robotic Intelligence for Solving Everyday Problems

[[Mahi Shafiullah - Robotic Intelligence for Solving Everyday Problems]]
- Develop accessible, low-cost systems to collect robot data across embodiments using minimal hardware and mobile devices
- VQ-BeT and Robot Utility Models to efficiently learn multimodal behaviors from real-world data, achieving high zero-shot success rates
- Tackled long-horizon mobile manipulation with hybrid memory (DynaMem)

## [2025-04-25] Russ Tedrake - Multitask Transfer in TRI's Large Behavior Models for Dexterous Manipulation

[[Russ Tedrake - Multitask Transfer in TRI’s Large Behavior Models for Dexterous Manipulation]]
- LBM pretraining reduces the amount of task-specific data required
- Given the same amount of task-specific data, finetuned specialists derived from pretrained LBMs outperform single-task models when aggregating over tasks
- Data normalization is much more important than architecture changes

## [2025-09-10] Yann Le Cun - SSL, JEPA, World Models and the Future of AI

[[Yann Le Cun - SSL, JEPA, World Models and the Future of AI]]
- Generative models don't work on images and videos since the world is only partially predictable
- Planning with world models: equivalent to MPC but with world model learned through observations rather than hand written equations
- Abandon generative models for JEPA, contrastive methods for regularization methods, probabilistic models for EBMs, RL for MPC

## [2025-09-19] Ken Goldberg - How to Close the 100,000 Year "Data Gap" in Robotics

[[Ken Goldberg - How to Close the 100,000 Year "Data Gap" in Robotics]]
- Dex-Net: Monte-carlo simulation to get probability of success for a grasp, 6.7 million examples: (object, grasp, probability), then train a network on this data to pick up objects from a bin
- $\pi_0$: 10k hours of data ~1 year; QWEN-2.5: 1.2B hours of data ~100,000 years
- Use model-based methods to make systems viable for deployment, then collect data at deployment time to improve model-free learning: Waymo, Ambi Robotics

## [2025-11-24] Binghao Huang - Learning with Scalable Tactile Skin for Fine-Grained Manipulation

[[Binghao Huang - Learning with Scalable Tactile Skin for Fine-Grained Manipulation]]
- 4D points for visuo-tactile information: 3D for space and 1D for tactile, balancing the number of points for vision and tactile space
- For RGB + tactile, encode RGB and masked tactile separately, then cross attend (also train a decoder than reconstructs the original tactile)
- Can do RL with simulated sensor environment
