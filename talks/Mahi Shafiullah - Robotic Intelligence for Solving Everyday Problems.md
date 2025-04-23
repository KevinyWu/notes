# Mahi Shafiullah - Robotic Intelligence for Solving Everyday Problems

#foundation-models
#behavioral-cloning
#imitation-learning

[[talks/README#[2025-04-23] Mahi Shafiullah - Robotic Intelligence for Solving Everyday Problems|README]]

## Accessible Systems for Real World Interaction

- Problem: data scarcity
- Simulation challenges
	- Sim2Real gap
	- Hard to model diverse environments
- "Robot-free" robot data collection
	- Only model the gripper - enables **cross-embodiment**
	- "The Stick" v1/v2, UMI
	- **How to do this for the new mobil manipulator robot**
	- Make it cheap (make a bunch of them)
		- Hardware and camera: use iPhone!
	- Make it easy (anyone can use)
		- UMI takes ~1 minute to calibrate
		- "The Stick" - AnySense app
			- Uses RGB, LIDAR depth, Gyroscope, IMU, Tactile

## Algorithms for Learning from Real-World Data

- Observation -> model -> action
- Need a large amount of data due to environment changes
- Need to learn rich, multi-model behavior
	- Solution in language models: predict categorical distribution rather than one answer
- VQ-BeT (Behavior Transformer)![[vq_bet.png]]
	- Discretizing actions is lossy
		- Solution: hierarchical action prediction, continuous action offset to cover entire action space
	- Similar performance to diffusion policy, but much lower inference time!
- Robot Utility Models
	- ~75% zero-shot deployment after pretraining on large dataset of 25 behaviors per task, 40 environments
	- How to improve?
		- VLM verifier tells it when it succeeds or fails
		- If fail, retry: now ~90% accuracy

## Zero-Shot, Long-Horizon Mobile Manipulation in Arbitrary Scenes

- Problem breakdown: navigate, pick, navigate, place
- Explicit vs. implicit memory
	- Explicit memory: 3D map
		- Stuck with fixed set of classes and properties
	- Implicit memory: parametrized by neural network
		- Need lots of data and compute to train
	- Use both!
		- DynaMem: spatio-semantic memory
- **Open problems in mobile manipulation**
	- More dextrous tasks
	- Multimodal perception
