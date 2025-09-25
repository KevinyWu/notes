# Self-Improving Autonomous Underwater Manipulation

**Authors**: Ruoshi Liu, Huy Ha, Mengxue Hou, Shuran Song, Carl Vondrick

[[papers/lifelong-learning/README#[2024-10] Self-Improving Autonomous Underwater Manipulation|README]]

[Paper](http://arxiv.org/abs/2410.18969)
[Code](https://github.com/cvlab-columbia/aquabot)
[Website](https://aquabot.cs.columbia.edu/)
[Video](https://youtu.be/iBB980UWdes?feature=shared)

## Abstract

> Underwater robotic manipulation faces significant challenges due to complex fluid dynamics and unstructured environments, causing most manipulation systems to rely heavily on human teleoperation. In this paper, we introduce AquaBot, a fully autonomous manipulation system that combines behavior cloning from human demonstrations with self-learning optimization to improve beyond human teleoperation performance. With extensive real-world experiments, we demonstrate AquaBot's versatility across diverse manipulation tasks, including object grasping, trash sorting, and rescue retrieval. Our real-world experiments show that AquaBot's self-optimized policy outperforms a human operator by 41% in speed. AquaBot represents a promising step towards autonomous and self-improving underwater manipulation systems. We open-source both hardware and software implementation details.

## Summary

- Underwater robots encounter unique challenges posed by high-dimensional and nonlinear fluid dynamics
- Teleoperation is not scalable and often suboptimal, especially underwater
- Two stage training
	- **Distill** human adaptability into closed-loop visuomotor policy (BC)
	- **Accelerate** learned behavior through self-guided optimization

## Background

- Comparison to classical underwater robot controllers
	- Versatility: same method across multiple tasks
	- Simplicity: handles perception, dynamics modeling, motion planning, control all end-to-end
	- Self-improving: robot improves through experience

## Method

- Hardware
	- QYSEA FIFISH V-EVO underwater drone
	- Mounted a second camera on the drone
	- Mounted two external cameras at two corners of the pool for real-time localization of the robot
		- The detected 3D position, plus the internal IMU sensor and compass, provide a full 6 DoF robot pose in the global coordinate system, which we use for navigation and reset
- Learning framework ![[papers/lifelong-learning/img/aquabot.png]]
- Behavioral cloning
	- Each recorded action is an 8D vector composing the 3 Cartesian directions, 3 rotational directions, and open/close gripper movement
	- CNN visual encoder for each camera
		- Observation horizon of 2
	- MLP takes encodings and predicts 8D action; MSE loss
- Self-learning
	- Due to the limitation of teleoperation systems and the human's lack of underwater motor skills, human demonstrations are likely to be sub-optimal
	- **Self-learning algorithm is designed to search for the optimal combination of speed parameters through trial and error**
	- Self-learning method can benefit from the continual accumulation of robot deployment data
	- **Underwater environment is key!**
		- Safer than land
		- Automatic environment reset when object drops to bottom
	- **Since the policy outputs a continuous 6 DoF force/torque control signal, we can learn to accelerate the policy by learning a scaling parameter for each control dimension, where the objective is to complete a manipulation task in the shortest time possible**'
	- Surrogate-based optimization algorithm
		- During one exploration episode: uniformly sample a (action scaling parameters)
		- During one exploitation episode: use a **neural surrogate model** to optimize for the best $\delta$
			- **Surrogate model takes in $\delta$ and predicts $r$, the task completion time**
		- Measure $r$ and add $\{\delta, r\}$ to the self-learning dataset to train surrogate model

## Results

- Behavioral cloning
	- Object grasping
		- Compares MLP, Diffusion Policy, and ACT
		- MLP outperforms both
		- Biggest failure mode of both DP and ACT is the gripper motion
	- Trash sorting
		- Used classifier to predict object category
		- Real-time localization and PID control to navigate robot to place object in correct bin
		- A single system can autonomously perform grasping, sorting, and placing of objects with a variety of appearances, material properties, geometry, and mass
	- Rescue retrieval
		- Grasp and move large heavy object to target area
		- Underwater robots can manipulate objects much larger and heavier than their own body due to the presence of buoyancy
	- **Shorter action horizon more effective due to uncertainties from currents**
- Self-learning
	- We use localization and navigation systems based on the external cameras to reset robot and object positions before each episode
	- Manipulation efficiency from self-learning was initially significantly worse than that of the BC policy human baseline (randomly sampled $\delta$ cause instability)
	- Within only 120 trials, self-learning finds speed parameters outperforming the human baseline by 41% and BC baseline by 68%
