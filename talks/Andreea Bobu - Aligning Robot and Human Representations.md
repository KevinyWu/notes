# Andreea Bobu - Aligning Robot and Human Representations

#human-robot-interaction
#inverse-reinforcement-learning

[[talks/README#[2024-07-10] Andreea Bobu - Aligning Robot and Human Representations|README]]

[Recording](https://www.youtube.com/watch?v=lgt255PXA4I)

## Representation Alignment Problem

- Aligning what the human cares about with what the robot cares about
- Paradigm 1: human input to specify reward
	- Problem: needs humans to label every demonstration
- Paradigm 2: deep learning from behavioral cloning
	- Problem: does not do well in aligning representations, neural network only tries to succeed at the task
- How to get the best of both worlds?
	- **Robots should engage with humans in an interactive process for finding a shared representation**
	- Divide & Conquer: focus explicitly on learning the representation from human input before using it for downstream robot learning

## Learning Representations from Humans

- Deep inverse reinforcement learning: modeled features as neural network
	- Learn both reward and features simultaneously
- Alternative: learn features from human input first, then learn reward from task demonstration using those features
	- Human input: ask human for labels about what is important on the image
		- Problem: robot needs too many labels to learn, people are imprecise at labeling
		- Solution: ask people to provide relative labels (sequence of states "Feature Trace" with monotonically increasing feature values)
		- In practice, this is manually moving a robot arm
		- Assumptions about the Feature Trace
			- Monotonicity (later state preferred over earlier -> quadratic scaling)
			- Start/end equivalence (equal preference)
		- Bradley-Terry model for loss in learning

## Learning a Lot from a Little

- High-dimensional inputs (like images) are data hungry
- A little bit of human-labeled data
	- Augment with simulator
	- Features enable zero-shot transfer from simulation to real world
- Learning emotion
	- VAD (valence-arousal-dominance) 3D emotive latent space
	- Emotional representation can be used with language models
- Using LLMs to generalize to new unseen situations

## Determining Misaligned Representations

- Use cognitive model from psychology (Boltzmann model)
- Detect misalignment with representation **confidence**
- Low confidence in misaligned representations leads to more robust learning
	- Robots doesn't learn from human inputs it does not understand
- Applications in teleoperation and demonstrations

## Future Work

- In reality, human-robot collaboration should be bidirectional
	- Robots and humans should learn a **shared representation**
- Robots should efficiently learn new tasks from humans
- Robots must be intuitive and transparent to humans
- Robots must behave according to user expectations and know when they aren't doing so
- Robots must adapt to different users in a way that minimizes psychological burden
- Robots must personalize to different users
- Robots should adapt to different human capabilities in shared autonomy systems
