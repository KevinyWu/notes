# Yuke Zhu - Data Pyramid and Data Flywheel for Robotic Foundation Models

#humanoid
#sim2real
#foundation-models

[[talks/README#[2024-11-04] Yuke Zhu - Data Pyramid and Data Flywheel for Robotic Foundation Models|README]]

[Recording](https://ai.princeton.edu/events/robotics-symposium-videos)
[Slides](https://rpl.cs.utexas.edu/talks/2024_11_04_data_pyramid_and_data_flywheel_for_robotic_foundation_models.pdf)

## Why Humanoids?

- Versatility
- Hardware is getting cheaper
- Safety: humanoids can be more predictable by non-expert humans
- Data: human-centered data widely available
- **Research principle #1: first generalist, then better specialist**
	- [OKAMI: Teaching Humanoid Robots Manipulation Skills through Single Video Imitation](https://ut-austin-rpl.github.io/OKAMI/)
		- Learning from human videos
		- Reference plan generation
		- Object-aware retargeting
	- [GR00T N1: An Open Foundation Model for Generalist Humanoid Robots](https://arxiv.org/pdf/2503.14734)

## Data Pyramid for Building Robotic Foundation Models

- Web data > synthetic data > real-robot data
- **Research principle #2: Need to learn to generalize across the data pyramid**
- Hierarchical autonomy stack
	- System 2: multimodal foundation model (~1 Hz)
	- System 1: sensorimotor policy (~20-50 Hz), whole-body controller (~500-1k Hz)
- Solutions for System 2
	- VLMs trained on large-scale web data
	- [BUMBLE: Unifying Reasoning and Acting with Vision-Language Models for Building-wide Mobile Manipulation](https://robin-lab.cs.utexas.edu/BUMBLE/)
		- VLM serves as reasoning core
		- Use COT to reason what skill to perform from the skill library
		- Long and short term memory to store past behaviors to reason about future behavior
- Solutions for System 1
	- Real-time teleoperation (but we don't understand scaling laws yet)
	- Synthetic data is Yuke's bet!
	- [RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots](https://robocasa.ai/)
		- Text-to-3D object generation
		- 9 most common kitchen layouts
	- [DexMimicGen: Automated Data Generation for Bimanual Dexterous Manipulation via Imitation Learning](https://dexmimicgen.github.io/)
		- Real2Sim2Real
		- Human teleoperation -> 5 source demos -> DexMimicGen -> 1000s of simulated demos
		- Object-centered representations to generate trajectories
			- Relative pose between gripper and object stays the same
		- There will be failures, but we are in simulation so we can discard them
		- Multi-task imitation learning augmented with RoboCasa and DexMimicGen shows steady improvements in policy success rate

## Three-phase Training for Robotic Foundation Models

- LLMs: self-supervised pretraining -> supervised fine-tuning -> RLHF
- Robot foundation model analogy: pre-training with data pyramid -> fine-tuning on domain-specific data -> alignment during deployment
	- Data flywheel: more data -> better learning -> more capable robots -> increased deployments -> more data
- **Research Principle #3: Data Flywheel through Trustworthy and Safe Deployment**
- Robot learning on the job
	- [Robot Learning on the Job: Human-in-the-Loop Autonomy and Learning During Deployment](https://ut-austin-rpl.github.io/sirius/)
		- Shared control for on-the-job policy improvement
	- [Multi-Task Interactive Robot Fleet Learning with Visual World Models](https://ut-austin-rpl.github.io/sirius-fleet/)
		- Train a visual world model to predict future frame based on past history
		- Can anticipate future failures to signal when human intervention is needed
		- With this, on person can command a fleet of robots to collect data
- Future hope: as we turn the data flywheel, the data pyramid will flip upside down!
