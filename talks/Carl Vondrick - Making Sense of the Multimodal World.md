# Carl Vondrick - Making Sense of the Multimodal World

#unsupervised
#marine-robotics
#reinforcement-learning
#behavioral-cloning

[[talks/README#[2024-08-25] Carl Vondrick - Making Sense of the Multimodal World|README]]

[Recording](https://youtu.be/3yl3Cdz36nw?feature=shared)

## Discovery Through Experimentation

- [Intelligent crow](https://www.youtube.com/watch?v=NGaUM_OngaY)
	- Shows efficiency of biological brains to reason!
	- Meanwhile LLMs require large data centers
- Paper airplane experimental robot ![[paper_plane.png]]
	- Surrogate "predictive model" that predicts the performance ( distance of the flight)
	- Relatively simple design space
		- 5-dimensional: 4 points to fold and placement of black strip
- Kirigami gripper
	- gripper made of cut slits in paper
	- Where to make cuts on the paper to maximize the gripping force?

## Learning Underwater

- **Practical issues with real-world learning**
	- Robots break things
	- Robots overheat
	- Environments don't reset
- [[Self-Improving Autonomous Underwater Manipulation|Aquabot]]
	- Underwater, there is a safety buffer
	- Environments reset as objects drop to the floor
	- Start with behavioral cloning
		- Suboptimal demonstrations - difficult to teleoperate underwater
	- **Self-learning underwater**![[aquabot.png]]
		- The self-learned delta adjust actions to more efficiently and accurately pick up rocks
		- BC + self-learning works better and faster by both BC and human teleoperation
		- Can do long-horizon tasks: picking up and sorting trash

## Generalization with GenAI

- Small amount of physical robot data compared to visual
- Zero123 (Zero-shot 1 image to 3D object)
	- Reconstructs models of things from one image
- Dreamitate ![[dreamitate.png]]
	- Idea: replace prediction with GenAI
	- Generate a "dream" - a video of a person performing a task
		- Generated videos may have nonsense or physically impossible things
		- Use 3D tracker to tack end-effector for a robot
	- Enables generalization to manipulate never-seen-before objects
