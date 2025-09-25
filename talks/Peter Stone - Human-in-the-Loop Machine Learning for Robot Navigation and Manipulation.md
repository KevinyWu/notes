# Peter Stone - Human-in-the-Loop Machine Learning for Robot Navigation and Manipulation

[[talks/README#[2025-02-10] Peter Stone - Human-in-the-Loop Machine Learning for Robot Navigation and Manipulation|README]]

[Recording](https://youtu.be/9E8NpqE1T-o?feature=shared)

## Motivation

- **Question**: To what degree can autonomous intelligent agents learn in the presence of teammates and/or adversaries in real-time, dynamic domains?
- Human in the loop learning
	- TAMER
	- Deep TAMER
	- Imitation from Observation

## Human in the Loop Learning for Robot Navigation

- Problem: moving from one point to another without collision
- Classical navigation stack does poorly in constrained spaces
	- Often sample-based, so uses a lot of parameters
- **Adaptive Planner Parameter Learning (APPL)**
	- Learning local planners' **parameters** (keep navigation stack)
		- Combine these methods with other methods like A* or RRT to plan global path
	- Learning from non-expert humans using **different interaction modalities**
	- Replace the **parameters** of local planner with machine learning
		- Classical navigation systems require expert roboticists to manually tune parameters
	- Idea: humans are not robotics experts, but they are navigation experts
	- Different modalities
		- Learning from **demonstration** (APPL-D)![[appld.png]]
			- Conditioned on a context (open space, corridor, etc), learn a parameter library with behavioral cloning (supervised learning)
			- Automatic segmentation of human demonstration into different contexts based on sensory data
		- Learning from **interventions** (APPL-I) ![[appli.png]]
			- Robots do not behave suboptimally everywhere: intervene only when necessary
		- Learning from **evaluative feedback** (APPL-E) ![[apple.png]]
			- Non-expert users may not be able to take control of the robot
			- Similar to TAMER, provide positive/negative reinforcement signal
			- Start with existing parameter library: collect feedback set, train feedback predictor, deploy parameter policy
		- Learning from **reinforcement** (APPL-R) ![[applr.png]]
			- Reinforcement learning in simulation
			- Benchmark autonomous robot navigation (BARN) dataset
			- Reward function: $R_{f} + 0.5R_{p}+ 0.05R_c$
				- Taks completion, distance to goal, negative inverse distance to obstacle
			- Still learning **parameters**, not motor actions
- Cycle of learning for APPL: start with APPL-R in simulation, then tune with other modalities

## Human in the Loop Learning for Robot Manipulation

- Learning from human video data
- [ORION: Vision-based Manipulation from Single Human Video with Open-World Object Graphs](https://ut-austin-rpl.github.io/ORION-release/)
	- Maximize utility of each video: for every skill you want to teach the robot, you only need one demonstration
	- Insight: move objects along trajectories similar to ones shown in video
	- Object-centric representation
		- Track object motion in video
		- Localize objects at test time
	- **ORION (Open-World Video Imitation)** ![[orion_train.png]]
		- Video ->
			- RGB-D video with stationary camera
			- Single-handed interaction
			- Minimal annotation of objects in video
		- Vision foundation models ->
			- Object tracking with OS, keypoint motion tracking with TAP, hand tracking
		- Open-world Object Graph (OOGs) ->
			- Extract keyframes from video, construct graphs that show relation between objects
		- Generalizeable policy at test time ![[orion_test.png]]
	- Segment the video to account for infinite length videos
	- Future work
		- Eliminate need for depth
		- Higher DoF robots (i.e. humanoids)
		- Dynamic and forceful manipulation
