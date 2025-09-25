# Saurabh Gupta - Robot Learning by Understanding Egocentric Videos

[[talks/README#[2024-05-07] Saurabh Gupta - Robot Learning by Understanding Egocentric Videos|README]]

[Recording](https://www.youtube.com/watch?v=RdPftGBhN8c)

## Scaling Up Robot Learning

- Challenges in robotics
	- Sparse-reward RL is challenging
	- Teleoperation is slow and difficult
- How to scale up
	- Self-supervision
	- Arm-farms
	- Robots in homes
	- Simplifying data collection, i.e. UMI (still difficult to scale)
	- Sim2real (also needs real2sim)
- **Learning from observing other agents**
	- "Human children use observation and imitation"
	- Demonstrations can be experts showing how to solve a task
	- Even when it is not "expert", it shows how the world works
	- Problems
		- Embodiment differences
		- Goals and intents not known
		- Only on-policy data, or trajectories may be suboptimal
	- Control hierarchy (lowest to highest)
		- Motor torque -> macro action -> skills -> subgoals -> goals in the world

## Learning Manipulation from Videos

- [Project website](https://s-gupta.github.io/hands-as-probes/)
- Object understanding
	- Which sites to interact at
	- How to interact with those sites
	- What happens after interaction
	- Learn these by **observing human hands**
- Learning object affordances
	- Data creation with off-the-shelf methods
		- Use models to predict where hands are and grasp type
		- Mask out hand region
	- Train encoder-decoder model to predict grasp
- Learning effects of actions
	- Train a **state-sensitive feature space**
	- Prepare dataset across video
		- Track hand and object movement
		- Feature learning on top of tracks
			- SimCLRv2
			- Leverage temporal consistency: frames close by should be similar in feature space
			- Object-hand consistency: similarity in states through similarity in interaction
		- **Objects may appear different, but similar in the way the hand moves; don't want to prescribe a task to the object in the representation**
			- What matters for the object depends on a task - train a linear classifier to decide what action to do
		- Evaluation on [EPIC-STATES](https://s-gupta.github.io/hands-as-probes/) dataset
- Limitations - addressed in next sections
	- 1) Hands are a nuisance: human-robot domain gap!
	- 2) 3D hand pose estimation: 2D aspects of interaction, doesn't translate to 3D action
	- 3) Compounding execution errors: only shows on-policy data; but need off-policy data for robots (so they learn how to recover)

## 1) Hands are a Nuisance

- [Project website](https://matthewchang.github.io/vidm/)
- Learning **factored representations**: separate image into two images - one containing agent, one containing everything else
	- Use diffusion model in pixel space (rather than in feature space)
	- Agent segmentation model
	- **Video Inpainting Diffusion Model (VIDM)**: recover pixels behind agent
		- Requires strong priors: diffusion model trained on large-scale single-frame data
		- Cross attention to leverage visual history
- Using the factored representations
	- Learn affordances
	- Learn reward functions (Simple one: centroid of the mask)
	- Factorization + inpainting does better than only factorization or only inpainting
		- Both agent and background are important!

## 2) 3D Hand Pose Estimation

- [Project website](https://ap229997.github.io/projects/hands/)
- FrankMocap baseline - trained on lab datasets
	- Not good at generalizing in the wild - OOD
	- Ambiguity in 3D pose in crops
- **WildHands** uses auxiliary supervision on in-the-wild data
	- Produces more curled 3D hands in grasping actions

## 3) Compounding Execution Errors

- [Project website](https://sites.google.com/view/diffusion-meets-dagger)
- Robots may go off distribution and not know how to recover
- Previous solution: DAgger (dataset aggregation)
	- Get more off policy data, label them
- **Diffusion Meets Dagger**
	- Use diffusion model to generate observations and action labels in OOD states
	- Imitation learning with eye-in-hand camera
	- **Generate a perturbed image with diffusion**
		- Perturbation vector $\Delta p$
	- **Label generation for the perturbed image**
		- Let $a_t$ be the action to get from state $s_t$ to $s_{t+1}$
		- The label for the perturbed image is $a_{t} - \Delta p$
			- Basically going back to on-policy image at current time step, then to on-policy image at next time step
	- Shows significant improvement over BC by augmenting the BC data

## Discussion Questions

- **Can't play tennis just by reading about it, or watching a lot of US Open - interactive learning still needed**
- Egocentric vs exocentric video data
- Videos are more grounded in sensory observations than LLMs
- Videos only provide a kinematic observation, we don't observe any forces
