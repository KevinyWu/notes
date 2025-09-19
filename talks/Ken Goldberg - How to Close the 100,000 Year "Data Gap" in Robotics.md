# Ken Goldberg - How to Close the 100,000 Year "Data Gap" in Robotics

#lifelong-learning
#kinematics

[[talks/README#[2025-09-19] Ken Goldberg - How to Close the 100,000 Year "Data Gap" in Robotics|README]]

## Robotic Data

- Large data solved vision and language
- Waymo: data + engineering
	- Modularity
	- Human brain is also modular
- Open-X embodiment: almost all tasks are pick and place
	- Pick and place solved by classic algorithms: [Russ Tedrake](https://manipulation.csail.mit.edu/pick.html)

## Dex-Net

- Computer vision: Image-Net
- Robotics: [Dex-Net](https://berkeleyautomation.github.io/dex-net/)
	- Dataset of 3D objects
	- Monte-carlo simulation to get probability of success for a grasp
	- 6.7 million examples: (object, grasp, probability)
		- Positive and negative examples
	- Train a network on this data to pick up objects from a bin
- Suction gripper
	- Predict where the good/bad suction points are
	- [Ambi Robotics](https://www.ambirobotics.com/)

## End-to-End Learning

- Bitter Lesson
- $\pi_0$: 10k hours of data ~1 year
- QWEN-2.5: 1.2B hours of data ~100,000 years
- Sources of robot data
	- Simulation
		- Works aerial vehicles, locomotion, whole-body control
		- Does not work for manipulation
	- Videos
		- Not 3D
	- Teleoperation
		- Tedious and slow
	- **Real production**
		- Collecting data from real robots
		- At Ambi: started collecting data for maintenance purposes
			- Ambi now has 22 years of data
			- **PRIME-1: trained on 1% of the production data, performs better than original model!**
		- Use model-based methods to make systems viable for deployment, then collect data at deployment time to improve model-free learning
			- Waymo doing this
- Good old fashioned engineering: [Latent Policy Barrier](https://project-latentpolicybarrier.github.io/)
