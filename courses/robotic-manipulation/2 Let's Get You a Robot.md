# Let's Get You a Robot

#robotics
#hardware

[[courses/robotic-manipulation/README#[2] Let's get you a robot|README]]

[Lecture](https://youtu.be/q896_lTh8eA?feature=shared)
[Notes](https://manipulation.csail.mit.edu/robot.html)
[Deepnote](https://deepnote.com/workspace/bubhub-afbb4b47-384c-4b93-a423-6aad7f9e29f7/project/02-Lets-get-you-a-robot-58888247-ce19-4f68-822f-76be6ce00f27)

## 2.1 Robot Description Files

- UDRF: Universal Robot Description Format
- SDF: Simulation Description Format
- MJCF: MuJoCo XML Configuration Format

## 2.2 Arms

- Gears
	- Gear ratio: output teeth:input teeth
	    - Example: 10:1 means 10 output teeth for every 1 input tooth, 10 rotations of input gear for 1 rotation of output gear
- **Tradeoff speed for torque**
    - Motor has high speed and low torque, while the work requires low speed and high torque
    - Solution: use a high gear ratio
- Most motors have a fixed range of rotational speed and torque
    - Work that needs to be done generally requires much lower speeds and higher torque than what the motor can deliver efficiently
- Position-controlled robots
	- Given a desired joint position, robot executes
	- Basically means it does not offer torque control
	- **Torque control**: the rotational force applied by a motor is directly controlled
		- Allows robot to adapt to external forces
	- Position control is the norm
		- Reason: for electric motors, torque of motor output is directly proportional to current
		- We typically choose small electric motors with large gear reductions (large gear ratio), which come with difficult to model friction, vibration, backlash, etc.
		- Thus, the directly proportional relationship between torque and current is not upheld
- Position control
	- Need sensors on motor side
	- Position sensor is most common: encoder or potentiometer
		- Provides accurate measure of joint position and velocity
		- Sufficient to accurately track position trajectory
		- Joint position $q$, desired position $q^d$, error $e = q_d - q$
		- **PID controller**: $\tau = k_p e + k_d \dot{e} + k_i \int e dt$
			- $k_p$: proportional gain
			- $k_d$: derivative gain
			- $k_i$: integral gain
- **Reflected inertia**: the inertial load that is felt on the opposite side of a transmission
	- Mass of the motor is small relative to robot mass, but they play a significant role in the dynamics
- Torque control
	- Some robots *do* have electric motors with smaller gear reduction
	- Hydraulic actuators can apply large torque without large transmission
	- Add sensors to measure torque at the joint side of the actuator (rather than motor side)
		- Series elastic actuators: spring between motor and load
			- Example: Baxter
- Physic engine in Drake: MultibodyPlant
- Visualizer in Drake: SceneGraph

## 2.3 Hands

- **Dexterity**: the ability to manipulate objects in a variety of ways
	- Lack of dextrous hands on the market
- Simple grippers: two fingers, like Schunk WSG 050
- Soft/underactuated hands
	- **Underactuated hands**: hands with fewer actuators than degrees of freedom
	- Often use cable-drive mechanism to close fingers, where a single tendon can move multiple joints
	- Soft hands can improve dexterity and safety
- Other end effectors: suction cups, jamming gripper, etc.

## 2.4 Sensors

- So far, we have seen:
	- Joint feedback
		- iiwa: measured position, estimated velocity, measured torque
		- Schunk gripper: measured state (position + velocity), measured force
	- Joint acceleration typically too noisy to rely on
