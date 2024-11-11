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
	- [Gear ratio](https://www.youtube.com/watch?v=txQs3x-UN34): driven gear teeth:driving gear teeth
	    - Example: 10:1 means 10 driven teeth for every 1 driver tooth
- **Tradeoff speed for torque**
    - Motor has high speed and low torque, while the work requires low speed and high torque
    - Solution: use a high gear ratio $N$
	    - $\tau_{driven} = N\times \tau_{driving}$
	    - $\text{RPM}_{driven}= \text{RPM}_{driving}/N$
	    - Intuitively, high gear ratio increases the lever length ("leverage"), which increases the torque (strength) but reduces the speed (RPM)
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
	- Moment of inertia is how "resistant" an object is to changes in rotational motion (Inertia in rotational motion is analogous to mass in linear motion)
	- Mass of the motor is small relative to robot mass, but they play a significant role in the dynamics
	- For a system where a motor drives a load through a gear box with gear ratio $N$: $I_{reflected} = \frac{I_{load}}{N^2}$
		- With a high gear ratio, the reflected inertia is reduced, making the load "feel" lighter to the motor, allowing it to accelerate and decelerate the load more easily with less torque
		- Ex. high gear on a bike, reflected inertia is low, making it easy for your legs to maintain a steady pace with low effort once moving
- Torque control
	- Some robots *do* have electric motors with smaller gear reduction
	- Hydraulic actuators can apply large torque without large transmission
	- Add sensors to measure torque at the joint side of the actuator (rather than motor side)
		- Series elastic actuators: spring between motor and load
			- Example: Baxter
- Physics engine in Drake: MultibodyPlant
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

## 2.5-2.6 Putting It All Together

- HardwareStation: diagram containing all components necessary for simulation of hardware and environment
	- Inputs: robot position, torque, gripper position, gripper force limit
	- Outputs: Commanded position, measured position, measured velocity, state information, torque information, camera observations, privileged information (only available in simulation, i.e. contact results, query object)
- HardwareStationInterface provides the same system but for real hardware
	- Easy transition between simulation and real world
- Context: class that holds and manages the overall state of a robotic system, including robot data, sensor readings, state variables, etc.

## 2.7 Exercises

- [Exercise 1 - Role of Reflected Inertia](https://deepnote.com/workspace/bubhub-afbb4b47-384c-4b93-a423-6aad7f9e29f7/project/02-Lets-get-you-a-robot-58888247-ce19-4f68-822f-76be6ce00f27/notebook/01reflectedinertia-51e447e108664d65af292e6c1cb0e069?utm_medium=product&utm_source=share_modal&utm_campaign=copy_link&utm_content=58888247-ce19-4f68-822f-76be6ce00f27) ![[2.7.png]]
	- Dynamics equation: $ml^2\ddot{\theta}= -mgl\sin{\theta} + \tau$
		- Inertial torque (LHS)
			- Newton's second law: torque ($Nm$) = moment of inertia ($kg\cdot m^2$) x angular acceleration ($\text{rad}/s^2$)
			- Moment of inertia of a mass $m$ at distance $l$ from the pivot is $ml^2$
		- Torque from gravitational force and external force (RHS)
			- $\tau = l \times F$
			- $mg\sin{\theta}$ is force from the bob perpendicular to the lever arm $l$
			- Negative because torque restores pendulum to equilibrium
	- When a motor with gear ratio $N$, inertia $I_m$, and torque $\tau_m$ is driving the pendulum
		- Inertial torque (LHS): $(N^2I_{m}+ ml^2)\ddot{\theta}= -mgl\sin{\theta} + N\tau_m$
			- Added inertia due to the motor; $I_m$ is the *reflected inertia*, so the load inertia is $N^2I_m$
		- Torque from gravitational force and motor (RHS)
			- Gravitational force the same
			- Motor is the driving gear, load is the driven gear, and $\tau_{driven} = N\times \tau_{driving}$
- [Exercise 2 - Input and Output Ports on the Manipulation Station](https://deepnote.com/workspace/bubhub-afbb4b47-384c-4b93-a423-6aad7f9e29f7/project/02-Lets-get-you-a-robot-58888247-ce19-4f68-822f-76be6ce00f27/notebook/02hardwarestationio-fcfe19639d294460a097b9219e75def8?utm_medium=product&utm_source=share_modal&utm_campaign=copy_link&utm_content=58888247-ce19-4f68-822f-76be6ce00f27)
	- $\tau_{\text{ff}}$ (Feedforward Torque): The torque applied to the robot based on a model or prediction, used to anticipate dynamics like gravity or inertia without waiting for feedback
	- $\tau_{\text{no ff}}$ (Torque Without Feedforward): The torque applied when no feedforward is used, relying solely on feedback to correct any errors or disturbances
	- $\tau_{\text{commanded}}$ (Commanded Torque): The total torque the robot is instructed to apply, which includes both feedforward and feedback components to achieve the desired motion
	- $\tau_{\text{commanded}} = \tau_{\text{no ff}} + \tau_{\text{ff}}$
- [Exercise 3 - Direct Joint Teleop in Drake](https://deepnote.com/workspace/bubhub-afbb4b47-384c-4b93-a423-6aad7f9e29f7/project/02-Lets-get-you-a-robot-58888247-ce19-4f68-822f-76be6ce00f27/notebook/03directjointcontrol-654e34e2f8154bb698f66ca17891032b?utm_medium=product&utm_source=share_modal&utm_campaign=copy_link&utm_content=58888247-ce19-4f68-822f-76be6ce00f27)
	- Get current joint positions: `q_current = station.GetOutputPort("iiwa.position_commanded").Eval(context)`
	- Command new joint positions: `q_current_cmd = station.GetInputPort("iiwa.position").FixValue(context, q_cmd)`
