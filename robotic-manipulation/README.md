# Robotic Manipulation

[Full course notes](https://manipulation.csail.mit.edu/)

[Lectures (fall 2023)](https://youtube.com/playlist?list=PLkx8KyIQkMfWr191lqbN8WfV08j-ui8WX&feature=shared)

[Companion code](https://github.com/RussTedrake/manipulation)

Notes from the course "Robotic Manipulation" taught by Russ Tedrake at MIT.

- [Robotic Manipulation](#robotic-manipulation)
  - [1 Introduction](#1-introduction)
  - [2 Let's get you a robot](#2-lets-get-you-a-robot)
    - [2.1 Robot Description Files](#21-robot-description-files)
    - [2.2 Arms](#22-arms)
    - [2.3 Hands](#23-hands)
    - [2.4 Sensors](#24-sensors)
  - [3 Basic Pick-and-Place](#3-basic-pick-and-place)

## 1 Introduction

[Lecture 1](https://youtu.be/v04rn86Dehg?feature=shared)

[Deepnote](https://deepnote.com/workspace/bubhub-afbb4b47-384c-4b93-a423-6aad7f9e29f7/project/01-Introduction-bdfeaeb4-e107-472c-a8e7-6848fbd990d0)

- Manipulation is more than pick-and-place
  - 80s and 90s: manipulation referred to pick-and-place and grasping
  - Now, manipulation is broader: buttoning shirt, spreading peanut butter, etc.
- Open-world problem: the world has infinite variability
  - Diversity in open-world problems might make the problem easier
  - For example, now we need quirky solutions to specific problems
  - These quirky solutions may be discarded when the landscape is more diverse
- Simulation
  - Modern simulators can even train models and expect them to work in the real world
  - Models like transformer more general: won't overfit to quirks of simulator image data
  - [Drake](http://drake.mit.edu/) is a simulator that emphasizes the governing equations of motion and physics

## 2 Let's get you a robot

[Lecture 2](https://youtu.be/q896_lTh8eA?feature=shared)

[Deepnote](https://deepnote.com/workspace/bubhub-afbb4b47-384c-4b93-a423-6aad7f9e29f7/project/02-Lets-get-you-a-robot-58888247-ce19-4f68-822f-76be6ce00f27)

### 2.1 Robot Description Files

- UDRF: Universal Robot Description Format
- SDF: Simulation Description Format
- MJCF: MuJoCo XML Configuration Format

### 2.2 Arms

- Gears
  - Gear ratio: output teeth:intput teeth
    - Exmaple: 10:1 means 10 output teeth for every 1 input tooth, 10 rotations of input gear for 1 rotation of output gear
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
    - Join position $q$, desired position $q^d$, error $e = q_d - q$
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

### 2.3 Hands

- **Dexterity**: the ability to manipulate objects in a variety of ways
  - Lack of dextrous hands on the market
- Simple grippers: two fingers, like Schunk WSG 050
- Soft/underactuated hands
  - **Underactuated hands**: hands with fewer actuators than degrees of freedom
  - Often use cable-drive mechanism to close fingers, where a single tendon can move multiple joints
  - Soft hands can improve dexterity and safety
- Other end effectors: suction cups, jamming gripper, etc.

### 2.4 Sensors

- So far, we have seen:
  - Joint feedback
    - iiwa: measured position, estimated velocity, measured torque
    - Schunk gripper: measured state (position + velocity), measured force
  - Joint acceleration typically too noisy to rely on

## 3 Basic Pick-and-Place

[Lecture 3](https://youtu.be/0-34RZJxyww?feature=shared)

[Lecture 4](https://youtu.be/1mkzXp9_QYY?feature=shared)

[Lecture 5](https://youtu.be/YaQrC_Zm8qg?feature=shared)
