# Robotic Manipulation

Notes from the course "Robotic Manipulation" taught by Russ Tedrake at MIT (Fall 2023).

[Lectures (fall 2023)](https://youtube.com/playlist?list=PLkx8KyIQkMfWr191lqbN8WfV08j-ui8WX&feature=shared)
[Full course notes](https://manipulation.csail.mit.edu/)
[Deepnote](https://deepnote.com/workspace/bubhub-afbb4b47-384c-4b93-a423-6aad7f9e29f7/projects/Robotic-Manipulation-c00edfe0-5ae4-4acd-a05a-f8f182589fd0)
[Companion code](https://github.com/RussTedrake/manipulation)

## Contents

- [[courses/robotic-manipulation/README#[1] Introduction|1 Introduction]]
- [[courses/robotic-manipulation/README#[2] Let's Get You a Robot|2 Let's Get You a Robot]]
- [[courses/robotic-manipulation/README#[3] Basic Pick-and-Place|3 Basic Pick-and-Place]]
- [[courses/robotic-manipulation/README#[4] Geometric Pose Estimation|4 Geometric Pose Estimation]]

## [1] Introduction

[[courses/robotic-manipulation/1 Introduction|1 Introduction]]
- Robotics manipulation now includes complex tasks like buttoning shirts, addressing real-world variability
- Simulators like Drake enable training models that generalize to real-world scenarios without overfitting to specific quirks

## [2] Let's Get You a Robot

[[2 Let's Get You a Robot]]
- Robot Description Files include UDRF, SDF, and MJCF formats for defining robot properties and configurations
- Position control is common in robots, utilizing sensors like encoders and PID controllers, while torque control allows adaptation to external forces but is less common due to motor limitations
- Dexterity in robotic hands is limited; underactuated hands and various end effectors like suction cups enhance manipulation capabilities

## [3] Basic Pick-and-Place

[[3 Basic Pick-and-Place]]
- The basic pick-and-place task involves moving a gripper to predefined positions (pregrasp, grasp, desired pose) to manipulate an object while utilizing spatial transforms and kinematics for accurate positioning
- Differential kinematics employs the Jacobian to relate joint changes to changes in end-effector pose and includes techniques for inverse kinematics, such as using the Moore-Penrose pseudo-inverse to manage multiple degrees of freedom
- Challenges like kinematic singularities can be addressed through optimization approaches, incorporating constraints on velocities, positions, and accelerations to ensure smooth and safe operation of robotic arms

## [4] Geometric Pose Estimation

[[4 Geometric Pose Estimation]]
