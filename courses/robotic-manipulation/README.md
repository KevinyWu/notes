# Robotic Manipulation

Notes from the course "Robotic Manipulation" taught by Russ Tedrake at MIT.

[Lectures (fall 2023)](https://youtube.com/playlist?list=PLkx8KyIQkMfWr191lqbN8WfV08j-ui8WX&feature=shared)
[Full course notes](https://manipulation.csail.mit.edu/)
[Deepnote](https://deepnote.com/workspace/bubhub-afbb4b47-384c-4b93-a423-6aad7f9e29f7/projects/Robotic-Manipulation-c00edfe0-5ae4-4acd-a05a-f8f182589fd0)
[Companion code](https://github.com/RussTedrake/manipulation)

## Contents

- [[courses/robotic-manipulation/README#[1] Introduction|1 Introduction]]
- [[courses/robotic-manipulation/README#[2] Let's Get You a Robot|2 Let's Get You a Robot]]
	- [[2 Let's Get You a Robot#2.1 Robot Description Files|2.1 Robot Description Files]]
	- [[2 Let's Get You a Robot#2.2 Arms|2.2 Arms]]
	- [[2 Let's Get You a Robot#2.3 Hands|2.3 Hands]]
	- [[2 Let's Get You a Robot#2.4 Sensors|2.4 Sensors]]
	- [[2 Let's Get You a Robot#2.5-2.6 Putting It All Together|2.5-2.6 Putting It All Together]]
	- [[2 Let's Get You a Robot#2.7 Exercises|2.7 Exercises]]
- [[courses/robotic-manipulation/README#[3] Basic Pick-and-Place|3 Basic Pick-and-Place]]
	- [[3 Basic Pick-and-Place#3.1 Monogram Notation|3.1 Monogram Notation]]
	- [[3 Basic Pick-and-Place#3.2 Pick and Place via Spatial Transforms|3.2 Pick and Place via Spatial Transforms]]
	- [[3 Basic Pick-and-Place#3.3 Spatial Algebra|3.3 Spatial Algebra]]
	- [[3 Basic Pick-and-Place#3.4 Forward Kinematics|3.4 Forward Kinematics]]
	- [[3 Basic Pick-and-Place#3.5 Differential Kinematics (Jacobians)|3.5 Differential Kinematics (Jacobians)]]
	- [[3 Basic Pick-and-Place#3.6 Differential Inverse Kinematics|3.6 Differential Inverse Kinematics]]
	- [[3 Basic Pick-and-Place#3.7-3.9 Pick and Place|3.7-3.9 Pick and Place]]
	- [[3 Basic Pick-and-Place#3.10 Differential Inverse Kinematics with Constraints|3.10 Differential Inverse Kinematics with Constraints]]
- [[courses/robotic-manipulation/README#[4] Geometric Pose Estimation|4 Geometric Pose Estimation]]
	- [[4 Geometric Pose Estimation#4.1 Cameras and Depth Sensors|4.1 Cameras and Depth Sensors]]
	- [[4 Geometric Pose Estimation#4.2 Representations for Geometry|4.2 Representations for Geometry]]
	- [[4 Geometric Pose Estimation#4.3 Point Cloud Registration with Known Correspondences|4.3 Point Cloud Registration with Known Correspondences]]
	- [[4 Geometric Pose Estimation#4.4 Iterative Closest Point (ICP)|4.4 Iterative Closest Point (ICP)]]
	- [[4 Geometric Pose Estimation#4.5 Dealing with Partial Views and Outliers|4.5 Dealing with Partial Views and Outliers]]
	- [[4 Geometric Pose Estimation#4.6 Non-Penetration and "Free-Space" Constraints|4.6 Non-Penetration and "Free-Space" Constraints]]

## [1] Introduction

#robotics
[[courses/robotic-manipulation/1 Introduction|1 Introduction]]
- Robotics manipulation now includes complex tasks like buttoning shirts, addressing real-world variability
- Simulators like Drake enable training models that generalize to real-world scenarios without overfitting to specific quirks

## [2] Let's Get You a Robot

#robotics
#hardware
[[2 Let's Get You a Robot]]
- Robot Description Files include UDRF, SDF, and MJCF formats for defining robot properties and configurations
- Position control is common in robots, utilizing sensors like encoders and PID controllers, while torque control allows adaptation to external forces but is less common due to motor limitations
- Dexterity in robotic hands is limited; underactuated hands and various end effectors like suction cups enhance manipulation capabilities

## [3] Basic Pick-and-Place

#robotics
#kinematics
#inverse-kinematics
[[3 Basic Pick-and-Place]]
- The basic pick-and-place task involves moving a gripper to predefined positions (pregrasp, grasp, desired pose) to manipulate an object while utilizing spatial transforms and kinematics for accurate positioning
- Differential kinematics employs the Jacobian to relate joint changes to changes in end-effector pose and includes techniques for inverse kinematics, such as using the Moore-Penrose pseudo-inverse to manage multiple degrees of freedom
- Challenges like kinematic singularities can be addressed through optimization approaches, incorporating constraints on velocities, positions, and accelerations to ensure smooth and safe operation of robotic arms

## [4] Geometric Pose Estimation

#tags
[[4 Geometric Pose Estimation]]
