# Robot Dynamics and Control

Notes from the book "Robot Dynamics and Control, 2nd Edition" by Mark W. Spong, Seth Hutchinson, and M. Vidyasagar.

## Contents

- [[books/robot-dynamics-control/README#[1] Introduction|1 Introduction]]
	- [[books/robot-dynamics-control/1 Introduction#1.1 Robotics|1.1 Robotics]]
	- [[books/robot-dynamics-control/1 Introduction#1.2 History of Robotics|1.2 History of Robotics]]
	- [[books/robot-dynamics-control/1 Introduction#1.3 Components and Structure of Robots|1.3 Components and Structure of Robots]]
	- [[books/robot-dynamics-control/1 Introduction#1.4 Outline of the Text|1.4 Outline of the Text]]
- [[books/robot-dynamics-control/README#[2] Rigid Motions and Homogeneous Transformations|2 Rigid Motions and Homogeneous Transformations]]
	- [[2 Rigid Motions and Homogeneous Transformations#2.1 Representing Positions|2.1 Representing Positions]]
	- [[2 Rigid Motions and Homogeneous Transformations#2.2 Representing Rotations|2.2 Representing Rotations]]
	- [[2 Rigid Motions and Homogeneous Transformations#2.3 Rotational Transformations|2.3 Rotational Transformations]]
	- [[2 Rigid Motions and Homogeneous Transformations#2.4 Composition of Rotations|2.4 Composition of Rotations]]
	- [[2 Rigid Motions and Homogeneous Transformations#2.5 Parametrization of Rotations|2.5 Parametrization of Rotations]]
	- [[2 Rigid Motions and Homogeneous Transformations#2.6 Homogeneous Transformations|2.6 Homogeneous Transformations]]
- [[books/robot-dynamics-control/README#[3] Forward Kinematics The Denavit-Hartenberg Convention|3 Forward Kinematics: The Denavit-Hartenberg Convention]]
	- [[3 Forward Kinematics - The Denavit-Hartenberg Convention#3.1 Kinematic Chains|3.1 Kinematic Chains]]
	- [[3 Forward Kinematics - The Denavit-Hartenberg Convention#3.2 Denavit-Hartenberg Representation|3.2 Denavit-Hartenberg Representation]]
	- [[3 Forward Kinematics - The Denavit-Hartenberg Convention#3.3 Examples|3.3 Examples]]
- [[books/robot-dynamics-control/README#[4] Inverse Kinematics|4 Inverse Kinematics]]
	- [[4 Inverse Kinematics#4.1 The General Inverse Kinematics Problem|4.1 The General Inverse Kinematics Problem]]
	- [[4 Inverse Kinematics#4.2 Kinematic Decoupling|4.2 Kinematic Decoupling]]
	- [[4 Inverse Kinematics#4.3 Inverse Position A Geometric Approach|4.3 Inverse Position A Geometric Approach]]
	- [[4 Inverse Kinematics#4.4 Inverse Orientation|4.4 Inverse Orientation]]

## [1] Introduction

#robotics
[[books/robot-dynamics-control/1 Introduction|1 Introduction]]
- Robotics involves kinematics, dynamics, motion planning, computer vision, and control, with robots defined as reprogrammable manipulators for various tasks
- Robots consist of links, joints, degrees of freedom, workspaces, servo control, and control resolution
- The text covers forward/inverse kinematics, velocity kinematics, path planning, vision, dynamics, and control strategies

## [2] Rigid Motions and Homogeneous Transformations

#rotations
#transformations
[[2 Rigid Motions and Homogeneous Transformations]]
- Rotations are represented by matrices (basic axes, Euler angles, axis-angle) with specific rules for transforming coordinates between frames, though they don't generally commute
- Rigid motions, combining rotation and translation, are captured by homogeneous transformations (4x4 matrices), preserving distances and angles
- Rotations can be parametrized using minimal sets like Euler angles, yaw-pitch-roll, or axis-angle representations

## [3] Forward Kinematics: The Denavit-Hartenberg Convention

#kinematics
#transformations
[[3 Forward Kinematics - The Denavit-Hartenberg Convention]]
- Forward kinematics calculates the position and orientation of a robot's end-effector using joint variables and transformation matrices for each joint in the kinematic chain
- The Denavit-Hartenberg (DH) convention represents each link transformation as a sequence of four basic transformations using parameters like joint angle, link offset, length, and twist
- DH assumptions and frame assignments simplify the transformation matrix computation, and this method applies to any manipulator, ensuring consistent results despite non-unique frame choices

## [4] Inverse Kinematics

#kinematics
#inverse-kinematics
[[4 Inverse Kinematics]]
- Inverse kinematics involves solving for joint variables that achieve a desired transformation, typically using a homogeneous transformation matrix
- For 6-DOF arms with spherical wrists, the problem can be decoupled into inverse position (solved geometrically for the wrist center) and inverse orientation (solved using rotational matrices)
- The geometric approach helps calculate the first three joint variables, while Euler angles can solve for the final three orientation variables
