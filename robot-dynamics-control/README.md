# Robot Dynamics and Control

Notes from the book "Robot Dynamics and Control, 2nd Edition" by Mark W. Spong, Seth Hutchinson, and M. Vidyasagar.

- [Robot Dynamics and Control](#robot-dynamics-and-control)
  - [1 Introduction](#1-introduction)
    - [1.1 Robotics](#11-robotics)
    - [1.2 History of Robotics](#12-history-of-robotics)
    - [1.3 Components and Structure of Robots](#13-components-and-structure-of-robots)
    - [1.4 Outline of the Text](#14-outline-of-the-text)

## 1 Introduction

### 1.1 Robotics

- Kinematics, dynamics, motion planning, computer vision, and control

### 1.2 History of Robotics

- RIA definition of robot: "A robot is a reprogrammable multifunctional manipulator designed to move material, parts, tools, or specialized devices through variable programmed motions for the performance of a variety of tasks"
- Modern robots came from teleoperators and numerically controlled milling machines

### 1.3 Components and Structure of Robots

- Composed of links connected by joints into a kinematic chain
- Revolute joints (R): rotation about an axis
- Prismatic joints (P): translation along an axis
- **Degrees of freedom** (DOF): number of independent joint variables required to specify the configuration of the robot
  - At least 6 DOF to access any point in 3D space (3 for position, 3 for orientation)
- Workspace: set of all points reachable by the end-effector
  - Reachable workspace: set of all points reachable by the end-effector without considering joint limits
  - Dextrous workspace: set of all points reachable by the end-effector while satisfying joint limits
- Servo robot: robot with closed-loop control
  - Closed-loop control: system output is monitored and used to adjust the input
- **Robotic system consists of: arm, external power source, end-of-arm tooling, external and internal sensors, computer interface, and control computer**
- Controller resolution: smallest increment of control input that can be commanded
- Rotational axes more accumulation of errors than linear axes

### 1.4 Outline of the Text

- Forward kinematics: mapping from joint space to task space
  - Base frame: fixed frame to which all objects are referenced
  - Denavit-Hartenberg equations
- Inverse  kinematics: mapping from task space to joint space
- Velocity kinematics: mapping from joint space to task space velocities
  - Differentiate Denavit-Hartenberg forward kinematics equations
  - Obtain the Jacobian matrix of the manipulator, $J$
  - $\dot{x} = J\dot{\theta} \leftrightarrow \dot{\theta} = J^{-1}\dot{x}$
  - Jacobian without inverse is a singular configuration
- Path Planning
  - Path planning, trajectory generation, trajectory planning
- Vision
  - Camera sensors as opposed to joint sensors
- Dynamics
  - How much force and torque is required to move the robot
- Position control
  - Control algorithms to execute programmed tasks
  - Tracking and disturbance rejection: tracking desired trajectories and rejecting disturbances i.e. friction, noise
- Force control
  - Errors in position could lead to large forces that damage the end-effector
  - Solution is force control, ex. hybrid control, impedance control
