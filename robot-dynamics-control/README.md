# Robot Dynamics and Control

Notes from the book "Robot Dynamics and Control, 2nd Edition" by Mark W. Spong, Seth Hutchinson, and M. Vidyasagar.

- [Robot Dynamics and Control](#robot-dynamics-and-control)
  - [1 Introduction](#1-introduction)
    - [1.1 Robotics](#11-robotics)
    - [1.2 History of Robotics](#12-history-of-robotics)
    - [1.3 Components and Structure of Robots](#13-components-and-structure-of-robots)
    - [1.4 Outline of the Text](#14-outline-of-the-text)
  - [2 Rigid Motions and Homogeneous Transformations](#2-rigid-motions-and-homogeneous-transformations)
    - [2.1 Representing Positions](#21-representing-positions)
    - [2.2 Representing Rotations](#22-representing-rotations)
    - [2.3 Rotational Transformations](#23-rotational-transformations)
    - [2.4 Composition of Rotations](#24-composition-of-rotations)
    - [2.5 Parametrization of Rotations](#25-parametrization-of-rotations)
    - [2.6 Homogeneous Transformations](#26-homogeneous-transformations)

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

## 2 Rigid Motions and Homogeneous Transformations

### 2.1 Representing Positions

- Geometric reasoning
  - Synthetic approach: reason directly about geometric objects (points, lines, planes)
  - Analytic approach: reason about coordinates of points, use algebraic manipulations
- Vectors are invariant w.r.t. coordinate systems, but their representation by coordinates is not
- Need to make all coordinate vectors defined w.r.t. a common frame to perform algebraic manipulations using coordinates

### 2.2 Representing Rotations

- Denote $x_1^0$ as the point $x_1$ expressed in frame $0$
  - If the reference frame is obvious, we can drop the superscript
- Rotation matrix (2D example)
  - $$R_1^0 = [x_1^0 | y_1^0] = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$
  - <img src="figures/2.2.png" width="400" alt="2.2">
  - **Dot product of two vectors is the cosine of the angle between them**
    - We can instead project the axis of the $1$ frame onto the $0$ frame
    - $R_1^0 = \begin{bmatrix} x_1 \cdot x_0 & y_1 \cdot x_0 \\ x_1 \cdot y_0 & y_1 \cdot y_0 \end{bmatrix}$
    - Note that if we project $0$ onto $1$, we have $R_0^1 = (R_1^0)^T = (R_1^0)^{-1}$
- Properties of rotation matrices, $n = 2,3$
  - $R\in SO(n)$
  - **Orthogonal**: $R^T = R^{-1}$
  - $\det(R) = 1$
  - Columns are unit length and mutually orthogonal
- **Right hand rule**: thumb points in the direction of the first vector, fingers curl in the direction of the second vector
- **Basic rotation matrix**
  - Subscript denotes rotation axis
  - $R_x(\theta) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\theta & -\sin\theta \\ 0 & \sin\theta & \cos\theta \end{bmatrix}$, $R_y(\theta) = \begin{bmatrix} \cos\theta & 0 & \sin\theta \\ 0 & 1 & 0 \\ -\sin\theta & 0 & \cos\theta \end{bmatrix}$, $R_z(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}$
  - Properties
    - $R_{z,0} = I$
    - $R_{z,\theta}\cdot R_{z,\phi} = R_{z,\theta+\phi}$
    - Implies that $R_{z, \theta}^{{-1}} = R_{z, -\theta}$

### 2.3 Rotational Transformations

- Consider point $p$ in frame $1$, ie.e $p_1$
  - Rotation matrix can transform coordinates from one frame to another
  - $p_0 = R_1^0p_1$, represents the same point $p$ in frame $0$
- We can also use rotation matrices to represent rigid motions in the same frame
  - If point $p_b$ is obtained by rotating the point $p_a$ by $R$, then the coordinates of $p_b$ are $R\cdot p_a$

### 2.4 Composition of Rotations

- **Composition law for rotational transformations**: $R_2^0 = R_1^0\cdot R_2^1$
  - This is rotating from frame $2$ to frame $1$ and then from frame $1$ to frame $0$
  - $p^0 = R_1^0\cdot R_2^1\cdot p^2$
  - Rotational transformations do not commute in general
- Rotation with respect to a **fixed** frame
  - Want to perform sequence of rotations each about the same fixed frame
  - **Reverse order than composition law for rotational transformations**
  - Example
    - $o_0x_0y_0z_0$ is the reference frame
    - $o_1x_1y_1z_1$ is the frame obtained by rotating $o_0x_0y_0z_0$ by $R_1^0$
    - $o_2x_2y_2z_2$ is the frame obtained by rotating $o_1x_1y_1z_1$ **w.r.t. reference frame** (not $o_1x_1y_1z_1$), denote this rotation $R$
    - Let $R_2^0$ be the orientation of $o_2x_2y_2z_2$ w.r.t. $o_0x_0y_0z_0$
      - This matrix is what we want to find
      - By the composition law, $R_2^0 = R_1^0\cdot R_2^1$
      - Need to determine $R_2^1$
      - $R_2^1$ is equivalent to first rotating the $1$ frame to the reference frame, $(R_1^0)^{-1}$, and then rotating the by $R$ to get the $2$ frame from the $1$ frame in the fixed frame, then we must undo the first rotation by $R_1^0$
      - Thus, $R_2^0 = R_1^0\cdot R_2^1 = R_1^0[(R_1^0)^{-1}RR_1^0] = RR_1^0$

### 2.5 Parametrization of Rotations

- We can represent rotations in different ways, only needing three independent parameters
- Euler angles
  - <img src="figures/2.5.1.png" width="500" alt="2.5.1">
  - Successive rotations about the moving axes
  - $(\phi, \theta, \psi)$
    - Rotate about $z$ by $\phi$
    - Rotate about current $y$ by $\theta$
    - Rotate about current $z$ by $\psi$
  - Also denoted $R_1^0 = R_{z,\phi}R_{y,\theta}R_{z,\psi}$
    - See the basic rotation matrices in section [2.2](#22-representing-rotations)
    - Can derive angles from a rotation matrix by expanding the basic rotation matrices and solving for the angles (see book pg. 50)
- Yaw-Pitch-Roll angles
  - <img src="figures/2.5.2.png" width="200" alt="2.5.2">
  - Successive rotations about the fixed axes
  - $(\phi, \theta, \psi)$
    - Rotate about $x_0$ by $\phi$ (yaw)
    - Rotate about $y_0$ by $\theta$ (pitch)
    - Rotate about $z_0$ by $\psi$ (roll)
  - Also denoted $R_1^0 = R_{z,\phi}R_{y,\theta}R_{x,\psi}$
- Axis-Angle representation
  - <img src="figures/2.5.3.png" width="300" alt="2.5.3">
  - Let $k = [k_x, k_y, k_z]^T$ be a unit vector in the $0$ frame and $\theta$ be the angle of rotation
  - Rotation matrix $R_{k, \theta}$
    - To derive this, rotate $k$ into a coordinate axes (say $z_0$), then about $z_0$ by $\theta$, then rotate $k$ back to its original position
    - In the example above, $R_{k, \theta} = R_{z, \alpha}R_{y, \beta}R_{z, \theta}R_{y,-\beta}R_{z,-\alpha}$
    - Since $k$ is a unit vector, we can fine the rotation matrix $R_{k, \theta}$
  - **Any rotatation matrix can be represented by a single rotation about a suitable axis**: $k = [k_x, k_y, k_z]^T$ and angle $\theta$
    - This is four parameters, however, we only need two of the three components of $k$ since $k$ is a unit vector
    - Can write it as $r = [r_x, r_y, r_z]^T = [\theta k_x, \theta k_y, \theta k_z]^T$
  - Given a rotation matrix $R$
    - $\theta$ = $\cos^{-1}\left(\frac{1}{2}(\text{tr}(R) - 1)\right) = \cos^{-1}\left(\frac{1}{2}(r_{11} + r_{22} + r_{33} - 1)\right)$
    - $k = \frac{1}{2\sin\theta}\begin{bmatrix} r_{32} - r_{23} \\ r_{13} - r_{31} \\ r_{21} - r_{12} \end{bmatrix}$
  - $R_{k, \theta} = R_{-k, -\theta}$

### 2.6 Homogeneous Transformations

- A transformation of the form $p^0 = R_1^0p^1 + d_1^0$ is said to define a **rigid motion** if $R$ is orthogonal
  - Composition of rigid motions is a rigid motion
    - $p^0 = R_1^0p^1 + d_1^0$
    - $p^1 = R_2^1p^2 + d_2^1$
    - $p^0 = R_1^0R_2^1p^2 + R_1^0d_2^1 + d_1^0 = R_2^0p^2 + d_2^0$
      - Where $d_2^0 = R_1^0d_2^1 + d_1^0$
      - Vector from origin of $0$ frame to origin $2$ frame has coordinates given by the sum of $d_1^0$ (vector from $o_0$ to $o_1$ w.r.t. $0$) and the rotated $d_2^1$ (vector from $o_1$ to $o_2$ w.r.t. $0$ after rotation $R_1^0$)
- Rigid motions can be expressed as $H = \begin{bmatrix} R & d \\ 0 & 1 \end{bmatrix}$
  - $0$ and $d$ are vectors, $R$ is a rotation matrix
- This $H$ is called a **homogeneous transformation**
  - Inverse of $H$ is $H^{-1} = \begin{bmatrix} R^T & -R^Td \\ 0 & 1 \end{bmatrix}$
  - Need to augmeent the vectors $p^0$ and $p^1$ with a $1$ to make them homogeneous
  - $P^0 = [p^0; 1]^T$, $P^1 = [p^1; 1]^T$
  - $P^0, P^1$ known as homogeneous representations
- Set of $4\times 4$ matrices $H$ of that form are denoted $E(3)$
- **Basic homogeneous transformations**
  - These generate $E(3)$, the set of all rigid motions
  - $\text{Trans}_{x,a} = \begin{bmatrix} 1 & 0 & 0 & a \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$, $\text{Trans}_{x,b} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & b \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$, $\text{Trans}_{x,c} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & c \\ 0 & 0 & 0 & 1 \end{bmatrix}$
  - $\text{Rot}_{x,\alpha} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & \cos\alpha & -\sin\alpha & 0 \\ 0 & \sin\alpha & \cos\alpha & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$, $\text{Rot}_{y,\beta} = \begin{bmatrix} \cos\beta & 0 & \sin\beta & 0 \\ 0 & 1 & 0 & 0 \\ -\sin\beta & 0 & \cos\beta & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$,  $\text{Rot}_{z,\gamma} = \begin{bmatrix} \cos\gamma & -\sin\gamma & 0 & 0 \\ \sin\gamma & \cos\gamma & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$
- General homogeneous transformation
  - $H_1^0 = \begin{bmatrix} n_x & s_x & a_x & d_x \\ n_y & s_y & a_y & d_y \\ n_z & s_z & a_z & d_z \\ 0 & 0 & 0 & 1 \end{bmatrix} = \begin{bmatrix} n & s & a & d \\ 0 & 0 & 0 & 1 \end{bmatrix}$
  - n represents the $x$ axis of frame $1$ w.r.t. frame $0$, s represents the $y$ axis of frame $1$ w.r.t. frame $0$, a represents the $z$ axis of frame $1$ w.r.t. frame $0$, d represents the origin of frame $1$ w.r.t. frame $0$
- $E(3)$ has same composition and ordering of transformations as $3\times 3$ rotations
