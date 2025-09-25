# Basic Pick-and-Place

[[courses/robotic-manipulation/README#[3] Basic Pick-and-Place|README]]

[Lecture 3](https://youtu.be/0-34RZJxyww?feature=shared)
[Lecture 4](https://youtu.be/1mkzXp9_QYY?feature=shared)
[Lecture 5](https://youtu.be/YaQrC_Zm8qg?feature=shared)
[Notes](https://manipulation.csail.mit.edu/pick.html)
[Deepnote](https://deepnote.com/workspace/bubhub-afbb4b47-384c-4b93-a423-6aad7f9e29f7/project/03-Basic-Pick-and-Place-Duplicate-80b33bd1-45f4-4c4b-8e9b-5e98e0e84bf5)

## 3.1 Monogram Notation

- **$^Bp^A_F$: position of point $A$ measured from point $B$ in frame $F$**
	- $^Bp^A_{F_x}$ is the $x$ component of the above
- $W$ is the world frame
	- $^Wp^A_W$ is the position of point $A$ relative to the origin of $W$ measured in the $W$ frame
	- **Shorthand: if the position is measured from and expressed in the same frame, we can omit the subscript**
		- $^Wp^A$ is the same as $^Wp^A_W$
- $B_i$ denotes the frame for body $i$
- $^BR^A$ denotes the orientation of frame $A$ measured from frame $B$
- Frame $F$ can be specified by position and rotation relative to another frame
	- "Spatial pose" = position + orientation
	- **"Transform" is the verb form of "pose"**
	- $^BX^A$ denotes the pose of frame $A$ relative to frame $B$
	- When we refer to pose of object $O$ without mentioning relative frame, we mean $^WX^O$
	- No subscript, we always want pose expressed in the reference frame
- Notation in code
	- $^Bp^A_C$ is written as `p_BA_C`
	- $^BR^A$ is written as `R_BA`
	- $^BX^A$ is written as `X_BA`

## 3.2 Pick and Place via Spatial Transforms

- Basic pick and place problem formulation
	- **Object** $O$
	- **Gripper** $G$
	- Ideal perception sensor gives us $^WX^O$
	- Desired frame $O_d$ of the object
	- Pick and place tries to make $X^O$ match $X^{O_d}$
- Steps
	- Move gripper to a pregrasp position $^OX^{G_{pregrasp}}$ above $O$ (to avoid collision with object)
	- Move gripper to a grasp pos $^OX^{G_{grasp}}$
	- Close gripper
	- Move gripper to desired pose, $X^O = X^{O_d}$
	- Open gripper and retract the hand

## 3.3 Spatial Algebra

- Adding position: $^Ap^B_F + ^Bp^C_F = ^Ap^C_F$
- Additive inverse: $^Ap^B_F + ^Bp^A_F = 0$
- Multiplication by rotation: $^Ap^B_G = ^GR^F \cdot ^Ap^B_F$
- Multiplying rotations: $^AR^B \cdot ^BR^C = ^AR^C$
- Inverse of rotation: $(^AR^B)^{-1} = ^BR^A$
- Rotation matrices are orthonormal: $R^T = R^{-1}$
- Transforms: $^Gp^A = ^GX^F \cdot ^Fp^A = ^Gp^F + ^Fp^A_G = ^Gp^F + ^GR^F \cdot ^Fp^A$
- Transform composition: $^AX^B \cdot ^BX^C = ^AX^C$
- Inverse of transform: $(^AX^B)^{-1} = ^BX^A$
	- Note: $X^T \neq X^{-1}$
- In practice, transforms implemented as homogeneous coordinates
- 3D rotation representations
	- $3 \times 3$ rotation matrix
		- Orthonormal: columns are orthogonal and unit vectors
			- Orthogonality preserves angles after rotation
			- Norm of 1 preserves lengths of vector after rotation
	- Euler angles (roll-pitch-yaw)
	- Axis angles
	- Unit quaternions
		- Singularity in roll-pitch-yaw (at pitch = $\pm \frac{\pi}{2}$, roll and yaw indistinguishable when converting rotation matrix to Euler angles)
			- Becomes problematic when taking derivatives
		- Need at least 4 numbers to represent 3D rotation, hence quaternions

## 3.4 Forward Kinematics

- Forward kinematics: given joint angles, find pose of end effector
	- **Joint positions denoted $q$**
	- **Goal: produce a map $X^G = f_{kin}^G(q)$**
- Kinematic tree
	- Every body in kinematic tree has a parent except world, connected by joints
	- Every joint has position variables that know how compute transforms
		- Configuration between any body and its parent: $^PX^C(q) = ^PX^{J_P} \cdot ^{J_P}X^{J_C}(q) \cdot ^{J_C}X^C$
- Forward kinematics for pick and place
	- Query parent of gripper frame in kinematic tree
	- Recursively compose transforms to get pose of gripper in world frame, $X^G$

## 3.5 Differential Kinematics (Jacobians)

- Change in pose related to change in joint positions by the partial derivative of forward kinematics
	- $dX^G = \frac{\delta f_{kin}^G(q)}{\delta q} dq = J^G(q)dq$
	- $J^G(q)$ is the Jacobian - the derivative of kinematics
- **Spatial velocity (twist) of frame $B$ measured in frame $A$ expressed in frame $C$**: $^AV^B_C = \begin{bmatrix} ^A\omega^B_C \\ ^Av^B_C \end{bmatrix} \in \mathbb{R}^6$
	- $\omega \in \mathbb{R}^3$: angular velocity
		- Can be expressed in three components (but rotations need 4) because $\omega$ is not periodic, it can take any value
	- $v \in \mathbb{R}^3$: translational (linear) velocity
	- Properties
	    - Adding angular velocities: $^A\omega^B_F + ^B\omega^C_F = ^A\omega^C_F$
	    - Additive inverse: $^A\omega^B_F + ^B\omega^A_F = 0$
	    - Translational velocities compose: $^Av^C_F = ^Av^B_F + ^Bv^C_F + ^A\omega^B_F \times ^Bp^C_F$
	    - Additive inverse: $-^Av^B_F = ^Bv^A_F + ^A\omega^B_F \times ^Bp^A_F$
	  - There can be many other representations kinematic Jacobian due to different representations of 3D rotation
	    - **Analytic Jacobian** is the one previously defined: $dX^G = J^G(q)dq$
			- Linearly relates pose change with joint changes
	    - **Geometric Jacobian**: $V^G = J^G(q)v$
			- Linearly relates spatial velocity with generalized velocities
		- Note: $dq \neq v$ necessarily
			- $dq$ is the change in joint positions, but $q$ is often in $\mathbb{R}^7$ due to the representation of rotation as a quaternion in $\mathbb{R}^4$
			- $v$ is in $\mathbb{R}^6$ so the fimensions don't match

## 3.6 Differential Inverse Kinematics

- Geometric Jacobian: $V^G = J^G(q)v$
	- $v$ is the generalized velocity
	- $V^G$ is the spatial velocity of the gripper
	- $v = [J^G(q)]^{-1}V^G$
- For a 7-DOF iiwa robot, $J^G(q_{iiwa}) \in \mathbb{R}^{6 \times 7}$
	- This is not square, so it is not invertible!
	- More DOF than desired spatial velocity representation, which is good: many solutions for $v$ that give the same $V^G$
- **Moore-Penrose pseudo-inverse**: $J^+$
	- $v = [J^G(q)]^+V^G$
	- When $J$ square and invertible, $J^+ = J^{-1}$
	- When many solutions, $J^+$ gives the solution with the smallest norm
	- When no solutions, $J^+$ gives the joint velocities that produce a spatial velocity as close to desired $V^G$ as possible
- **Kinematic singularities**: configurations $q$ for which $\text{rank}(J(q)) < 6$
	- When smallest singular value approaches zero, the robot is near a singularity: norm of $J^+$ becomes large

## 3.7-3.9 Pick and Place

- Note xyz = rgb in frame visualizations
- Pregrasp pose is above the object to avoid collisions
- Need to define desired $X$ poses (i.e. moving to pre-grasp, grasp, pre-place, and place)
	- $X$ is made up of position and orientation
	- Thus, we can get a pose trajectory
	- We can convert this pose trajectory to a spatial velocity trajectory $V$
	- Then, we can use the Pseudo-inverse Jacobian controller to get the velocity commands ![[3.7.png]]
	- This completes the pick and place task!

## 3.10 Differential Inverse Kinematics with Constraints

- The above pick-and-place solution works, but the pseudo-inverse controller does not perform well around singularities
	- Small singular value leads to large joint velocities
	- Cannot handle constraints on joint angles, velocities, accelerations, torques (robot will clip out of range commands)
- **Pseudo-inverse as optimization**
	- $\min_v |J^G(q)v - V^{G_d}|_2^2$
	- The solution to this is the pseudo-inverse
- Velocity constraints
	- $\min_v |J^G(q)v - V^{G_d}|_2^2$ subject to $v_{min} \leq v \leq v_{max}$
	- Convex quadratic programming (QP) problem
- Adding position and acceleration constraints
	- Denote time step $h$
	- $\min_{v_n} |J^G(q)v_n - V^{G_d}|_2^2$ subject to
		- $v_{min} \leq v_n \leq v_{max}$
		- $q_{min} \leq q + h v_n \leq q_{max}$
		- $\dot{v}_{min} \leq \frac{v_{n+1} - v_n}{h} \leq \dot{v}_{max}$
- Joint centering
	- $J^G$ for iiwa is $6 \times 7$, so rank of $J^G$ is less than the DOF
		- Infinite number of solutions
		- Convex optimization solver normally chooses something reasonable, but we can modify the problem to choose a unique solution
  - **Secondary controller** that attempts to control all the joints
    - Consider simple proportional joint-space controller $v=K(q_0 - q)$
    - Denote $P(q)$ as an orthonormal basis for the kernel (null space) of a Jacobian
		- Can be implemented with pseudoinverse: $P(q) = I - J^+(q)J(q)$
		- The null space of a Jacobian represents the directions in which the arm can move without changing the end effector pose
		- Secondary controller acts only in null space so does not interfere with primary end-effector control
    - Adding $Pv = PK(q_0 - q)$ as a secondary objective
    - $\min_{v_n} |J^G(q)v_n - V^{G_d}|_2^2 + \epsilon |P(q)(v_n - K(q_0 - q))|^2_2$ subject to constraints
		- **This secondary objective represents the velocity (projected onto the null space) that tries to achieve the desired velocity and reduce the configuration error $q_0 - q$**
		- $q_0$ is the nominal (or "home") configuration of the robot, a preferred joint configuration that we would like the robot to maintain or return to when possible, assuming it doesn't interfere with the primary task
		- $\epsilon \ll 1$ only needed if there are constraints because constraints cause these two objectives to clash

## 3.11 Exercises
