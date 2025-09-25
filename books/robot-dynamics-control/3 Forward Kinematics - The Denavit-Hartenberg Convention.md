# Forward Kinematics: The Denavit-Hartenberg Convention

[[books/robot-dynamics-control/README#[3] Forward Kinematics The Denavit-Hartenberg Convention|README]]

## 3.1 Kinematic Chains

- Forward kinematics problem: determine position and orientation of end-effector given joint variables
- Robot manipulator is composed of a set of links connected together by various joints (revolute, prismatic, ball-and-socket, etc.)
- For single DOF joints, the action can be described as one number (angle for revolute, displacement for prismatic)
- Robot manipulator with $n$ joints will have $n+1$ links
	- Joints $1, 2, \ldots, n$ connect links $0, 1, \ldots, n$
	- Joint $i$ connects link $i-1$ to link $i$
	- When joint $i$ is actuated, link $i$ moves (so link $0$ is fixed)
	- **Joint variable $q_i$ describes the $i^{th}$ joint**
	- Attach a coordinate frame $o_ix_iy_iz_i$ to each link $i$
		- The coordinates of each point on link $i$ always constant w.r.t. frame $i$ ![[3.1.png]]
	- Inertial frame $o_0x_0y_0z_0$ is fixed
- $A_i$ is the homogeneous transformation matrix that expresses the position and orientation of $o_ix_iy_iz_i$ w.r.t. $o_{i-1}x_{i-1}y_{i-1}z_{i-1}$
	- $A_i$ depends only on the joint variable $q_i$: $A_i = A_i(q_i)$
- **Transformation matrix** $T^i_j$ expresses the position and orientation of $o_jx_jy_jz_j$ w.r.t. $o_ix_iy_iz_i$
	- $T^i_j = A_{i+1}\ldots A_{j}$
	- $T^i_j = I$ if $i = j$
	- $T^i_j = (T^j_i)^{-1}$ if $j > i$
- **Homogeneous transformation matrix**: $H = \begin{bmatrix} R_n^0 & o_n^0 \\ 0 & 1 \end{bmatrix}$
	- $o^0_n \in \mathbb{R}^3$ is the coordinates of the origin of the end effector frame w.r.t. the base frame
	- $R_n^0 \in SO(3)$ is the orientation of the end effector w.r.t. the base frame
	- Then $H = T_n^0 = A_1(q_1)\cdots A_n(q_n)$
- $A_i = \begin{bmatrix} R_i^{i-1} & o_i^{i-1} \\ 0 & 1 \end{bmatrix}$
	- Then $T_j^i = A_{i+1}\cdots A_j = \begin{bmatrix} R_j^i & o_j^i \\ 0 & 1 \end{bmatrix}$
	- $R_j^i = R_{i+1}^i \cdots R_j^{j-1}$
	- Recursively $o_j^i = o_{j-1}^i + R_{j-1}^io_j^{j-1}$ (check this by multiplying the matrices)

## 3.2 Denavit-Hartenberg Representation

- Each homogeneous transformation $A_i$ for link $i$ and joint $i$ is represented as a product of four basic transformations
- $A_i = Rot_{z,\theta_i} Trans_{z,d_i} Trans_{x, a_i} Rot_{x, \alpha_i}$
	- Thus, $A_i = \begin{bmatrix} c_{\theta_i} & -s_{\theta_i} c_{\alpha_i} & s_{\theta_i} s_{\alpha_i} & a_i c_{\theta_i} \\ s_{\theta_i} & c_{\theta_i} c_{\alpha_i} & -c_{\theta_i} s_{\alpha_i} & a_i s_{\theta_i} \\ 0 & s_{\alpha_i} & c_{\alpha_i} & d_i \\ 0 & 0 & 0 & 1 \end{bmatrix}$
	- Where $c_{\theta_i} = \cos\theta_i$, $s_{\theta_i} = \sin\theta_i$
	- $a_i$: link length
	- $\alpha_i$: link twist
	- $d_i$: link offset
	- $\theta_i$: joint angle
	- Three of these four values are fixed for each link ($\theta_i$ varies for revolute joint, $d_i$ varies for prismatic joint)
- Arbitrary homogeneous transformations $H = \begin{bmatrix} R & d \\ 0 & 1 \end{bmatrix} \in \mathbb{R}^{4\times 4}$ need six parameters: three for position and three Euler angles for orientation
	- DH only has four!
	- This is possible due to clever choice of origin and coordinate axes
- DH assumptions
	- **DH1: the axis $x_1$ is perpendicular to the axis $z_0$**
	- **DH2: the axis $x_1$ intersects the axis $z_0$**
	- The coordinate frames below satisfy DH1 and DH2 ![[3.2.png]]
	- **Under these conditions, there exists unique $a, d, \theta, \alpha$ such that $A = \begin{bmatrix} R_1^0 & o_1^0 \\ 0 & 1 \end{bmatrix} = Rot_{z,\theta} Trans_{z,d} Trans_{x, a} Rot_{x, \alpha}$**
	    - This is proved using the key idea that DH1 implies $x_1 \cdot z_0 = 0$ and DH2 implies the dispacement between $o_0$ and $o_1$ can be expressed as a linear combination of $x_1$ and $z_0$ ($o_1 = o_0 + d z_0 + a x_1$)
	- Physical interpretation (see above figure)
	    - $a$: distance between $z_0$ and $z_1$ along $x_1$
	    - $\alpha$: angle between $z_0$ and $z_1$ measured in a plane normal to $x_1$
			- Use right hand rule to get the positive direction
	    - $d$: distance between $o_0$ and the intersection of $x_1$ and $z_0$ along $z_0$
	    - $\theta$: angle between $x_0$ and $x_1$ measured in a plane normal to $z_0$
			- Use right hand rule to get the positive direction
- Assigning the coordinate frames
	- We can always choose frames for each link such that DH1 and DH2 are satisfied for any manipulator
	- **These choices are not unique, but the end matrix $T_n^0$ will be the same**
	- The final coordinate system $o_nx_ny_nz_n$ is the end effector or tool frame
	- **Steps**
		- Step 1: locate a label the joint axes $z_0, z_1, \ldots, z_{n-1}$
		- Step 2: Establish the base frame, setting the origin anywhere on the $z$ axis then choosing $x_0$ and $y_0$ to form a right-handed coordinate system
		- For $i = 1, \ldots, n-1$
			- Step 3: Locate the origin $o_i$ of where the common normal to $z_{i-1}$ and $z_i$ intersects $z_i$ (if $z_{i-1}$ and $z_i$ are nto coplanar). If they intersect, $o_i$ is at the intersection. If they are parallel, $o_i$ is at any convenient location along $z_i$
			- Step 4: Choose $x_i$ to be the common normal to $z_{i-1}$ and $z_i$ through $o_i$. If $z_{i-1}$ and $z_i$ intersect, $x_i$ is along the direction normal to the $z_{i-1} - z_i$ plane
			- Step 5: Choose $y_i$ to complete the right-handed coordinate system
		- Step 6: Establish end-effector frame $o_nx_ny_nz_n$ at the end of the manipulator
		- Step 7: Create table of link parameters $a_i, \alpha_i, d_i, \theta_i$ for $i = 1, \ldots, n$
		- Step 8: Write the transformation matrix $A_i$ for each link. Recall each $A_i$ is a function of a single joint variable
		- Step 9: Compute the transformation matrix $T_n^0 = A_1\cdots A_n$, giving the position and orientation of the end effector frame w.r.t. the base frame

## 3.3 Examples

- Planar elbow manipulator ![[3.3.png]]
- Link parameters
	- $a_1 = a_1$, $\alpha_1 = 0$, $d_1 = 0$, $\theta_1 = \theta_1$
	- $a_2 = a_2$, $\alpha_2 = 0$, $d_2 = 0$, $\theta_2 = \theta_2$
	- The only variables are $\theta_1, \theta_2$
