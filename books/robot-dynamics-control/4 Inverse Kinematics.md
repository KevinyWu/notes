# Inverse Kinematics

#kinematics
#inverse-kinematics

[[books/robot-dynamics-control/README#[4] Inverse Kinematics|README]]

## 4.1 The General Inverse Kinematics Problem

- General inverse kinematics problem: **given a $4\times 4$ homogeneous transformation matrix $H$, find the joint variables $q_i$ that achieve this transformation**
	- $T_n^0(q_1, \ldots, q_n) = A_1(q_1)\cdots A_n(q_n) = H = \begin{bmatrix} R & o \\ 0 & 1 \end{bmatrix}$
	- Twelve nonlinear equations in $n$ variables (the bottom row of $T_n^0$ and $H$ are $[0,0,0,1]$, which is trivial)
	- May or may not have a solution
		- If a solution exists, may not be unique
- Want to find a closed for solution (rather than numerical)
	- $q_k = f_k(h_{11}, \ldots, h_{34}), k = 1, \ldots, n$
	- Faster for real-time control

## 4.2 Kinematic Decoupling

- For a 6-DOF arm with spherical wrist, you can decouple inverse kinematics problem into **inverse position kinematics** and **inverse orientation kinematics**
	- $R_6^0(q_1, \ldots, q_6) = R$
	- $o_6^0(q_1, \ldots, q_6) = o$
	- Given $o$ and $R$, the desired position and orientation of the end effector
- Spherical twist assumption means that axes $z_3, z_4, z_5$ intersect at **wrist center** $o_c$
	- Motion of the final three links about these axes doesn't change position of $o_c$, so position of the wrist center is a function of only first three joint variables
	- Origin of tool frame obtained by translation of $d_6$ along $z_5$ from $o_c^0$: $o = o_c^0 + d_6R[0,0,1]^T$
	- **Thus, wrist center:** $o_c^0 = o - d_6R[0,0,1]^T$
		- Or $\begin{bmatrix} x_c \\ y_c \\ z_c \end{bmatrix} = \begin{bmatrix} o_x \\ o_y \\ o_z \end{bmatrix} - d_6\begin{bmatrix} r_{13} \\ r_{23} \\ r_{33} \end{bmatrix}$
- Using the previous equation we can find the values of the first three joint variables that determine $R_3^0$
	- Then we can solve $R = R_3^0R_6^3$, so $R_6^3 = (R_3^0)^{-1}R = (R_3^0)^TR$ ![[4.2.png]]

## 4.3 Inverse Position: A Geometric Approach

- How to find the values of the first three joint variables that determine $R_3^0$
- Consider the manipulator below, where $o_c$ is given by $[x_c, y_c, z_c]^T$ ![[4.3.png]]
- $\theta_1 = \tan^{-1}\left(\frac{y_c}{x_c}\right)$
- Law of cosines: $\cos \theta_3 = \frac{r^2 + s^2 - a_2^2 - a_3^2}{2a_2a_3} = \frac{x_c^2 + y_c^2 - d^2 + z_c^2 - a_2^2 - a_3^2}{2a_2a_3} := D$
	- $\theta_3 = \cos^{-1}D$
- $\theta_2 = \tan^{-1}(r,s) - \tan^{-1}(a_2 + a_3c_3, a_3s_3) = \tan^{-1}(\sqrt{x_c^2 + y_c^2 - d^2}, z_c) - \tan^{-1}(a_2 + a_3c_3, a_3s_3)$

## 4.4 Inverse Orientation

- We can use the Euler angle representation to solve for $\phi = \theta_4, \theta = \theta_5, \psi = \theta_6$
