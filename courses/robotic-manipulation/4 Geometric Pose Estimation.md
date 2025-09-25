# Geometric Pose Estimation

[[courses/robotic-manipulation/README#[4] Geometric Pose Estimation|README]]

[Lecture 6](https://www.youtube.com/watch?v=1a3KhOq1938&list=PLkx8KyIQkMfWr191lqbN8WfV08j-ui8WX&index=8)
[Lecture 7](https://www.youtube.com/watch?v=Cs49WVNqEdk&list=PLkx8KyIQkMfWr191lqbN8WfV08j-ui8WX&index=9)
[Notes](https://manipulation.csail.mit.edu/pose.html)
[Deepnote](https://deepnote.com/workspace/bubhub-afbb4b47-384c-4b93-a423-6aad7f9e29f7/project/04-Geometric-Pose-Estimation-24486d11-2a0b-4aaa-800e-8fe1eda187f6)

## 4.1 Cameras and Depth Sensors

- Improvements in cameras have made estimation easier, with better resolution and frame rate
- RGB-D cameras
	- Intel RealSense D415
	- Far from ideal: lighting conditions, material properties, color channels are noisy
	- Occlusions: cameras only get line of sight
	- Cluttered scenes will occlude object of interest
	- Ideal range: 0.3-1.5 meters, so if too close or too far, depth data is unreliable
	- Partial solution is to put as many cameras as we can in the scene

## 4.2 Representations for Geometry

- Many different representations possible
- We focus on depth images and point clouds
- Depth image
	- Each pixel value is a single number that represents the distance between the camera and the nearest object in the scene
	- Combining this with the camera intrinsics, we can get a 3D point cloud, $s_i$ ("scene points")
- **Point cloud, $s$**
	- These points have a pose and color value
	- Pose of each point is defined in camera frame: $^CX^{s_i}$
	- Conversion from depth image to point cloud does lose some information: information about the ray that was cast from the camera to arrive at that point
	- In addition to declaring "there is geometry at this point", the depth image combined with the camera pose also implies that "there is no geometry in the straight line path between camera and this point"

## 4.3 Point Cloud Registration with Known Correspondences

- **Objective**: we have a known object in the robot's workspace and we have a point cloud from depth cameras; how do we estimate the pose of the object $X^O$?
	- First point cloud: object described by **model points** $m_i$, their pose in object frame $^OX^{m_i}$, total $N_m$ points
	- Second point cloud: scene points $s_i$, their pose in camera frame $^CX^{s_i}$, total $N_s$ points
- **Point cloud registration**: aligning two point clouds
	- Assumption 1: we know pose of camera $X^C$ (i.e. by solving forward kinematics of arm for wrist camera)
		- Camera calibration very important, small errors lead to large artifacts
	- Assumption 2: **"known correspondences"**
		- For each point in scene point cloud, we can pair it with a point in the model point cloud
		- Not a realistic assumption in practice
		- Correspondence vector $c\in [1, N_m]^{N_s}$: $c_i = j$ means $s_i$ corresponds to $m_j$
	- **Point cloud registry is just inverse kinematics:** $^Wp^{m_{c_i}} = ^WX^O \cdot ^Op^{m_{c_i}} = ^WX^C \cdot ^Cp^{s_i}$
		- Only one unknown: $^WX^O$
		- Differential kinematics easier than inverse kinematics, but we need to solve this inverse kinematics problem at least once
	- **Least squares solution:** $\min_{X\in \text{SE}(3)} \sum_{i=1}^{N_s} ||X \cdot ^Op^{m_{c_i}} - X^C\cdot ^Cp^{s_i}||^2$
		- $\text{SE}(3)$: special Euclidean group, i.e. denotes that $X$ must be a valid rigid transform
	- Ex. using rotation matrices
		- $\min_{p\in \mathbb{R}^3, R\in \mathbb{R}^{3\times 3}} \sum_{i=1}^{N_s} ||p + R \cdot ^Op^{m_{c_i}} - X^C\cdot ^Cp^{s_i}||^2$ subject to
			- $R^T = R^{-1}$
			- $\det(R) = 1$ (if $\det(R) = -1$, we have an "improper" rotation; i.e. a reflection across the axis of rotation)
- **The relative position between points is affected by rotation but not by translation**
	- We can subtract the **central point** and make the problem translation-invariant, then add it back at the end
	- For least squares, this is the center of mass $\bar{m}, \bar{s}$
	    - $^Op^{\bar{m}} = \frac{1}{N_m} \sum_{i=1}^{N_s} (^Op^{m_{c_i}})$
	    - $^Cp^{\bar{s}} = \frac{1}{N_s} \sum_{i=1}^{N_s} (^Cp^{s_i})$
- **Elegant numerical solution to least squares problem: singular value decomposition (SVD)**
	- $W = \sum_{i=1}^{N_s} (p^{s_i} - p^{\bar{s}})(^Op^{m_{c_i}} - ^Op^{\bar{m}})^T$
	- SVD: $W = U\Sigma V^T$
	- Optimal solution: $R^* = UDV^T$, $p^* = p^{\bar{s}} - R^* \cdot ^Op^{\bar{m}}$
	- $D$ is a diagonal matrix with entries $[1,1,\det(UV^T)]$

## 4.4 Iterative Closest Point (ICP)

## 4.5 Dealing with Partial Views and Outliers

## 4.6 Non-Penetration and "Free-Space" Constraints
