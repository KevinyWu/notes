# Learning with 3D rotations, a hitchhiker's guide to SO(3)

**Authors**: A. René Geist, Jonas Frey, Mikel Zhobro, Anna Levina, Georg Martius

[[templates/README|README]]

[Paper](http://arxiv.org/abs/2404.11735)
[Code](https://github.com/martius-lab/hitchhiking-rotations)

## Abstract

> Many settings in machine learning require the selection of a rotation representation. However, choosing a suitable representation from the many available options is challenging. This paper acts as a survey and guide through rotation representations. We walk through their properties that harm or benefit deep learning with gradient-based optimization. By consolidating insights from rotation-based learning, we provide a comprehensive overview of learning functions with rotation representations. We provide guidance on selecting representations based on whether rotations are in the model's input or output and whether the data primarily comprises small angles.

## Summary

- **Our recommendations for neural network regression (gradient-based learning) with 3D rotations:**
	- **Changing the loss does not fix discontinuities** representations with three or four parameters introduce discontinuities into the target function when rotations are in the output
		- The subsequent issues arising in learning the target function are not fixed using distance picking or computing distances in $\mathrm{SO}(3)$
	- **For rotation estimation (rotations in model output)** use $\mathbb{R}^9+\mathrm{SVD}$ or $\mathbb{R}^6+\mathrm{GSO}$
		- If the regression targets are only small rotations, using quaternions with a halfspace-map is a good option
	- **For feature prediction (rotations in model input)** use rotation matrices
		- If under memory constraints, quaternions with a halfspace-map and data-augmentation are viable
- $f$ and $g$ are functions that map from rotation representation to/from rotation matrix in $\mathrm{SO}(3)$; need $f(g(R)) = R$ ![[rotations.png]]

## Background

- Let $r$ be the rotation representation and $R$ be the rotation matrix
- Recent works suggest rotation representations with four or less dimensions do not facilitate sample-efficient learning
- **Problem formulation**
	- Given data $\mathcal{D} = \{x_{i}, y_{i}\}_{i=1}^{N}$ find the parameters $\theta$ of the neural network $h: X \rightarrow Y$ that minimize the loss function $L(\mathcal{D}, \theta) = \sum\limits_{x,y\in \mathcal{D}} d(y, h(x, \theta))$
	- Using the parameter gradient $\nabla_{\theta}L$
- For simplicity; consider cases where rotations occur in the input
	- $\mathcal{A}$ is the feature space; i.e. camera image, point cloud, etc
	- $a = h(r)$; i.e. $X = \mathcal{R}, Y = \mathcal{A}$, like rendering an object from a particular direction or predicting dynamics
	- $r = h(a)$; i.e. $X = \mathcal{A}, Y = \mathcal{R}$, like pose estimation from images
- Representations of rotation
	- SO(2) representations
		- Single angle $\alpha$ (jump in function $g$)
		- No discontinuity in $(\cos \alpha, \sin \alpha)$
	- Euler angles
		- $\alpha, \beta, \gamma$ describe a rotation; $f(r) := R_{3}(\gamma)R_{2}(\beta)R_{1}(\alpha)$
		- Discontinuities when bounding range of angles
		- Same point in $\mathrm{SO}(3)$ can be described by difference representations
			- $[0, \frac{\pi}{2,}0]$ same as $[-\pi/2, \pi/2, -\pi/2]$
		- **Discouraged to use these**
	- Exponential coordinates
		- Rotation axis $\omega\in \mathbb{R}^3$ and an angle
		- User $\|\omega\|$ to encode the angle
		- Can also be expressed by two different vectors
	- Axis angle and quaternions
		- Make $\omega\in \mathbb{R}^3$ a unit vector and have $r = (\omega, \alpha)\in \mathbb{R}^4$
			- Note only two numbers are needed to express the vector as the magnitude is constrained to 1, so axis-angle actually in $\mathbb{R}^3$
		- Transform exponential coordinates to axis angles, then axis angle vector to $\mathrm{SO}(3)$ with Rodrigues' rotation formula
			- Rodrigues' rotation formula:
				- $f(r) := \mathbb{I} + \sin(\alpha)[\omega]_{\times}+ (1cos(\alpha))[\omega]_{\times}^2$
				- $[\omega]_{\times} = \begin{bmatrix}0 & -\omega_z & \omega_y \\\omega_z & 0 & -\omega_x \\-\omega_y & \omega_x & 0\end{bmatrix}$is the skew-symmetric matrix (cross-product matrix) of $\omega$
				- i.e. $[\omega]_{\times}v = \omega \times v$
				- Intuitively the first term keeps the part of $v$ that's unchanged, the second term rotates perpendicular components by sine of the angle, the third term completes the rotation to the full cosine–sine relation
				- Derived from matrix exponential $R = \exp(\alpha [\omega]_{\times}) = \mathbb{I} + \sin(\alpha)[\omega]_{\times}+ (1cos(\alpha))[\omega]_{\times}^2$
			- [Rodrigues' formula](https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula)
		- Quaternions extend complex numbers to higher dimensions
			- Unit quaternions double cover $\mathrm{SO}(3)$ such that $f(q) = f(-q)$
	- $\mathbb{R}^6$ + Gram-Schmidt orthonormalization (GSO)
		- GSO turns a set of linearly independent vectors into an orthonormal basis spanning the same subspace
		- $r = (v_{1}, v_{2}) \in \mathbb{R}^{3\times 2}$
		- $R = f(r) = \mathrm{GSO}(v_{1},v_{2}))$
		- $g(R) = \mathrm{diag} (1,1,0)R$
		- From [Zhou et al., 2019](https://arxiv.org/pdf/1812.07035): ![[gso.png]]
			- Cross product to get 3rd column because rotation matrices are orthogonal
	- $\mathbb{R}^{9}$ + singular value decomposition (SVD)
		- Given $r = M\in \mathbb{R}^{3\times 3}$, project onto $\mathrm{SO}(3)$
		- SVD decomposes the matrix into $M = U\Sigma V^T$ where $U, V\in \mathbb{R}^{3\times 3}$ are rotations or reflections and $\Sigma =\mathrm{diag}(\sigma_{1}, \sigma_{2}, \sigma_{3})$ is a diagonal matrix with singular values denoting scaling parameters
			- Rotations have determinant +1
			- Reflections (improper) have determinant -1
			- **Magnitude of determinant represents the factor by which volume is scaled by transformations with the matrix**
			- **The sign of the determinant indicates whether the transformation preserves or reverses the orientation of space**
		- $f(r) = \mathrm{SVD}^+(M)=U\mathrm{diag}(1,1,\det(UV^T))V^T$
			- $\det(UV^{T})$ ensures that $\det(SVD^+(M))=1$
			- This is because determinant of a matrix is $\det(U)(\prod_{i}\sigma_{i})\det(V)$
			- **This operation $f$ finds the rotation matrix with the least-squares distance to $M$**
		- $g(R) = R$
	- Summary ![[rotation_representations.png]]

## Method

- Distance metrics in rotation learning
	- Need to measure in either $\mathrm{SO}(3)$ or $\mathcal{R}$

## Results

- Notable results from the paper
