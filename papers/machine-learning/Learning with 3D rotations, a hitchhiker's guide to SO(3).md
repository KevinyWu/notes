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

- Recent works suggest rotation representations with four or less dimensions do not facilitate sample-efficient learning
- **Problem formulation**
	- Given data $\mathcal{D} = \{x_{i}, y_{i}\}_{i=1}^{N}$ find the parameters $\theta$ of the neural network $h: X \rightarrow Y$ that minimize the loss function $L(\mathcal{D}, \theta) = \sum\limits_{x,y\in \mathcal{D}} d(y, h(x, \theta))$
	- Using the parameter gradient $\nabla_{\theta}L$
- For simplicity; consider cases where rotations occur in the input
	- $A$ is the feature space; i.e. camera image, point cloud, etc
	- $a = h(r)$; i.e. $X = R, Y = A$, like rendering an object from a particular direction or predicting dynamics
	- $r = h(a)$; i.e. $X = A, Y = R$, like pose estimation from images
- Representations of rotation
	- SO(2) representations
	- Euler angles
	- Exponential coordinates
	- Axis angle and quaternions
	- $\mathbb{R}^6$ + Gram-Schmidt orthonormalization (GSO)
	- $\mathbb{R}^{9}$ + singular value decomposition (SVD)
	- 

## Method

- ![[rotation_representations.png]]

## Results

- Notable results from the paper
