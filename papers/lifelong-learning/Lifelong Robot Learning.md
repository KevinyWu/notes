# Lifelong Robot Learning

**Authors**: Sebastian Thrun, Tom M. Mitchell

#lifelong-learning
#q-learning

[[papers/lifelong-learning/README#[1995-01] Lifelong Robot Learning|README]]

[Paper](https://www.ri.cmu.edu/publications/lifelong-robot-learning/)

## Abstract

> Learning provides a useful tool for the automatic design of autonomous robots. Recent research on learning robot control has predominantly focused on learning single tasks that were studied in isolation. If robots encounter a multitude of control learning tasks over their entire lifetime there is an opportunity to transfer knowledge between them. In order to do so, robots may learn the invariants of the individual tasks and environments. This task-independent knowledge can be employed to bias generalization when learning control, which reduces the need for real-world experimentation. We argue that knowledge transfer is essential if robots are to learn control with moderate learning times in complex scenarios. Two approaches to lifelong robot learning which both capture invariant knowledge about the robot and its environments are presented. Both approaches have been evaluated using a HERO-2000 mobile robot. Learning tasks included navigation in unknown indoor environments and a simple find-and-fetch task.

## Summary

- Why learning?
	- Limitations of assuming accurate a priori knowledge of the robot and environment: bottlenecks in knowledge, engineering, tractability, precision
	- Robots need to face unknown, hostile environments and explore autonomously and recover from failures
		- Internal prior knowledge too weak to solve a concrete problem offline
	- 3 aspects of learning: exploration, generalization, prior knowledge about the world
	- **All empirically learned knowledge is grounded in the real world**
- Tackled two types of lifelong robot learning scenarios
	- Single environment, multiple tasks
	- Multiple environments
	- Exploiting previously learned knowledge simplifies learning control

## Background

- Robot learning problem
	- $W$ world, $W:Z\times A \rightarrow Z$
	- $z \in Z$ state
	- $S$ set of possible sensations through sensors
	- $A$ set of all actions
	- Reward function $R: S \rightarrow \mathbb{R}$
- Control learning problem:
	- $\langle S,A,W,R \rangle \rightarrow F: S^{*}\rightarrow A$
	- Such that control function $F$ maximizes $R$ over time
- Assumptions
	- In RL, restrictive assumption that the robot can sense the world reliably
	- Even if robots have access to complete state descriptions, learning control in complex robot with large state spaces is infeasible because it takes too much experimentation to acquire knowledge needed for maximizing reward
	- **Real-world experimentation will be the central bottleneck of any general learning technique that does not utilize prior knowledge about the robot and its environment**
- Lifelong learning problem: learn a collection of control policies for a variety of tasks
	- Constant $A$ and $S$; $W$ and $R$ may vary
	- $\{\langle S,A,W_i,R_i \rangle \rightarrow F_{i}| F_i: S^{*}\rightarrow A\}$ such that $F_i$ maximizes $R_i$ over time
	- Opportunity to do better than handling each problem individually
		- This predicted what [[Russ Tedrake - Multitask Transfer in TRI’s Large Behavior Models for Dexterous Manipulation|LBM]] showed: pretraining on diverse tasks is better than single-task policies
		- Motivates "bootstrapping" (meaning using existing knowledge/experience) learning algorithms that transfer knowledge across tasks
		- Similar to humans, robots should first learn simple tasks and then transfer that knowledge to more complex tasks
- Related work
	- Learning models
		- Sutton 1990
		- Lin 1992
	- Learning behaviors and abstractions
		- Behaviors are abstract action spaces, which is smaller than the number of actions
	- Learning inductive function approximation bias
		- Directly learning inductive bias of function approximators to learn control directly
	- Learning representations

## Method

- Learning action models
	- Action models are functions of the type $M:S\times A \rightarrow S$
	- Each individual control learning problem requires a different policy
	- Decompose the problem of learning $F_i$ into the problem of learning an **evaluation function** $Q_i$
		- $Q_{i}:S\times A \rightarrow \mathbb{R}$ is the expected future cumulative reward after executing action $a$ in state $s$
		- Optimal action $a^{*}= \arg\max_{a\in A} Q_{i}(s,a)$
		- Can be learned from training samples that succeed at a task
- EBNN: Explanation-Based Neural Network
	- Subproblem: **same environment, different tasks**
	- Assume the agent has learned approximately accurate action models (now called world models) that model effect of actions on environment state
		- **Key idea: train neural network to match both outputs and the derivatives of the action model**
		- $L_{EBNN} = \sum_{i}\left(\alpha(f_{\theta}(x_{i})- y_{i})^2+\beta\sum_{j}\left(\frac{\partial f_{\theta}(x_{i})}{\partial x_{j}}- \frac{\partial \hat{f}(x_{i})}{\partial x_{j}}\right)\right)$
			- Here $f$ is $Q_{i}:S\times A \rightarrow \mathbb{R}$
			- Need partial derivatives of $Q_i$ with respect to states
		- Opposed to standard supervised learning that only learns to match outputs
		- NN training guided by approximate physics model
		- Derivatives generalize across tasks, so robot can transfer knowledge to new problems
	- Imperfect action models may mislead generalization
	- **How can a robot agent deal with incorrect prior knowledge?**
		- Step size for weight updates is multiplied by the estimated slow accuracy when learning slopes
		- With inaccurate action models EBNN will not perform better than a purely inductive learning procedure (learning from scratch)
			- When the agent is more "mature" and has learned domain-specific bias, transfer learning applies
- Lifelong learning in multiple environments
	- Seems harder, but even across environments there are invariants that may be learned and used as a bias
		- Same robot, same end-effectors, same sensors
	- Learning to interpret sensations: navigation
		- Exploration task: HERO-2000 has a rotating sonar sensor and wheel encoders to detect stuck or slipping wheels
			- **Idea: robot does not know what sonar signals mean; there are no heuristics**
		- Negative reward for collision, positive reward for entering unexplored areas
		- Sensations are 24-dim sonar scans and a single bit that encodes wheel state
			- Network $\mathcal{R}$ maps a single sonar scan to probability of collision
			- Input is vector of sonar values and coordinates of the query point relative to the robot's local coordinate frame
				- $(s, \Delta x, \Delta y)$
			- Output 1 means predicted collision, 0 if predicted free-space
			- After collecting training demos by exploration, can $\mathcal{R}$ using supervised learning
	- Building maps
		- Confidence network $\mathcal{C}$
		- When constructing maps from the small local maps in the previous experiments, some points in the world will have conflicting sensations
			- Sonar sensors are noisy
			- Sonar is blind behind obects
		- Input to $\mathcal{C}$ the same as $\mathcal{R}$
			- Target output is the normalized prediction error of $\mathcal{R}$
			- Thus, $\mathcal{C}$ predicts the expected deviation of the sensor interpretations, denoted $\mathcal{C}(s, \Delta x, \Delta y)$
			- Confidence of $\mathcal{R}(s, \Delta x, \Delta y)$ given by $-\ln\mathcal{C}(s, \Delta x, \Delta y)$

## Results

- EBNN
	- Picking up a cup: six training episodes collected
		- Plain learning procedure predicts positive reward solely based on the andlge of the cup
		- EBNN learned that grasping will fail if the cup is too far away
		- Effect of slopes: the evaluation functions learned with EBNN discovered the correlation of the distance of the cup and the success of the grab action from the neural network action model
	- EBNN is a method for lifelong agent learning, since it learns and re-uses learned knowledge that is independent of the particular control learning problem at hand
	- Generalization improvement scales linearly with the number of input features
- Learning in multiple environments
	- Designed robot simulator
	- Collected 8,192 training examples
	- Network learns how to interpret sonar signals
		- Maps small signals (meaning sonar signal bounced back early) to obstacles nearby
	- Also learned invariants in training environments, like typical wall sizes
	- Building maps succeeded in an autonomous navigation competition
