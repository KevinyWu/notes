# Perspectives and problems in motor learning

**Authors**: Daniel M Wolpert, Zoubin Ghahramani, J.Randall Flanagan

#neuroscience

[[papers/neuroscience/README|README]]

[Paper](https://linkinghub.elsevier.com/retrieve/pii/S1364661300017733)

## Abstract

> Movement provides the only means we have to interact with both the world and other people. Such interactions can be hard-wired or learned through experience with the environment. Learning allows us to adapt to a changing physical environment as well as to novel conventions developed by society. Here we review motor learning from a computational perspective, exploring the need for motor learning, what is learned and how it is represented, and the mechanisms of learning. We relate these computational issues to empirical studies on motor learning in humans.

## Summary

- The goal of learning is to improve performance
- Learning is the only mechanism fast enough to allow us to master new tasks
- Although much of our motor repertoire is acquired during our lifetime, we do not start life with a motor tabula rasa - ex. blind and deaf children do facial gestures
- "Motor learning is a consequence of the co-adaptation of the neural machinery and structural anatomy"

## Background

- Brain is a processing system that converts inputs (sensory information) to outputs (motor commands on muscles)
	- Motor learning: process of transforming sensory inputs into consequent motor outputs
	- Transformation: kinematic + dynamic (forward and inverse)
	- Internal forward dynamic model is a model within the brain that can predict the sensory consequence of an action

## Method

- Three ways for learning system to interact with environment
	- Supervised (teaching)
		- Self-supervised: provide internal goals
		- Distal supervised learning: uses a forward internal model of the system to convert sensory errors into required changes to the motor command
		- Feedback error: learns a feedback controller to achieve the same thing as distal
		- Relationship between the inputs and outputs of an inverse model can be one-to-many
	- Reinforcement
		- No target behavior, only reward
		- Rewards can depend on history of past motor commands
		- In the presence of noise, the same sequence of motor commands will lead to a probability distribution over movements
	- Unsupervised
		- Ex. Hebbian learning rule: strength of a connection is increased when there is a coincidence of the firing of the pre-synaptic and post-synaptic neuron
		- Ex. PCA
- Supervised and unsupervised learning can be seen as using Bayes rule to combine the current model ('the prior') with new data ('the evidence') to generate an updated model ('the posterior')
- Why is motor control difficult?
	- Latency in sensory information
	- Control problem very high dimensional: both observation and action space
		- Generate lower dimensional representations
- How is motor learning represented?
	- Lookup tables: not generalizable
	- Parametric equations: kinematics equations
		- Not very flexible due to small number of input parameters
- Many situations that we encounter are derived from a combination of previously experienced situations, such as novel conjoints of manipulated objects and environments
	- Internal models are "motor primitives" that can be used to construct intricate motor behaviors
