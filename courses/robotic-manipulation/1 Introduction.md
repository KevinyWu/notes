# Introduction

[[courses/robotic-manipulation/README#[1] Introduction|README]]

[Lecture 1](https://youtu.be/v04rn86Dehg?feature=shared)
[Notes](https://manipulation.csail.mit.edu/intro.html)
[Deepnote](https://deepnote.com/workspace/bubhub-afbb4b47-384c-4b93-a423-6aad7f9e29f7/project/01-Introduction-bdfeaeb4-e107-472c-a8e7-6848fbd990d0)

- Manipulation is more than pick-and-place
	- 80s and 90s: manipulation referred to pick-and-place and grasping
	- Now, manipulation is broader: buttoning shirt, spreading peanut butter, etc.
- Open-world problem: the world has infinite variability
	- Diversity in open-world problems might make the problem easier
	- For example, now we need quirky solutions to specific problems
	- These quirky solutions may be discarded when the landscape is more diverse
- Simulation
	- Modern simulators can even train models and expect them to work in the real world
	- Models like transformer more general: won't overfit to quirks of simulator image data
	- [Drake](http://drake.mit.edu/) is a simulator that emphasizes the governing equations of motion and physics
