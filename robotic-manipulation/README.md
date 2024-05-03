# Robotic Manipulation

[Full course notes](https://manipulation.csail.mit.edu/)

[Lectures (fall 2023)](https://youtube.com/playlist?list=PLkx8KyIQkMfWr191lqbN8WfV08j-ui8WX&feature=shared)

[Companion code](https://github.com/RussTedrake/manipulation)

Notes from the course "Robotic Manipulation" taught by Russ Tedrake at MIT.

- [Robotic Manipulation](#robotic-manipulation)
  - [1 Introduction](#1-introduction)
  - [2 Let's get you a robot](#2-lets-get-you-a-robot)

## 1 Introduction

[Lecture 1](https://youtu.be/v04rn86Dehg?feature=shared)

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

## 2 Let's get you a robot

[Lecture 2](https://youtu.be/q896_lTh8eA?feature=shared)

- 
