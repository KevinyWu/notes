# Practice Makes Perfect: Planning to Learn Skill Parameter Policies

**Authors**: Nishanth Kumar, Tom Silver, Willie McClinton, Linfeng Zhao, Stephen Proulx, Tomás Lozano-Pérez, Leslie Pack Kaelbling, Jennifer Barry

#tags

[[papers/robot-learning/README#[2024-02] Practice Makes Perfect Planning to Learn Skill Parameter Policies|README]]

[Paper](https://arxiv.org/abs/2402.15025v2)
[Code](https://github.com/bdaiinstitute/predicators/releases/tag/planning-to-practice-ees)
[Website](http://ees.csail.mit.edu/)
[Video](https://www.youtube.com/watch?v=123DXatw1V8)

## Abstract

> One promising approach towards effective robot decision making in complex, long-horizon tasks is to sequence together parameterized skills. We consider a setting where a robot is initially equipped with (1) a library of parameterized skills, (2) an AI planner for sequencing together the skills given a goal, and (3) a very general prior distribution for selecting skill parameters. Once deployed, the robot should rapidly and autonomously learn to improve its performance by specializing its skill parameter selection policy to the particular objects, goals, and constraints in its environment. In this work, we focus on the active learning problem of choosing which skills to practice to maximize expected future task success. We propose that the robot should estimate the competence of each skill, extrapolate the competence (asking: "how much would the competence improve through practice?"), and situate the skill in the task distribution through competence-aware planning. This approach is implemented within a fully autonomous system where the robot repeatedly plans, practices, and learns without any environment resets. Through experiments in simulation, we find that our approach learns effective parameter policies more sample-efficiently than several baselines. Experiments in the real-world demonstrate our approach's ability to handle noise from perception and control and improve the robot's ability to solve two long-horizon mobile-manipulation tasks after a few hours of autonomous practice. Project website: <http://ees.csail.mit.edu>

## Summary

- Notes from the introduction and conclusion sections

## Background

- Notes about the background information

## Method

- Notes about the method

## Results

- Notable results from the paper
