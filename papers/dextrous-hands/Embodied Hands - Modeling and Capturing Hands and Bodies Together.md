# Embodied Hands - Modeling and Capturing Hands and Bodies Together

**Authors**: Javier Romero, Dimitrios Tzionas, Michael J. Black

#dextrous-hands
#learning-from-video

[[papers/dextrous-hands/README#[2017-11] Embodied Hands - Modeling and Capturing Hands and Bodies Together|README]]

[Paper](http://arxiv.org/abs/2201.02610)
[Code](https://github.com/otaheri/MANO)
[Website](https://mano.is.tue.mpg.de/)
[Video](https://www.youtube.com/watch?v=_1o21xc3TD0)

## Abstract

> Humans move their hands and bodies together to communicate and solve tasks. Capturing and replicating such coordinated activity is critical for virtual characters that behave realistically. Surprisingly, most methods treat the 3D modeling and tracking of bodies and hands separately. Here we formulate a model of hands and bodies interacting together and fit it to full-body 4D sequences. When scanning or capturing the full body in 3D, hands are small and often partially occluded, making their shape and pose hard to recover. To cope with low-resolution, occlusion, and noise, we develop a new model called MANO (hand Model with Articulated and Non-rigid defOrmations). MANO is learned from around 1000 high-resolution 3D scans of hands of 31 subjects in a wide variety of hand poses. The model is realistic, low-dimensional, captures non-rigid shape changes with pose, is compatible with standard graphics packages, and can fit any human hand. MANO provides a compact mapping from hand poses to pose blend shape corrections and a linear manifold of pose synergies. We attach MANO to a standard parameterized 3D body shape model (SMPL), resulting in a fully articulated body and hand model (SMPL+H). We illustrate SMPL+H by fitting complex, natural, activities of subjects captured with a 4D scanner. The fitting is fully automatic and results in full body models that move naturally with detailed hand motions and a realism not seen before in full body performance capture. The models and data are freely available for research purposes in our website (<http://mano.is.tue.mpg.de>).

## Summary

- Formulate a model of hands and bodies interacting together and fit it to full-body 4D sequences
- MANO (hand Model with Articulated and Non-rigid defOrmations)
	- Factors geometric changes into 1) changes inherent to the identity of the subject and 2) those caused by pose
	- Trained to minimize vertex error in the training set
- Two contributions: learn a new model of the hand, and tracking hands and bodies together
	- These notes focus on the first
- Collect database of detailed hand scans of 31 subjects, 51 poses

## Background

- Previous hand models ![[hand_models.png]]
- Dimensionality reduction
	- Hands have many DoF but many are not independently controllable
	- In natural movements, hand poses are effectively low dimensional

## Method

- Two-stage approach for creating a dextrous full-body model
	- 1) Collect a large number of scans of hands in isolation
	- 2) Train a hand model using an iterative process of aligning a template to the scans using the model and learning a model from the registered scans
- Captured scans (total 2018) have ~50k vertices
- We minimize the distance between the scan and a registered mesh, V, with respect to the registration vertex locations, while keeping the registration likely according to the model
- Given this set of curated registrations, the goal is to learn the parameters of a SMPL-style hand model so that it fits the registrations
	- MANO contains 15 joints plus global orientation

## Results

- Hand pose embedding
	- In order to make the model practical for the purpose of scan registration, we will try to expose a set of parameters that efficiently explain the most common hand poses
	- 6, 10, 15 PCA components explain 81%, 90%, 95% of the full space
