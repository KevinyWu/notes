# Tactile-Augmented Radiance Fields

**Authors**: Yiming Dou, Fengyu Yang, Yi Liu, Antonio Loquercio, Andrew Owens

#3d-representation
#tactile

[[papers/3d-representation/README#[2024-05] Tactile-Augmented Radiance Fields|README]]

[Paper](http://arxiv.org/abs/2405.04534)
[Code](https://github.com/Dou-Yiming/TaRF)
[Website](https://dou-yiming.github.io/TaRF/)

## Abstract

> We present a scene representation, which we call a tactile-augmented radiance field (TaRF), that brings vision and touch into a shared 3D space. This representation can be used to estimate the visual and tactile signals for a given 3D position within a scene. We capture a scene's TaRF from a collection of photos and sparsely sampled touch probes. Our approach makes use of two insights: (i) common vision-based touch sensors are built on ordinary cameras and thus can be registered to images using methods from multi-view geometry, and (ii) visually and structurally similar regions of a scene share the same tactile features. We use these insights to register touch signals to a captured visual scene, and to train a conditional diffusion model that, provided with an RGB-D image rendered from a neural radiance field, generates its corresponding tactile signal. To evaluate our approach, we collect a dataset of TaRFs. This dataset contains more touch samples than previous real-world datasets, and it provides spatially aligned visual signals for each captured touch signal. We demonstrate the accuracy of our cross-modal generative model and the utility of the captured visual-tactile data on several downstream tasks. Project page: <https://dou-yiming.github.io/TaRF>

## Summary

- Collecting tactile data is expensive: requires physical interaction with environment
- Mount a touch sensor to camera to collect large real-world data set of **aligned visual-tactile data**
- Cross-modal prediction models can accurately estimate touch from sight for natural scenes
- Combine **sparse estimates** of touch with quasi-dense tactile signals estimated using diffusion models
- **Limitation**: touch sensor is zoomed in, so misalignments occur

## Background

- Previous work uses NeRF and captured touch data to generate a tactile field for several small objects
- Recent methods localize objects in NeRFs using joint embeddings between images and language

## Method

- Human data collector moves through a scene and records a video, then construct NeRF
    - Simultaneously collect touch signal on the mounted **vision-based tactile sensor** sensor
    - Find relative 6-DOF $(R, t)$ pose between camera and touch sensor with a braille board
    - Find $(R,t)$ by minimizing reprojection error
- Training
    - First, pre-train a cross-modal visual-tactile encoder with self-supervised contrastive learning on our dataset
    - We encode visual and tactile data into latent vectors in the resulting shared representation space
    - Second, train the diffusion model from scratch and pre-train it on the task of unconditional tactile image generation on the YCB-Slide dataset
- Use a **diffusion model** to estimate the touch signal (represented as an image from a vision-based touch sensor) for other locations within the scene
    - Dense touch estimation process ![[tarf-method.jpg]]
        - Evaluated against ground truth using **Frechet Inception Distance**

## Results

- Dataset has 19.3k temporally aligned vision-touch image pairs across 13 scenes
- Ablations
    - Removing RGB results in large performance drop
    - Removing depth image or contrastive pretraining has small effect on CVTP but results in a drop on FID
- **Tactile localization**
    - Given a tactile signal, find the corresponding regions in a 2D image or in a 3D scene that are associated with it
- **Material classification**
    - Three subtasks
        - Classify material into one of 20 classes
        - Softness classification: hard or soft
        - Hardness classification: rough or smooth
    - This dataset improves previous baselines, showing that it covers wide range of materials
