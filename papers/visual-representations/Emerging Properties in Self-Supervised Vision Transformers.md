# Emerging Properties in Self-Supervised Vision Transformers

**Authors**: Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, Armand Joulin

[[papers/visual-representations/README#[2021-04] Emerging Properties in Self-Supervised Vision Transformers|README]]

[Paper](http://arxiv.org/abs/2104.14294)
[Code](https://github.com/facebookresearch/dino)
[Video](https://www.youtube.com/watch?v=h3ij3F3cPIk)
[Blog](https://ai.meta.com/blog/dino-paws-computer-vision-with-self-supervised-transformers-and-10x-more-efficient-training/)

## Abstract

> In this paper, we question if self-supervised learning provides new properties to Vision Transformer (ViT) that stand out compared to convolutional networks (convnets). Beyond the fact that adapting self-supervised methods to this architecture works particularly well, we make the following observations: first, self-supervised ViT features contain explicit information about the semantic segmentation of an image, which does not emerge as clearly with supervised ViTs, nor with convnets. Second, these features are also excellent k-NN classifiers, reaching 78.3% top-1 on ImageNet with a small ViT. Our study also underlines the importance of momentum encoder, multi-crop training, and the use of small patches with ViTs. We implement our findings into a simple self-supervised method, called DINO, which we interpret as a form of self-distillation with no labels. We show the synergy between DINO and ViTs by achieving 80.1% top-1 on ImageNet in linear evaluation with ViT-Base.

## Summary

- Self-supervised ViT features explicitly contain the scene layout and, in particular, object boundaries; this information is directly accessible in the self-attention modules of the last block
- DINO: self-**di**stillation with **no** labels
- Directly predicts the output of a teacher network
- **DINO only works with a centering and sharpening of the teacher output to avoid collapse**
- BYOL: Bootstrap Your Own Latent; features are trained by matching them to representations obtained with a momentum encoder

## Background

- Momentum encoder
    - Target (teacher) network: main encoder whose parameters are updated slowly
    - Query (student) network: encoder whose parameters are updated quickly through backpropagation during training
    - Parameter update: $\theta_{\text{t}} \leftarrow m\theta_{\text{t}} + (1-m)\theta_{\text{s}}$, momentum $m$ is close to 1 for stability and smoothness
    - MoCo uses this idea
    - Advantages: stability in learning, consistency in representations especially with unlabeled data
- Knowledge distillation idea from SimCLRv2
    - DINO's teacher is dynamically built during training rather than pretrained
    - This way, knowledge distillation is not a post-processing step but a self-supervised learning task
- **Knowledge distillation**
    - A learning paradigm to train a student network $g_{\theta_{s}}$ to mimic the behavior of a teacher network $g_{\theta_{t}}$
    - Given an input image $x$, the both networks output probability distributions over $K$ dimensions denoted by $P_s$ and $P_t$
		- Normalized with softmax
		- Temperature parameter $\tau$ that controls the sharpness of the distribution
    - Given a fixed teacher, learn by minimizing cross-entropy loss between the student and teacher outputs
		- $\mathcal{L} = \min_{\theta_s} H(P_t(x), P_s(x))$, where $H(a, b) = -a\log b$

## Method

- Dino adapts **knowledge distillation** for self-supervised learning ![[dino.png]]
- Construct different crops (distorted views) of the same image
	- Two large global views $x_1^g, x_2^g$ and several smaller local views
	- All crops passed to student, only global views passed to teacher, encouraging local-to-global correspondence
	- $\mathcal{L} = \min_{\theta_s} \sum_{x\in \{x_1^g, x_2^g\}} \sum_{x'\in V, x'\neq x} H(P_t(x), P_s(x'))$
- Teacher network
	- exponential moving average (EMA) on the student weights
	- $\theta_{\text{t}} \leftarrow m\theta_{\text{t}} + (1-m)\theta_{\text{s}}$
- Network architecture
	- Backbone $f$ is ViT or ResNet
	- Projection head $h$ is a 3-layer MLP
	- Note that ViT does not use batchnorm, so we also don't use batchnorm in the projection head
- Avoiding collapse
	- **Collapse**: failure to capture the diversity and complexity of the data, instead learns to output the nearly constant representation for all inputs
	- Centering and sharpening the teacher output
	- **Centering**: subtract the mean of the teacher output
        - $g_t(x) \leftarrow g_t(x) + c$
        - $c = mc + (1-m)\frac{1}{B} \sum_{x=1}^B g_{{\theta}_t}(x_i)$
	- **Sharpening**: divide by the standard deviation of the teacher output
        - Done by using a low temperature ${\tau}_t$ for the teacher softmax
	- Centering prevents one dimension to dominate but encourages collapse to uniform distribution, sharpening does the opposite

## Results

- Self-supervised ViT features perform particularly well with a basic k-NN without any finetuning, linear classifier, nor data augmentation
- **One advantage of self-supervised learning: does not hyperoptimize from the specific images in the dataset for the specific task**
    - In the image below, you can see DINO focuses on the main object where supervised learning learns some random parts of the background ![[dino_attention.png]]
