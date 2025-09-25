# A Simple Framework for Contrastive Learning of Visual Representations

**Authors**: Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton

[[papers/visual-representations/README#[2020-02] A Simple Framework for Contrastive Learning of Visual Representations|README]]

[Paper](http://arxiv.org/abs/2002.05709)
[Code](https://github.com/google-research/simclr)
[Blog](https://research.google/blog/advancing-self-supervised-and-semi-supervised-learning-with-simclr/)

## Abstract

> This paper presents SimCLR: a simple framework for contrastive learning of visual representations. We simplify recently proposed contrastive self-supervised learning algorithms without requiring specialized architectures or a memory bank. In order to understand what enables the contrastive prediction tasks to learn useful representations, we systematically study the major components of our framework. We show that (1) composition of data augmentations plays a critical role in defining effective predictive tasks, (2) introducing a learnable nonlinear transformation between the representation and the contrastive loss substantially improves the quality of the learned representations, and (3) contrastive learning benefits from larger batch sizes and more training steps compared to supervised learning. By combining these findings, we are able to considerably outperform previous methods for self-supervised and semi-supervised learning on ImageNet. A linear classifier trained on self-supervised representations learned by SimCLR achieves 76.5% top-1 accuracy, which is a 7% relative improvement over previous state-of-the-art, matching the performance of a supervised ResNet-50. When fine-tuned on only 1% of the labels, we achieve 85.8% top-5 accuracy, outperforming AlexNet with 100X fewer labels.

## Summary

- Data augmentation improves unsupervised constrastive learning
- Learnable nonlinear transformation between representation and contrastive loss improves learned representations
- Representation learning with contrastive cross-entropy loss benefits from normalized embeddings and temperature param
- Contrastive learning benefits from larger batch sizes and longer training compared to supervised counterpart

## Background

- Comparison to MoCo
	- Operates without a memory bank, generating all necessary negative samples from the current batch
	- Includes a projection head to map representations for contrastive loss but discards this head after pretraining
		- Finds that the representation before the nonlinear projection is more useful for downstream tasks, contrasting with MoCo's continued use of both query and key encoders
	- Systematically studies and emphasizes diverse data augmentation techniques

## Method

- SimCLR learns representations by maximizing agreement between differently augmented views of the same data instance via a contrastive loss in the latent space
- Framework ![[simclr.png]]
    - **Data augmentation** to generate positive pairs
    - **Base encoder** $f(\cdot)$ to extract representation vectors: ResNet
    - **Projection head** $g(\cdot)$ to map representations to a space where contrastive loss is applied: MLP, 1 hidden layer
		- This is discarded after pretraining
    - **Contrastive loss function**: attempt to identify the positive pair from a set of examples (treat ALL others as negative)
		- $\text{sim}(u, v) = \frac{u^Tv}{\|u\|\|v\|}$ cosine similarity
		- For a positive pair of examples $(i, j)$, NT-Xent (normalized temperature-scaled cross-entropy loss) is used
		- $l_{i,j} = -\log\left(\frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N}\mathbb{1}_{[k\neq i]}\exp(\text{sim}(z_i, z_k)/\tau)}\right)$
			- $\tau$ is the temperature parameter, higher temperature leads to softer probability distribution
			- $z_i = g(f(x_i))$ is the projection of the representation of $x_i$
- Training and Evaluation
    - No memory bank (to store embeddings of past data samples)
    - Batch size $N$ gives $2N-2$ negative examples per positive pair
    - LARS optimizer: Layer-wise Adaptive Rate Scaling
- Data Augmentation for Contrastive Representation Learning
	- Composition of multiple augmentations
	- Crop, resize, rotate, cutout, color distort, sobel, Gaussian blur
	- **No single transformation suffices to learn good representations**
- Architectures for Encoder and Head
	- Unsupervised learning benefits more from bigger models than its supervised counterpart
	- **The hidden layer before the projection head is a better representation than the layer after, i.e. $h = f(x)$ is better than $z = g(h)$**
		- Projection head $g$ is discarded after pretraining
		- Possible reason: the importance of using the representation before the nonlinear projection is due to loss of information induced by the contrastive loss
		- $g$ is trained to be invariant to data transformation, so it can remove information that may be useful for the task like color or orientation of objects

## Results

- Performs better than MoCo and other unsupervised baselines
- With more parameters, SimCLR matches the performance of supervised ResNet-50
