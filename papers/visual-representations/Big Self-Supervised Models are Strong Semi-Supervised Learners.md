# Big Self-Supervised Models are Strong Semi-Supervised Learners

**Authors**: Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, Geoffrey Hinton

[[papers/visual-representations/README#[2020-06] Big Self-Supervised Models are Strong Semi-Supervised Learners|README]]

[Paper](http://arxiv.org/abs/2006.10029)
[Code](https://github.com/google-research/simclr)
[Video](https://www.youtube.com/watch?v=2lkUNDZld-4)

## Abstract

> One paradigm for learning from few labeled examples while making best use of a large amount of unlabeled data is unsupervised pretraining followed by supervised fine-tuning. Although this paradigm uses unlabeled data in a task-agnostic way, in contrast to common approaches to semi-supervised learning for computer vision, we show that it is surprisingly effective for semi-supervised learning on ImageNet. A key ingredient of our approach is the use of big (deep and wide) networks during pretraining and fine-tuning. We find that, the fewer the labels, the more this approach (task-agnostic use of unlabeled data) benefits from a bigger network. After fine-tuning, the big network can be further improved and distilled into a much smaller one with little loss in classification accuracy by using the unlabeled examples for a second time, but in a task-specific way. The proposed semi-supervised learning algorithm can be summarized in three steps: unsupervised pretraining of a big ResNet model using SimCLRv2, supervised fine-tuning on a few labeled examples, and distillation with unlabeled examples for refining and transferring the task-specific knowledge. This procedure achieves 73.9% ImageNet top-1 accuracy with just 1% of the labels ($\le$13 labeled images per class) using ResNet-50, a $10\times$ improvement in label efficiency over the previous state-of-the-art. With 10% of labels, ResNet-50 trained with our method achieves 77.5% top-1 accuracy, outperforming standard supervised training with all of the labels.

## Summary

- Unsupervised pretrain, supervised fine-tune
- For semi-supervised learning, the fewer the labels, the more it benefits from a bigger model

## Background

- Improvement over SimCLR
    - Larger models (deeper but less wide)
    - Increase capacity of non-linear projection head, $g(\cdot)$, recall that it was an MLP with 1 hidden layer in SimCLR
	- Also don't discard the projection head after pretraining, instead fine-tune from a middle layer
- Incorporate memory mechanism of MoCo

## Method

- Stages ![[simclrv2.png]]
    - **Pretrain**: first stage of unlabeled data, task-agnostic pretraining, learn general visual representations
    - **Fine-tune**: then, general representations are fine-tuned on a small labeled dataset
    - **Distill**: second stage of unlabled data, task-specific pretraining, learn task-specific representations
- Fine-tuning
    - Fine-tune from a middle layer of the projection head instead of the input layer of the projection head as in SimCLR
  - Knowledge distillation
    - Use the fine-tuned network as a teacher to assign labels for training a student network
		- **A teacher-assigned soft label for an image is a probability distributions over the classes**
    - Student network an be a smaller version of the teacher network with the same performance
    - Self-distillation: student is the same architecture as the teacher
    - $\mathcal{L}^{\text{distill}} = -\sum_{x_i\in\mathcal{D}}\left [\sum_{y}P^T(y|x_i;\tau)\log P^S(y|x_i;\tau)\right]$
		- $P^T(y|x_i)$ is the teacher's output, fixed during training
		- $P^T(y|x_i; \tau) = \exp(f^{\text{task}}(x_i) [y]/\tau)/\sum_{y'}\exp(f^{\text{task}}(x_i) [y']/\tau)$
		- $P^S(y|x_i)$ is the student's output, learned during training

## Results

- Bigger models, which could easily overfit with few labelled examples, actually generalize better
	- 73.9% top-1 accuracy on ImageNet with just 1% of the labels, representing a 10x improvement in label efficiency
- Findings can be used to improve accuracy in any application of computer vision where it is more expensive or difficult to label additional data than to train larger models
