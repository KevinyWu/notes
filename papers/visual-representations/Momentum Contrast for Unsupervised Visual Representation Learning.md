# Momentum Contrast for Unsupervised Visual Representation Learning

**Authors**: Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick

#visual-representations
#unsupervised

[[papers/visual-representations/README#[2019-11] Momentum Contrast for Unsupervised Visual Representation Learning|README]]

[Paper](http://arxiv.org/abs/1911.05722)
[Code](https://github.com/facebookresearch/moco)

## Abstract

> We present Momentum Contrast (MoCo) for unsupervised visual representation learning. From a perspective on contrastive learning [29] as dictionary look-up, we build a dynamic dictionary with a queue and a moving-averaged encoder. This enables building a large and consistent dictionary on-the-fly that facilitates contrastive unsupervised learning. MoCo provides competitive results under the common linear protocol on ImageNet classification. More importantly, the representations learned by MoCo transfer well to downstream tasks. MoCo can outperform its supervised pre-training counterpart in 7 detection/segmentation tasks on PASCAL VOC, COCO, and other datasets, sometimes surpassing it by large margins. This suggests that the gap between unsupervised and supervised representation learning has been largely closed in many vision tasks.

## Summary

- Maintain the dictionary as a queue of data samples: the encoded representations of the current mini-batch are enqueued, and the oldest are dequeued
- A slowly progressing key encoder, implemented as a momentum-based moving average of the query encoder, is proposed to maintain consistency
- A query matches a key if they are encoded views (e.g., different crops) of the same image

## Background

- Unsupervised representation learning successful in NLP but not in vision
- Contrastive losses measure the similarities of sample pairs in a representation space
- Pretext task: implies that the task being solved is not of genuine interest, but is solved only for the true purpose of learning a good data representation

## Method

- **Contrastive learning as dictionary look-up**
- Consider an encoded query $q$ and set of encoded samples $\{k_0, k_1, k_2, \ldots\}$ that are the keys of a dictionary
- Assume that there is a single key (denoted as $k^+$) in the dictionary that $q$ matches
- A contrastive loss is a function whose value is low when $q$ is similar to its positive key $k^+$ and dissimilar to all other keys (considered negative keys for $q$)
- InfoNCE loss: $\mathcal{L} = -\log\frac{\exp(q\cdot k^+/\tau)}{\sum_{k\in\mathcal{K}}\exp(q\cdot k/\tau)}$
    - $\tau$ is a temperature parameter
- Momentum contrast ![[moco.png]]
    - Queue of data samples can be much larger than a mini-batch (size is hyperparameter), older samples are dequeued as they are the most outdated keys
    - Momentum update: $\theta_k \leftarrow m\theta_k + (1-m)\theta_q$
		- $\theta_k$ is the key encoder's ($f_k$) parameters
		- $\theta_q$ is the query encoder's ($f_q$) parameters
		- $m$ is the momentum coefficient, empirically, larger (slowly evolving key encoder) is better
- Pretext task: query and key are positive pair if they originate from the same image, negative otherwise
	- Take two random crops of image under random augmentation as positive pair
- Use ResNet as encoder
- Batchnorm prevents good representations: use shuffling BN

## Results

- Downstream tasks: ImageNet-1M and Instagram-1B classification
- Achieves best classification accuracy compared to supervised baselines
- **MoCo has largely closed the gap between unsupervised and supervised representation learning in multiple vision tasks**
