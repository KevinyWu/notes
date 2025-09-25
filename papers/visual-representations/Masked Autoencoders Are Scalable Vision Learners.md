# Masked Autoencoders Are Scalable Vision Learners

**Authors**: Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick

[[papers/visual-representations/README#[2021-11] Masked Autoencoders Are Scalable Vision Learners|README]]

[Paper](http://arxiv.org/abs/2111.06377)
[Code](https://github.com/facebookresearch/mae)
[Video](https://www.youtube.com/watch?v=Dp6iICL2dVI)

## Abstract

> This paper shows that masked autoencoders (MAE) are scalable self-supervised learners for computer vision. Our MAE approach is simple: we mask random patches of the input image and reconstruct the missing pixels. It is based on two core designs. First, we develop an asymmetric encoder-decoder architecture, with an encoder that operates only on the visible subset of patches (without mask tokens), along with a lightweight decoder that reconstructs the original image from the latent representation and mask tokens. Second, we find that masking a high proportion of the input image, e.g., 75%, yields a nontrivial and meaningful self-supervisory task. Coupling these two designs enables us to train large models efficiently and effectively: we accelerate training (by 3x or more) and improve accuracy. Our scalable approach allows for learning high-capacity models that generalize well: e.g., a vanilla ViT-Huge model achieves the best accuracy (87.8%) among methods that use only ImageNet-1K data. Transfer performance in downstream tasks outperforms supervised pre-training and shows promising scaling behavior.

## Summary

- **MAE is asymmetric**
    - Encoder operates only on visible patches
    - Decoder is lightweight and reconstructs input from latent representation and masked patches
    - **Very high masking ratio**: optimizes accuracy and reduces training time
- Outperforms all previous results on ImageNet-1k

## Background

- Masked autoencoding in BERT: remove a portion of the input and predict it
- Autoencoding in vision lags behind NLP
    - Until recently, vision relied on CNN, so hard to integrate mask tokens or positional embeddings
    - Information density is different, i.e. masking 15% of words is nontrivial, but masking 15% of pixels is easier to reconstruct
    - Decoding reconstructs pixels (low semantic information), not words (high semantic information), so less trivial than BERT, which can use an MLP decoder

## Method

- **Encoder** maps observed signal to a latent representation
- **Decoder** reconstructs the original signal from latent representation
- MAE method ![[mae.png]]
- Divide image into non-overlapping patches, removing high percentage of random patches
- Encoder
	- ViT, only applied to visible patches
		- Adds positional embeddings to the patch embeddings
		- Can train very large encoders with less compute
- Decoder
	- Series of transformer blocks
	- Input is full set of tokens consisting of encoded visible patches and mask tokens
	- Mask token is a shared, learned vector that indicates presence of a missing patch to be predicted
	- MAE decoder only used during pre-training for the image reconstruction task, so architecture can be designed independently of the encoder
- Reconstruction target
    - Each element of decoder output is a vector of pixel values representing a patch
    - **Loss is MSE between reconstructed and original images in pixel space**

## Results

- Masking ratio: around 75% is best for both linear probing and fine-tuning
- Decoder depth
    - For linear probing, sufficiently deep decoder necessary to account for the specialization of the many encoder layers
    - For fine-tuning, single block decoder can perform strongly
- Mask token: encoder performs worse with mask tokens
- **Does not rely heavily on data augmentation like MoCo, SimCLR, BYOL**
	- Contrastive learning must rely on augmentation
- Transfer learning
    - MAE outperforms supervised pre-training on COCO object detection and segmentation
    - Does well on semantic segmentation and classification
