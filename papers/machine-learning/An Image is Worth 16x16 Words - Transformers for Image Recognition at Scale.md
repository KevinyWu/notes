# An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

**Authors**: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby

#transformers
#visual-representations

[[papers/machine-learning/README#[2020-10] An Image is Worth 16x16 Words Transformers for Image Recognition at Scale|README]]

[Paper](http://arxiv.org/abs/2010.11929)
[Code](https://github.com/google-research/vision_transformer)
[Video](https://www.youtube.com/watch?v=TrdevFK_am4)
[Annotated Code](https://nn.labml.ai/transformers/vit/index.html)

## Abstract

> While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.

## Summary

- Transformer architecture has been successful in NLP
- Apply standard transformer to image patches, providing sequence of linear embeddings of these patches as an input to a transformer
- Worse than ResNet on ImageNet, but better on larger datasets
- Completely discards the notion of convolutions

## Background

- **Transformer is a generalization of an MLP**
- The "weight" between nodes in consecutive layers is fixed in MLP, but computed on the fly in a transformer
- This makes a transformer the most general, least generalized thing we can train in ML!
- Transformer is less biased than other architectures
- This is why we concatenate the input and positional encoding in a transformer

## Method

- Follow original transformer as closely as possible
- Standard transformer receives a 1D sequence of token embeddings
- For 2d images, reshape the image $x \in \mathbb{R}^{H \times W \times C}$ into a sequence of flattened 2D patches $x \in \mathbb{R}^{N \times (P^2 \cdot C)}$
    - $N = H \times W / P^2$ is the number of patches; the input sequence length
    - $P$ is the patch size
		- Use patches instead of the whole image for computational efficiency
		- Naive self attention would require each pixel to attend to every other pixel
    - $C$ is the number of channels
    - Flatten the patches and map to $D$ dimensions with linear layer; output is called the "patch embedding"
    - Position embeddings added to patch embeddings to retain positional information
    - **Patch + position embeddings are input to the transformer encoder**, output feeds into MLP for classification
- ViT architecture ![[vit.png]]
- **ViT has less image specific inductive bias than CNNs**
	- Inductive bias refers to the set of assumptions a model makes about the data
    - CNNs assume **locality** (pixels close to each other are related) and **translation invariance** (features are the same regardless of where they are in the image)
    - ViTs use self attention to consider relationships between all parts of the image simultaneously
		- Positional embeddings at initialization don't carry information about 2D position of the patches
		- For classification, only the MLP head in ViT assumes locality and translational invariance

## Results

- ViT outperforms CNNs with the same computational budget
- Internal representations ![[vit_internal.png]]
    - First layer linearly projects flattened patches to lower dimension
    - Learned positional embeddings are added to the patch embeddings, closer patches tend to have more similar positional embeddings, as well as row-column structure
