# Transformers

## [2020-10] An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

#transformers
#visual-representations
[[An Image is Worth 16x16 Words - Transformers for Image Recognition at Scale]]
- ViT divides an image into fixed-size patches, flattens them, and embeds each patch into a linear space, treating these patches as tokens for the transformer, replacing traditional convolutions
- To preserve spatial information in the image, ViT adds learnable positional embeddings to the patch embeddings, allowing the transformer to capture the relative positions of patches without relying on CNN-style local features
- ViT minimizes image-specific assumptions (e.g., locality and translation invariance in CNNs), relying on global self-attention to learn relationships across the entire image
