# Visual Representations

## [2019-11] Momentum Contrast for Unsupervised Visual Representation Learning

[[Momentum Contrast for Unsupervised Visual Representation Learning]]
- Momentum-based moving average of the query encoder to maintain a consistent and slowly evolving key encoder, helping to stabilize contrastive learning
- MoCo applies the InfoNCE loss to minimize the distance between positive pairs (augmented views of the same image) while maximizing the distance from negative pairs (other images in the queue)
- MoCo has largely closed the gap between unsupervised and supervised representation learning in multiple vision tasks

## [2020-02] A Simple Framework for Contrastive Learning of Visual Representations

[[A Simple Framework for Contrastive Learning of Visual Representations]]
- SimCLR employs a contrastive learning approach by maximizing agreement between different augmented views of the same image instance
- Combining multiple data augmentation techniques (e.g., cropping, resizing, color distortion) to create effective positive pairs is critical for representation learning
- SimCLR introduces a projection head (a simple MLP) to map the representations to a space where the contrastive loss is applied, but this projection head is discarded after pretraining

## [2020-06] Big Self-Supervised Models are Strong Semi-Supervised Learners

[[Big Self-Supervised Models are Strong Semi-Supervised Learners]]
- Three stages: unsupervised pretraining of a large ResNet using SimCLRv2, supervised fine-tuning on a small labeled dataset, and knowledge distillation to a smaller student model using unlabeled data
- Larger models significantly improve semi-supervised learning, achieving 73.9% top-1 accuracy on ImageNet with just 1% of the labels, representing a 10x improvement in label efficiency
- The approach enhances training by retaining the projection head for fine-tuning and using soft labeling during distillation, allowing the student model to benefit from richer information

## [2021-04] Emerging Properties in Self-Supervised Vision Transformers

[[Emerging Properties in Self-Supervised Vision Transformers]]
- Self-supervised ViT features explicitly capture scene layout and object boundaries, leveraging self-attention modules in the final layer, which distinguishes them from features learned by supervised ViTs or convnets
- DINO is a self-distillation method with no labels that predicts the outputs of a dynamically constructed teacher network, utilizing momentum encoding to enhance stability and consistency in learning
- To avoid collapse in representation learning, DINO employs centering and sharpening techniques on teacher outputs, promoting diversity and preventing uniform distribution while maintaining effective local-to-global correspondence in image crops

## [2021-11] Masked Autoencoders Are Scalable Vision Learners

[[Masked Autoencoders Are Scalable Vision Learners]]
- Masked autoencoders (MAE) utilize an asymmetric encoder-decoder architecture, where the encoder processes only visible patches of an image while the lightweight decoder reconstructs the full image from latent representations and masked patches
- Masks a high proportion of input images (up to 75%), allowing for meaningful self-supervised learning that significantly improves generalization
- Does not heavily depend on data augmentation techniques

## [2022-09] VIP: Towards Universal Visual Reward and Representation via Value-Implicit Pre-Training

[[VIP - Towards Universal Visual Reward and Representation via Value-Implicit Pre-Training]]
- VIP leverages large-scale, offline human videos for pre-training a visual representation that generates dense rewards for unseen robotic tasks, without requiring task-specific robotic data
- Formulates reward learning as a goal-conditioned RL problem using a dual value-function approach, avoiding the need for action labels and enabling effective learning from action-free human videos
