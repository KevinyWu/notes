# Yann Le Cun - SSL, JEPA, World Models and the Future of AI

[[talks/README#[2025-09-10] Yann Le Cun - SSL, JEPA, World Models and the Future of AI|README]]

## Joint Embedding Predictive Architecture (JEPA)

- LLMs trained to predict the next input
	- Only works for discrete domains
	- Auto-Regressive: predicts a variable's future values based on its own past values
	- **Diverges exponentially**
		- Exponential on the probability of a correct prediction
- **Generative architectures don't work for images and video**
	- The world is only partially predictable and generative models must predict every detail of the world
- Solution: Joint Embedding Predictive Architecture (JEPA) ![[jepa.png]]
	- Predict (in representation space) masked parts of inputs
	- Let autoencoder abstract away aspects of an image that are unpredictable
- Moravec's Paradox
	- Things easy for humans are hard for machines and vice versa

## Energy-Based Models

- The only way to formalize all model types is **Energy-Based Models**
	- Train energy function that
		- Gives low energy to matching pairs
		- Gives high energy to incompatible pairs
	- Training energy functions with self-supervised learning (SSL)
		- Contrastive methods
			- However, in high dim space, need exponentially increasing number of pairs
		- Regularized methods
			- Minimizes amount of space that can have low energy
		- **Distillation methods**
			- BYOL, SimSiam, DINOv2, V-JEPA, I-JEPA, V-JEPA 2
			- Fast and no negative samples needed
			- We don't know why it works
		- SSL now overtakes SL!
			- [Scaling Language-Free Visual Representation Learning](https://davidfan.io/webssl/)
			- [DINOv2](https://ai.meta.com/dinov3/)
- In training, make sure no collapse (make sure autoencoder doesn't just learn the identity function, which can happen if you jointly train an encoder and decoder)

## World Models

- World model: given state of the world at time $t$ and an action, can you predict the state of the world at time $t+1$
	- This is identical to Model Predictive Control, but with a world model trained with observations instead of equations written by hand
	- Action inference by minimization of the objectives
- Need hierarchical planning
- Pretrained world models for downstream tasks
	- [DINO-WM](https://dino-wm.github.io/)
	- [Navigation World Models](https://www.amirbar.net/nwm/)
	- [V-JEPA 2](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/)

## Recommendations

- Abandon generative models for JEPA
- Abandon probabilistic models for EBMs
	- Every probabilistic model can be written as an EBM with $E[x] = -\log p(x)$
- Abandon contrastive methods for regularized methods
- Abandon RL for MPC
	- Use RL only when planning doesn't yield the predicted outcome, to adjust the world model or the critic
- Don't work on LLMs
