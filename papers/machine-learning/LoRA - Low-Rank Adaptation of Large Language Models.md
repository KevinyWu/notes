# LoRA - Low-Rank Adaptation of Large Language Models

**Authors**: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen

[[papers/machine-learning/README#[2021-06] LoRA Low-Rank Adaptation of Large Language Models|README]]

[Paper](http://arxiv.org/abs/2106.09685)
[Code](https://github.com/microsoft/LoRA)
[Video](https://youtu.be/DhRoTONcyZE?feature=shared)
[Blog](https://www.ibm.com/think/topics/lora)

## Abstract

> An important paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to particular tasks or domains. As we pre-train larger models, full fine-tuning, which retrains all model parameters, becomes less feasible. Using GPT-3 175B as an example – deploying independent instances of fine-tuned models, each with 175B parameters, is prohibitively expensive. We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times. LoRA performs on-par or better than fine-tuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3, despite having fewer trainable parameters, a higher training throughput, and, unlike adapters, no additional inference latency. We also provide an empirical investigation into rank-deficiency in language model adaptation, which sheds light on the efficacy of LoRA. We release a package that facilitates the integration of LoRA with PyTorch models and provide our implementations and model checkpoints for RoBERTa, DeBERTa, and GPT-2 at <https://github.calecom/microsoft/LoRA>.

## Summary

- Learned over-parameterized models reside on a low intrinsic dimension
	- LoRA allows us to train some dense layers in a neural network indirectly by optimizing rank decomposition matrices of the dense layers' change
- Can switch between small LoRA modules for different tasks
- In full fine-tuning, the model is initialized to the pretrained weights and all weights are updated by following the gradient to maximize the objective
	- In LoRA, we only need to optimize over a much smaller set of parameters

## Background

- Existing solutions for transfer learning
	- Adding adapter layers introduce inference latency that there is no way to bypass
	- Prefix tuning is difficult to optimize and its performance changes non-monotonically in trainable parameters

## Method

- Low rank decomposition $W_{0}+ \Delta W = W_{0}+ BA$
	- Weight matrix $W_{0}\in \mathbb{R}^{d\times k}$
	- $B\in \mathbb{R}^{d\times r}, A\in \mathbb{R}^{r\times k}$ with rank $r \ll \min(d, k)$
	- $A, B$ contain trainable parameters, $W_0$ frozen
- Forward pass: $h = W_{0}x + \Delta Wx = W_{0}x + BAx$
- Gaussian initialization for $A$ and zero for $B$ so that $\Delta W = BA = 0$ at the beginning of training
  ![[lora.png|300]]
- When we increase the rank $r$, LoRA converges to training the original model

## Results

- Limit study to adapting the attention weights of transformers
- LoRA performs better than other finetuning methods with less trainable parameters, often even better than full fine-tuning
- In a transformer, adapting $W_{q}, W_{v}$ (query, value) are the most effective compared to $W_{k}, W_{o}$ (key, output)
- Surprisingly, LoRa is competitive with very small $r$
	- Even $r = 1, 2$ is good, suggesting the weight matrix $\Delta W$ has a small intrinsic rank
