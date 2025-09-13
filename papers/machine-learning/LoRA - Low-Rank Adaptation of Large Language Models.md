# LoRA - Low-Rank Adaptation of Large Language Models

**Authors**: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen

#finetuning
#linear-algebra

[[papers/machine-learning/README#[2021-06] LoRA Low-Rank Adaptation of Large Language Models|README]]

[Paper](http://arxiv.org/abs/2106.09685)
[Code](https://github.com/microsoft/LoRA)
[Video](https://youtu.be/DhRoTONcyZE?feature=shared)
[Blog](https://www.ibm.com/think/topics/lora)

## Abstract

> An important paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to particular tasks or domains. As we pre-train larger models, full fine-tuning, which retrains all model parameters, becomes less feasible. Using GPT-3 175B as an example – deploying independent instances of fine-tuned models, each with 175B parameters, is prohibitively expensive. We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times. LoRA performs on-par or better than fine-tuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3, despite having fewer trainable parameters, a higher training throughput, and, unlike adapters, no additional inference latency. We also provide an empirical investigation into rank-deficiency in language model adaptation, which sheds light on the efficacy of LoRA. We release a package that facilitates the integration of LoRA with PyTorch models and provide our implementations and model checkpoints for RoBERTa, DeBERTa, and GPT-2 at <https://github.calecom/microsoft/LoRA>.

## Summary

- Learned over-parameterized models reside on a low intrinsic dimension
- Can switch between small LoRA modules for different tasks

## Background

- Notes about the background information

## Method

- Notes about the method

## Results

- Notable results from the paper
