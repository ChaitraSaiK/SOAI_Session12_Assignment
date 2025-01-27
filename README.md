# GPT Shakespeare - A Transformer-Based Text Generator

[![Model](https://img.shields.io/badge/Model-GPT--2%20Medium-blue)](https://huggingface.co/gpt2-medium)
[![Training](https://img.shields.io/badge/Training-Shakespeare%20Style-orange)](https://github.com/ChaitraSaiK/shakespeare-gpt)
[![Loss](https://img.shields.io/badge/Target%20Loss-0.099999-green)](https://github.com/ChaitraSaiK/shakespeare-gpt)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Spaces-Talk--To--Me--AI-yellow)](https://huggingface.co/spaces/ChaitraSaiK/Talk-To-Me-AI)
[![Made with PyTorch](https://img.shields.io/badge/Made%20with-PyTorch-red?logo=pytorch)](https://pytorch.org/get-started/locally/)

This repository contains an implementation of a GPT-based language model designed to mimic Shakespearean-style writing. The code leverages PyTorch for building and training the transformer architecture, providing options for customization and model checkpointing.


## Features

- Causal Self-Attention: Implements the core mechanism of GPT with Flash Attention for optimized performance.

- Modular Components: Easy-to-understand classes for GPT layers, blocks, and the model itself.

- Device Compatibility: Automatically detects CUDA or MPS support for accelerated computation.

- Logging and Checkpoints: Logs training progress and supports saving intermediate checkpoints.


## Training Details

The model was trained with:

- Mixed precision training

- Gradient accumulation

- Cosine learning rate schedule

- Early stopping

- Gradient clipping

- Target loss: 0.099999


## Model Deployment

The GPT-Shakespeare model is live and accessible on Hugging Face Spaces. 

Accessing the Model : Visit the following link to interact with the model: https://huggingface.co/spaces/ChaitraSaiK/Talk-To-Me-AI

![image](https://github.com/user-attachments/assets/78855059-47d7-4c5d-8525-48a1df3bfc9d)




