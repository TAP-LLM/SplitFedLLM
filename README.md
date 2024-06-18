# SplitFedLLM
![transformers](https://img.shields.io/badge/transformers->=4.38.0-blue)
![License](https://img.shields.io/badge/license-MIT-yellow)  

 Read this in [English](README_en.md)

## 项目介绍
本开源项目基于开源的LLAMA和GLM模型，实现了单机/多机部署的分布式联邦学习框架对模型进行微调以及推理,  
在保证客户端数据隐私安全的同时，实现模型参数的聚合，从而实现模型参数的共享。使得用户可以在自身算力  
有限的情况下利用项目部署平台的资源端进行模型微调，从而实现模型的垂直领域定制化。

## 支持的模型
| Model            | Type | Download                                                                                                                                |
|------------------|------|-----------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GLM-4-9B         | Base | [🤗 Huggingface](https://huggingface.co/THUDM/glm-4-9b)   | /                                                                                                                                                                                          |
| GLM-4-9B-Chat    | Chat | [🤗 Huggingface](https://huggingface.co/THUDM/glm-4-9b-chat)  |
| GLM-4-9B-Chat-1M | Chat | [🤗 Huggingface](https://huggingface.co/THUDM/glm-4-9b-chat-1m)                                                                                                                                                                                          |
| GLM-4V-9B        | Chat | [🤗 Huggingface](https://huggingface.co/THUDM/glm-4v-9b)                                                                                                             |
