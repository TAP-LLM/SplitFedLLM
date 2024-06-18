# SplitFedLLM
![transformers](https://img.shields.io/badge/transformers->=4.38.0-blue)
![License](https://img.shields.io/badge/license-MIT-yellow)  

 Read this in [English](README_en.md)

## 项目介绍
本开源项目基于开源的LLAMA和GLM模型，实现了单机/多机部署的分布式联邦学习框架对模型进行微调以及推理,  
在保证客户端数据隐私安全的同时，实现模型参数的聚合，从而实现模型参数的共享。使得用户可以在自身算力  
有限的情况下利用项目部署平台的资源端进行模型微调，从而实现模型的垂直领域定制化。

## 支持的模型
| Model            | Type | Seq Length | Download                                                                                                                                | Online Demo                                                                                                                                                                                |
|------------------|------|------------|-----------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GLM-4-9B         | Base | 8K         | [🤗 Huggingface](https://huggingface.co/THUDM/glm-4-9b)  [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/glm-4-9b)   [🟣 WiseModel](https://wisemodel.cn/models/ZhipuAI/glm-4-9b)    | /                                                                                                                                                                                          |
| GLM-4-9B-Chat    | Chat | 128K       | [🤗 Huggingface](https://huggingface.co/THUDM/glm-4-9b-chat)  [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/glm-4-9b-chat)   [🟣 WiseModel](https://wisemodel.cn/models/ZhipuAI/GLM-4-9B-Chat)      | [🤖 ModelScope CPU](https://modelscope.cn/studios/dash-infer/GLM-4-Chat-DashInfer-Demo/summary)<br> [🤖 ModelScope vLLM](https://modelscope.cn/studios/ZhipuAI/glm-4-9b-chat-vllm/summary) |
| GLM-4-9B-Chat-1M | Chat | 1M         | [🤗 Huggingface](https://huggingface.co/THUDM/glm-4-9b-chat-1m)  [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/glm-4-9b-chat-1m)  [🟣 WiseModel](https://wisemodel.cn/models/ZhipuAI/GLM-4-9B-Chat-1M)  | /                                                                                                                                                                                          |
| GLM-4V-9B        | Chat | 8K         | [🤗 Huggingface](https://huggingface.co/THUDM/glm-4v-9b)  [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/glm-4v-9b)   [🟣 WiseModel](https://wisemodel.cn/models/ZhipuAI/GLM-4V-9B  )    | [🤖 ModelScope](https://modelscope.cn/studios/ZhipuAI/glm-4v-9b-Demo/summary)                                                                                                              |
