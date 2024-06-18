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
|------------------|------|-----------------------------------------------------------------------------------------------------------------------------------------|                                                                                                                                                                                         
| Llama-2-7b-hf    | Chat | [🤗 Huggingface](https://huggingface.co/meta-llama/Llama-2-7b-hf)  |
| Llama-2-7b-chat-hf | Chat | [🤗 Huggingface](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)                                                                                                                                                                                          |

## 快速开始

### 安装依赖项
```bash
git clone https://github.com/TAP-LLM/SplitFedLLM.git
cd SplitFedLLM
pip install -e .
```
### 数据准备

请将训练数据集按照示例文件格式进行整理（一个单列的csv文件，输入整体是一句话，其中的特殊token的使用也符合文件示例：/data/train_test.csv）

### 设置脚本参数

请尽可能保证客户端和服务器端的参数设置一致（后续可以进行通信传递），以下的参数务必保持一致：  
```bash
    --learning_rate  
    --num_train_epochs
    --warmup_steps 
    --load_in_bits
    --lora_r 
    --lora_alpha
    --target_modules
    --seed
    --block_size
    --deepspeed ds_config_zero2.json # deepspeed配置文件复用
    --bf16  
```


客户端的--master_port='29501'如果单机测试不要改回29500  


### 通信


#### 启动服务
##### 微调：
客户端：./finetune/sft-client  
服务器：./finetune/sft-service
客户端，服务器端都是以启动脚本的方式启动  
以客户端为例：
``` bash
    cd ./finetune/sft-client
    chmod +x ./finetune/sft-client/finetune.sh
    sh ./finetune/sft-client/finetune.sh/finetune.sh
```
注意请先启动服务器端，最好等待服务器反馈"等待接收"后再启动客户端


##### 推理：
客户端：./eval/test-client  
服务器：./eval/test-service  
将服务器端和客户端的模型参数路径改写好：  
```python
finetune_model_path = "your_finetune_model_path"  # lora参数
base_model = "your_base_model_path"  # 原模型参数
```
```bash
python test_lora_model.py  # 也是先运行服务器端，在提示：“服务器端已准备就绪”后启动客户端
```
#### 结束服务
结束后，手动关闭终端。

## TODO
加入断点续训，目前暂不支持