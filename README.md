# SplitFedLLM
![transformers](https://img.shields.io/badge/transformers->=4.38.0-greene)
![License](https://img.shields.io/badge/license-MIT-yellow)  
![Python](https://img.shields.io/badge/Python->=3.10.4-blue)  
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
pip install -r requirements.txt
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
客户端：`./finetune/sft-client`  
服务器：`./finetune/sft-service`
客户端，服务器端都是以启动脚本的方式启动  
以客户端为例：
``` bash
    cd ./finetune/sft-client
    chmod +x ./finetune.sh
    sh ./finetune.sh
```
注意:
1. 在启动服务前，请先将所有的`'your_ip'`字段修改为您服务端与接收端的ip地址。
2. 请先启动服务器端，最好等待服务器反馈"等待接收"后再启动客户端。
3. 如果使用数据集为服务器接收端都可见，可以修改`finetune_clm_lora.py`参数为`fast_finetune_clm_lora.py`，注意服务器与接收端双方都需要修改。
4. 如果使用数据集为服务器不可见，即使用finetune_clm_lora.py文件的话，将/SplitFedLLM/finetune/sft-service/finetune.sh文件中 的train_files ，validation_files 参数保持默认值不变，仅修改client端的参数





##### 推理：
客户端：`./eval/client.py`  
服务器：`./eval/service.py`  
将服务器端和客户端的模型参数路径改写好：  
```python
finetune_model_path = "your_finetune_model_path"  # lora参数
base_model = "your_base_model_path"  # 原模型参数
```
```bash
python service.py
python client.py  
```
注意:
1. 在启动服务前，请先将所有的`'your_ip'`字段修改为您服务端与接收端的ip地址。
2. 请先启动服务器端，最好等待服务器反馈"等待接收"后再启动客户端。
#### 结束服务
结束后，手动关闭终端。

## TODO
加入断点续训，目前暂不支持

## 引用 
如果你觉得我们的工作有帮助的话，请引用如下论文。
```
@misc{zheng2024safelylearningprivatedata,
      title={Safely Learning with Private Data: A Federated Learning Framework for Large Language Model}, 
      author={JiaYing Zheng and HaiNan Zhang and LingXiang Wang and WangJie Qiu and HongWei Zheng and ZhiMing Zheng},
      year={2024},
      eprint={2406.14898},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2406.14898}, 
}
```