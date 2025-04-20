# FL-LLaMA
![transformers](https://img.shields.io/badge/transformers->=4.38.0-greene)
![License](https://img.shields.io/badge/license-MIT-yellow)  
![Python](https://img.shields.io/badge/Python->=3.10.4-blue)  
 Read this in [English](README_en.md)

## 支持的模型
| Model            | Type | Download                                                                                                                                |
|------------------|------|-----------------------------------------------------------------------------------------------------------------------------------------|                                                                                                                                                                                         
| Llama-2-7b-hf    | Base | [🤗 Huggingface](https://huggingface.co/meta-llama/Llama-2-7b-hf)  |
| Llama-2-7b-chat-hf | Chat | [🤗 Huggingface](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)                                                                                                                                                                                          |

## 快速开始

### 安装依赖项
```bash
git clone https://github.com/TAP-LLM/SplitFedLLM.git
cd SplitFedLLM/FL-LLaMa
pip install -r requirements.txt
```
### 数据准备
1. SuperGLUE Benchmark
任务类型：自然语言理解（NLU），包含 8 个子任务，覆盖常识推理、语义解析等复杂语言挑战。
数据集详情：
ReCoRD：通过消解段落中的歧义实体评估阅读理解能力。
COPA：通过选择前提的更可能原因或结果评估因果推理能力。
WSC：通过代词消解评估深度上下文理解能力。
RTE：通过判断句子间的蕴含、矛盾或无关关系评估文本蕴含能力。
BoolQ：基于上下文回答是 / 否问题。
WiC：通过判断单词在两个句子中的含义是否相同评估词义消歧能力。
CB：处理复杂句子中的蕴含关系。
MultiRC：基于多句子上下文回答多答案问题。
官方链接：https://super.gluebenchmark.com/tasks
下载方式
wget https://dl.fbaipublicfiles.com/glue/superglue/data/v2/combined.zip

2. CoQA 数据集
任务类型：对话式问答（QA），聚焦多轮对话中的上下文连贯性评估。
数据集详情：
包含 8,000 多个对话和 127,000 多个问题，覆盖 7 个领域。
近半数问题需要指代消解和语用推理。
官方链接：https://stanfordnlp.github.io/coqa/
下载方式：
wget https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json

3. XSum 数据集
任务类型：摘要生成，评估模型对新闻文章的极端压缩能力。
数据集详情：
包含 BBC 新闻文章和人工撰写的单句摘要,要求模型生成高度简洁的单句摘要。
官方链接：https://github.com/EdinburghNLP/XSum/tree/master
下载方式：
#### 通过 Hugging Face 下载
wget https://huggingface.co/datasets/EdinburghNLP/xsum/resolve/main/data/XSUM-EMNLP18-Summary-Data-Original.tar.gz?download=true
#### Xsum数据拆分
详见 FedGLM-LLaMA/data/Xsum

### 模型准备
1.下载Llama 2模型，保存至 FL-LLaMa/Centralized_Models/model_path/
2.分割模型 python SplitModel.py --SplitA --SplitB --SplitC

### 启动服务
设置脚本参数
工作目录：Fed-Llama-module
请保证客户端和服务器端的参数设置一致，以下的参数务必保持一致：

```bash
    --batch_size
    --learning_rate
    --max_output_length  
    --lora_r 
    --lora_alpha
    --target_modules
    --seed
    --max_source_length
    --max_target_length
    --modelA_layers
    --modelC_layers
    --server_ip
```

将服务器端和客户端的模型参数路径改写好：  
```python
modelA_name_or_path = "your_modelA_name_or_path"
modelB_name_or_path = "your_modelB_name_or_path"  
modelC_name_or_path = "your_modelC_name_or_path"
```

#### 单客户端微调：
`./scripts/train.sh`  

以客户端为例：
``` bash
    cd ./Fed-Llama-module
    chmod +x ./scripts/train.sh
    bash ./scripts/train.sh
```
注意:
1. 在启动服务前，请先将所有的`'your_ip'`字段修改为您服务端与接收端的ip地址。
2. 如果不通过train.sh,手动启动服务器端和客户端，最好等待服务器反馈"Starting Flower server, config: ServerConfig(num_rounds=None, round_timeout=None)"后再启动客户端。


#### 多客户端微调：
`./scripts/train.sh`  

以客户端为例：
``` bash
    cd ./Fed-Llama-module
    chmod +x ./scripts/multi-train.sh
    bash ./scripts/multi-train.sh
```
注意:
1. 在启动服务前，请先将所有的`'your_ip'`字段修改为您服务端与接收端的ip地址。
2. 如果不通过train.sh,手动启动服务器端和客户端，最好等待服务器反馈"Starting Flower server, config: ServerConfig(num_rounds=None, round_timeout=None)"后再启动客户端。

#### 推理：

```bash
    cd ./Fed-Llama-module
    chmod +x ./scripts/inference.sh
    bash ./scripts/inference.sh
```
注意:
1. 在启动服务前，请先将所有的`'your_ip'`字段修改为您服务端与接收端的ip地址。
2. 如果不通过train.sh,手动启动服务器端和客户端，最好等待服务器反馈"Starting Flower server, config: ServerConfig(num_rounds=None, round_timeout=None)"后再启动客户端。
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