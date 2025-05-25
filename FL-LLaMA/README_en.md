# FL-LLaMA
![transformers](https://img.shields.io/badge/transformers->=4.38.0-green)
![License](https://img.shields.io/badge/license-MIT-yellow)  
![Python](https://img.shields.io/badge/Python->=3.10.4-blue)  
Read this in [中文](README.md)

## Supported Models
| Model            | Type | Download                                                                                                                                |
|------------------|------|-----------------------------------------------------------------------------------------------------------------------------------------|                                                                                                                                                                                         
| Llama-2-7b-hf    | Base | [🤗 Huggingface](https://huggingface.co/meta-llama/Llama-2-7b-hf)  |
| Llama-2-7b-chat-hf | Chat | [🤗 Huggingface](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)                                                                                                                                                                                          |

## Quick Start

### Install Dependencies
```bash
git clone https://github.com/TAP-LLM/SplitFedLLM.git
cd SplitFedLLM/FL-LLaMa
pip install -r requirements.txt
```
### Data Preparation
1. SuperGLUE Benchmark
Task Type: Natural Language Understanding (NLU), encompassing 8 subtasks that cover complex language challenges such as commonsense reasoning and semantic parsing.
Dataset Details:
- ReCoRD: Assesses reading comprehension by resolving ambiguous entities in paragraphs.
- COPA: Evaluates causal reasoning by selecting the more likely cause or effect of a premise.
- WSC: Assesses deep contextual understanding through pronoun resolution.
- RTE: Evaluates textual entailment by judging the entailment, contradiction, or neutrality relationships between sentences.
- BoolQ: Answers yes/no questions based on context.
- WiC: Assesses word sense disambiguation by determining if a word has the same meaning in two sentences.
- CB: Handles entailment relationships in complex sentences.
- MultiRC: Answers multi-answer questions based on multi-sentence contexts.
Official Link: https://super.gluebenchmark.com/tasks
Download Method:
```bash
wget https://dl.fbaipublicfiles.com/glue/superglue/data/v2/combined.zip
```

2. CoQA Dataset
Task Type: Conversational Question Answering (QA), focusing on evaluating contextual coherence in multi-turn conversations.
Dataset Details:
- Contains over 8,000 conversations and more than 127,000 questions, covering 7 domains.
- Nearly half of the questions require coreference resolution and pragmatic reasoning.
Official Link: https://stanfordnlp.github.io/coqa/
Download Method:
```bash
wget https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json
```

3. XSum Dataset
Task Type: Summarization, assessing the model's ability to extremely compress news articles.
Dataset Details:
- Contains BBC news articles and manually written single-sentence summaries, requiring the model to generate highly concise single-sentence summaries.
Official Link: https://github.com/EdinburghNLP/XSum/tree/master
Download Method:
#### Download via Hugging Face
```bash
wget https://huggingface.co/datasets/EdinburghNLP/xsum/resolve/main/data/XSUM-EMNLP18-Summary-Data-Original.tar.gz?download=true
```
#### XSum Data Splitting
Refer to FedGLM-LLaMA/data/Xsum for details.

### Model Preparation
Note: Please cd to the working directory `cd /FL-LLaMA`

1. Download the Llama 2 model and save it to `FL-LLaMa/Centralized_Models/model_path/`

2. Split the model: `python src/SplitModel.py --SplitA --SplitB --SplitC`

3. Split the LoRA parameters: `python src/SplitModel.py --LoraA --LoraB --LoraC --lora_r 16 --lora_alpha 32` (optional, for debugging purposes to ensure consistent training from the same LoRA weights)

### Start the Service
Set Script Parameters

Modify the `cd /home/zhangzishuai/SplitFederated-LLaMA/FL-LLaMA` in the bash script to the path where FL-LLaMA is located.
Change `ServerIP` to your machine's IP address.

Ensure that the parameter settings are consistent between the client and server sides. The following parameters must be kept consistent:

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

#### Single-Client Fine-Tuning:
`./scripts/Auto_train.sh`  

Example for a client:
```bash
    cd ./FL-LLaMA
    chmod +x ./scripts/Auto_train.sh
    bash ./scripts/Auto_train.sh
```
Notes:
1. Before starting the service, please modify all `'your_ip'` fields to your server and receiver's IP addresses.
2. If not using `Auto_train.sh` to manually start the server and client, it is best to wait for the server to反馈 "Starting Flower server, config: ServerConfig(num_rounds=None, round_timeout=None)" before starting the client.
3. The `train_file` only has a value in `flclient_id_0.py` when it is in Train mode; otherwise, it is `None`. Similarly, the `test_file` only has a value in `flclient_id_0.py` when it is in Test mode.
4. During single-client fine-tuning, the `client_count` parameter of `flserver` must be set to 1; otherwise, it will get stuck.

#### Multi-Client Fine-Tuning:
`./scripts/Auto_BatchTrain.sh`  

Example for a client:
```bash
    cd ./FL-LLaMA
    chmod +x ./scripts/Auto_Batchtrain.sh
    bash ./scripts/Auto_Batchtrain.sh
```
Notes:
1. Before starting the service, please modify all `'your_ip'` fields to your server and receiver's IP addresses.
2. If not using `train.sh` to manually start the server and client, it is best to wait for the server to反馈 "Starting Flower server, config: ServerConfig(num_rounds=None, round_timeout=None)" before starting the client.

#### Inference:

```bash
    cd ./FL-LLaMA
    chmod +x ./scripts/Inference.sh
    bash ./scripts/Inference.sh
```
Notes:
1. Before starting the service, please modify all `'your_ip'` fields to your server and receiver's IP addresses.
2. If not using `train.sh` to manually start the server and client, it is best to wait for the server to反馈 "Starting Flower server, config: ServerConfig(num_rounds=None, round_timeout=None)" before starting the client.

#### End the Service
After finishing, manually close the terminal.

Feel free to explore more features.

## TODO
Add support for checkpoint resuming, which is currently not available.

## Citation
If you find our work helpful, please cite the following papers.
```
@article{zhang2025federated,
  title={A Federated Splitting Framework for LLMs: Security, Efficiency, and Adaptability},
  author={Zhang, Zishuai and Zhang, Hainan and Zheng, Jiaying and Wang, Ziwei and Tong, Yongxin and Dong, Jin and Zheng, Zhiming},
  journal={arXiv preprint arXiv:2505.15683},
  year={2025}
}

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
