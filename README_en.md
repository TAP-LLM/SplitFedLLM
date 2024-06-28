## Project Introduction
This open source project is based on the open source LLAMA and GLM models, and has implemented a distributed federated learning framework for model fine-tuning and inference on a single/multiple machine deployment.
While ensuring client data privacy and security, it aggregates model parameters to achieve model parameter sharing. This allows users with limited computing power to use the resources of the project deployment platform for model fine-tuning, thereby achieving vertical domain customization of the model.

## Supported Models
| Model            | Type | Download                                                                                                                                |
|------------------|------|-----------------------------------------------------------------------------------------------------------------------------------------|                                                                                                                                                                                         
| Llama-2-7b-hf    | Chat | [🤗 Huggingface](https://huggingface.co/meta-llama/Llama-2-7b-hf)  |
| Llama-2-7b-chat-hf | Chat | [🤗 Huggingface](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)                                                                                                                                                                                          |

## Quick Start

### Install Dependencies
```bash
git clone https://github.com/TAP-LLM/SplitFedLLM.git
cd SplitFedLLM
pip install -r requirements.txt
```
### Data Preparation

Please arrange the training dataset according to the example file format (a single-column csv file, the input is a sentence as a whole, and the use of special tokens also complies with the file example: /data/train_test.csv)

### Set Script Parameters

Please try to keep the parameter settings of the client and server consistent (communication can be carried out later), the following parameters must be consistent: 
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
    --deepspeed ds_config_zero2.json # reuse of deepspeed configuration file
    --bf16  
```


The client's --master_port='29501' should not be changed back to 29500 if tested on a single machine 


### Communication


#### Start Service
##### Fine-tuning:
Client: `./finetune/sft-client`
Server: `./finetune/sft-service`
Both the client and server are started by launching scripts
Taking the client as an example:
``` bash
    cd ./finetune/sft-client
    chmod +x ./finetune.sh
    sh ./finetune.sh
```
ATTENTION:
1. Before starting the service, please change all the fields of 'your_ip' to the IP addresses of your server and receiver.
2. Please start the server first, and it is best to wait for the server to respond with "waiting for reception" before starting the client.
3. If the dataset is visible to both the server and the receiver, you can change the parameter of finetune_clm_lora.py to fast_finetune_clm_lora.py, note that both the server and the receiver need to make the change.
4. If the dataset is invisible to the server, which means you are using the finetune_clm_lora.py file, then keep the train_files and validation_files parameters in the /SplitFedLLM/finetune/sft-service/finetune.sh file as their default values, and only modify the parameters on the client side.


##### Inference:
Client: `./eval/client.py`
Server: `./eval/service.py` 
Rewrite the model parameter path for the server and client:  
```python
finetune_model_path = "your_finetune_model_path"  # lora parameters
base_model = "your_base_model_path"  # original model parameters
```
```bash
python service.py
python client.py 
```
#### End Service
ATTENTION:
1. Before starting the service, please change all the fields of 'your_ip' to the IP addresses of your server and receiver.
2. Please start the server first, and it is best to wait for the server to respond with "waiting for reception" before starting the client.

## TODO
Join breakpoint continuation, currently not supported

## If you find this repository useful or our work is related to your research, please kindly cite it:
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