# FedGLM: Safely Learning with Private Data
## Project Introduction
This open source project is based on the open source LLAMA and GLM models, and has implemented a distributed federated learning framework for model fine-tuning and inference on a single/multiple machine deployment.
While ensuring client data privacy and security, it aggregates model parameters to achieve model parameter sharing. This allows users with limited computing power to use the resources of the project deployment platform for model fine-tuning, thereby achieving vertical domain customization of the model.

## Supported Models
| Model            | Type | Download                                                                                                                                |
|------------------|------|-----------------------------------------------------------------------------------------------------------------------------------------|
| ChatGLM-6B | Chat |https://github.com/THUDM/ChatGLM-6B|
| Llama-2-7b-hf    | Chat | [🤗 Huggingface](https://huggingface.co/meta-llama/Llama-2-7b-hf)  |
| Llama-2-7b-chat-hf | Chat | [🤗 Huggingface](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)                                                                                                                                                                                          |

## Notices
Note that the FedGLM folder contains the training scheme based on the flower framework and ChatGLM-6B, and FedGLM-LLaMA contains the training scheme based on the flask communication framework and LLaMA-7B, and the details of the operation can be referred to the readme file in each folder.

## Reference
If you find this repository useful or our work is related to your research, please kindly cite it:
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