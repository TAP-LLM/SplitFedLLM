import torch
import os
import sys
import logging

# sys.path.append('./client')
# from client.modeling_llama_A import LlamaForCausalLM
# from server.modeling_llama import LlamaForCausalLM

#自由分割模型
sys.path.append('/home/zhangzishuai/SplitFederated-LLaMA/Models/central_model/Inference')
from Inference.modeling_llama import LlamaForCausalLM


# sys.path.append('./client_part3')
# from client_part3.modeling_llama_C import LlamaForCausalLMC

log_path = "/home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module/splitmodel.log"
logging.basicConfig(filename=log_path,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                    level=logging.INFO,
                    filemode='w')

# create a logger object
logger = logging.getLogger(__name__)
# 加载预训练的Llama模型
model = LlamaForCausalLM.from_pretrained('/home/zhangzishuai/SplitFederated-LLaMA/Models/central_model/model_path')
# model = LlamaForCausalLM.from_pretrained('/home/zhangzishuai/SplitFederated-LLaMA/Models/Llama2-7B-chat-client')
#model = LlamaForCausalLMC.from_pretrained('/home/zhangzishuai/SplitFederated-LLaMA/Models/Llama2-7B-chat-client')
for name, module in model.named_modules():
    logger.info(f"Module: {name}, Type: {type(module)}")

for name, param in model.named_parameters():
    logger.info(f"Parameter Name: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")   

# # 拆分模型为modelA, modelB, modelC
# modelA = torch.nn.Sequential(model.model.embed_tokens, model.model.layers[:1])  # 第0层
# logger.info(f"下面是modelA的参数")
# for name, module in modelA.named_modules():
#     logger.info(f"Module: {name}, Type: {type(module)}")

# for name, param in modelA.named_parameters():
#     logger.info(f"Parameter Name: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")   
# modelB = model.model.layers[1:31]  # 第1到30层
# modelC = torch.nn.Sequential(model.model.layers[31:], model.model.norm, model.lm_head)  # 第31层

# 保存模型到不同的文件
#torch.save(model.state_dict(), '/home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module/client_part3/pytorch_model_C.bin')
# torch.save(modelB.state_dict(), '/home/zhangzishuai/SplitFederated-LLaMA/Models/SplitModel/server/pytorch_model_B.bin')
# torch.save(modelC.state_dict(), '/home/zhangzishuai/SplitFederated-LLaMA/Models/SplitModel/client/pytorch_model_C.bin')

# 如果需要将模型重新加载到对应部分，可以使用以下代码：
# modelA_loaded = torch.nn.Sequential(LlamaForCausalLM(config=LlamaConfig()).model.embed_tokens, model.model.layers[:1])
# modelA_loaded.load_state_dict(torch.load('pytorch_model_A.bin'))

# modelB_loaded = model.model.layers[1:27]
# modelB_loaded.load_state_dict(torch.load('pytorch_model_B.bin'))

# modelC_loaded = torch.nn.Sequential(model.model.layers[27:], model.model.norm, model.lm_head)
# modelC_loaded.load_state_dict(torch.load('pytorch_model_C.bin'))
