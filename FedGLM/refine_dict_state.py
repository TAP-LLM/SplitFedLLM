import argparse
import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('./chatglm-6b')

from modeling_chatglm import *
from client_model_partA import ChatGLMForConditionalGenerationClientSide
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)


config = AutoConfig.from_pretrained('./fed-glm-module/client/', trust_remote_code=True)
config.pre_seq_len = None
config.prefix_projection = False

# model = AutoModel.from_pretrained('./fed-glm-module/client/', config=config,  trust_remote_code=True)
model = ChatGLMForConditionalGenerationClientSide(config=config)
ss = model.state_dict()
print(model.state_dict())
# print(model) 通过
# 载入模型参数
state_dict1=torch.load('./save_model/clientsideglm.bin')
# print(state_dict) 参数名不同


# 替换state dict
a = state_dict1['0.weight']

print(type(a))
ss['transformer.word_embeddings.weight'] = a

for k,v in ss.items():
    if k.startswith('transformer.layers.0.'):
        string = '1.'+ k[len('transformer.layers.0.'):]
        print(string)
        ss[k] = state_dict1[string]
  
# ss['transformer.client_layers.0']=state_dict1['1']
print(ss) 
# 替换保存参数文件
torch.save(ss,'./client/client_model_partA_param.bin')
