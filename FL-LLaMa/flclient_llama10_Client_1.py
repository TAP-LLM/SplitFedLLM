# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#region
import json
from multiprocessing import connection
import sys 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.optimization import AdamW
from torch.optim.lr_scheduler import LinearLR, StepLR
import gc
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('./clientA')
sys.path.append('./clientC')
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from clientA.modeling_llama_A import LlamaForCausalLM
from clientC.modeling_llama_C import LlamaForCausalLMC
import argparse
import jieba
import math
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from datasets import load_dataset
from collections import OrderedDict
import collections
import numpy as np
from typing import Optional, Dict, Tuple
from flwr.common import NDArrays, Scalar
from flwr.client.client import Client
from flwr.client import NumPyClient
from logging import DEBUG, INFO
from flwr.common.logger import log
# import matplotlib.pyplot as plt
# import utils
import flwr as fl
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    BitsAndBytesConfig,
    is_torch_tpu_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from get_dataloader import get_dataset
from get_dataloader_no_pad import get_dataset_no_pad
from peft import (
    AutoPeftModelForCausalLM, 
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
import time
import wandb
import pickle
from multiprocessing import connection
import random
import re
# 设置CUBLAS_WORKSPACE_CONFIG环境变量
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

#For MultiRC测试
from Official_eval.multirc_eval_v1 import official_eval 
from collections import defaultdict

#endregion

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def FLparser(): 
    parser = argparse.ArgumentParser(description="Flower")
    '''=====================================================model args====================================================='''                    
    # 
    parser.add_argument("--modelA_name_or_path", type=str, default="/home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module-autosplit/clientA",
                        help="The model checkpoint for weights initialization. Don't set if you want to train a model from scratch.")
    parser.add_argument("--modelC_name_or_path", type=str, default="/home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module-autosplit/clientC",
                        help="The model checkpoint for weights initialization. Don't set if you want to train a model from scratch.")
    parser.add_argument("--model_type", type=str, default=None,
                        help="If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES))
    parser.add_argument("--config_overrides", type=str, default=None,
                        help="Override some existing default config settings when a model is trained from scratch. Example: n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index")
    parser.add_argument("--modelA_config_name", type=str, default=None,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--modelC_config_name", type=str, default=None,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", type=str, default="/home/zhangzishuai/SplitFederated-LLaMA/Models/Llama2-7B-chat-service",
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Where do you want to store the pretrained models downloaded from huggingface.co")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--target_modules", type=str, default='embed_tokens,q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj',
                        help="List of module names or regex expression of the module names to replace with Lora. For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'")
    parser.add_argument("--use_fast_tokenizer", default=False,
                        help="Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.")
    parser.add_argument("--pretraining_tp", type=int, default=1)
    parser.add_argument("--model_revision", type=str, default="main",
                        help="The specific model version to use (can be a branch name, tag name or commit id).")
    parser.add_argument("--use_auth_token", default=False,
                        help="Will use the token generated when running `huggingface-cli login` (necessary to use this script with private models).")
    parser.add_argument("--torch_dtype", type=str, default="float16",
                        choices=["auto", "bfloat16", "float16", "float32"],
                        help="Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the dtype will be automatically derived from the model's weights.")
    parser.add_argument("--lora_modelA_path", type=str, default=None,
                        help="The pretrained model checkpoint for lora weights.")
    parser.add_argument("--lora_modelC_path", type=str, default=None,
                        help="The pretrained model checkpoint for lora weights.")
    # parser.add_argument("--lora_modelA_path", type=str, default="/home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module-autosplit/output/Record/version_3_r8_al32_Lr0.00003/model-A/checkpoint-33000")
    # parser.add_argument("--lora_modelC_path", type=str, default="/home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module-autosplit/output/Record/version_3_r8_al32_Lr0.00003/model-C/checkpoint-33000")
    # parser.add_argument("--quantization_bit", type=int, help="quantization bit",  default=4)

    '''=====================================================training args====================================================='''  
    parser.add_argument("--batch_size", type=int, help="traing batch size", default=2) 
    parser.add_argument("--max_train_samples", type=int, default=25000)
    parser.add_argument("--max_eval_samples", type=int, default=None)  
    parser.add_argument("--seed", type=int, help="random seed", default=42)
    parser.add_argument("--device",type=str, default = 'cuda:1')
    parser.add_argument("--max_grad_norm", type=float, help='max grad_clipping norm', default=1.0)
    parser.add_argument("--lr", type=float, help='learning rate', default=1e-5)
    parser.add_argument("--betas", type=tuple, help='(adamhf)optimizer betas', default=(0.9,0.999))
    parser.add_argument("--eps", type=float, help='(adamhf)optimizer eps', default=1e-5)
    parser.add_argument("--weight_decay", type=float, help='(adamhf)optimizer weight decay', default=0.01)
    parser.add_argument("--output_dir", type=str, help = 'output folder path', default="/home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module-autosplit/output/50NewBatchTrain")
    parser.add_argument("--save_step", type=int, help = 'step to save the prefix encoder', default=1000)
    parser.add_argument("--overwrite_output_dir", type=bool, help = 'whether to overwrite the output folder', default=False)
    parser.add_argument("--block_size", type=int, help = 'I dont know yet', default=None)
    parser.add_argument("--custom_lora", action="store_true", help='Whether to customize lora weights.', default=True)
    parser.add_argument("--wandb", action="store_true", help='Whether to use wandb.', default=False)    

    '''=====================================================datasets args====================================================='''  
    parser.add_argument("--data_fold", type=str, default="/home/zhangzishuai/SplitFederated-LLaMA/Dataset/Record")
    parser.add_argument("--dataset_name", type=str,choices=["Xsum", "CoQA","MultiRC","Record"], default="Record")
    parser.add_argument("--max_source_length", type=int, help="构建数据集时input的长度(不包括标签)", default = 700)
    parser.add_argument("--max_target_length", type=int, default = 20)
    parser.add_argument("--source_prefix", type=str, default=None)
    parser.add_argument("--passage_column", type=str, default="passage")
    parser.add_argument("--passage2_column", type=str, default=None)
    parser.add_argument("--premise_column", type=str, default=None)
    parser.add_argument("--question_column", type=str, default="qas")
    parser.add_argument("--answer_column", type=str, default=None) 
    parser.add_argument("--history_column", type=str, default=None) 
    parser.add_argument("--preprocessing_num_workers", type=int, default=6)
    parser.add_argument("--overwrite_cache", type=str, default=None) 
    parser.add_argument("--ignore_pad_token_for_loss", type=bool, default=True)
    parser.add_argument("--max_step", type=int, help='number of max training steps, should be same with serve side!', default=15000)
    parser.add_argument("--dataloader_num_workers", type=int, default=6)
    parser.add_argument("--dataloader_pin_memory", type=bool, default=True)
    parser.add_argument("--dataloader_drop_last", type=bool, default=True)    
    parser.add_argument("--train_file", type=str,  default=None)
    # parser.add_argument("--train_file", type=str,  default="sample_train.jsonl") 
    parser.add_argument("--validation_file", type=str,  default=None)
    # parser.add_argument("--test_file", type=str,  default=None)
    parser.add_argument("--test_file", type=str,  default="test.jsonl")

    parser.add_argument("--from_pickle", action="store_true", help='Whether to run predictions on the test set.', default=True)
    parser.add_argument("--iter_test", action="store_true", help='Whether to run predictions on the test set.', default=False)
    parser.add_argument("--data2pic", action="store_true", help='Whether to run predictions on the test set.', default=False)
    parser.add_argument("--status", type=str, default=None, choices=["train","batch_train","eval","predict", "inference"])

    '''=====================================================Fl Arguments====================================================='''  
    parser.add_argument("--server_ip", type=str, help='ip4_address:port.', default="10.143.12.74:7030")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--cleint0_lora_modelA_path", type=str, default=None)
    parser.add_argument("--cleint0_lora_modelC_path", type=str, default=None)
    parser.add_argument("--cleint1_lora_modelA_path", type=str, default=None)
    parser.add_argument("--cleint1_lora_modelC_path", type=str, default=None)
    #parser.add_argument("--raw_prompt", type=str, help='Prompt received from bash script.', default=None)  
    parser.add_argument("--pipe", type=str, default=None)
    parser.add_argument("--max_predict_samples", type=int, default=10)
    parser.add_argument("--max_output_length", type=int, default = 256)
    parser.add_argument("--val_max_target_length", type=int, default=None)
    parser.add_argument("--do_train", action="store_true", help='Whether to run training.', default=True)
    parser.add_argument("--do_eval", action="store_true", help='Whether to run eval on the dev set.', default=False)
    parser.add_argument("--do_predict", action="store_true", help='Whether to run predictions on the test set.', default=False)
    parser.add_argument("--do_inference", action="store_true", help='Whether to inference using prompt from users.', default=False)
    parser.add_argument("--do_netdisplay", action="store_true", help='Whether to Netdisplay.', default=False)
    parser.add_argument("--do_BatchTrain", action="store_true", help='Whether to inference using prompt from users.', default=True)
    parser.add_argument("--AggregatedClients", action="store_true", help='Whether to run predictions with aggregated model As and model Cs.', default=False)
    parser.add_argument("--AggregatedAvg", action="store_true", help='Whether to run predictions with aggregated model As and model Cs.', default=False)
    parser.add_argument("--AggregatedIter", action="store_true", help='Whether to run predictions with aggregated model As and model Cs.', default=False)
    parser.add_argument("--resume_from_checkpoint", type=bool, help='Whether to resume from checkpoint.', default=False)
    parser.add_argument("--dry", type=bool, default=False, required=False, help="Do a dry-run to check the client")
    parser.add_argument("--client_id", type=int, default=1, choices=range(0, 10), required=False, help="Specifies the artificial data partition of CIFAR10 to be used. \
        Picks partition 0 by default")
    parser.add_argument("--toy", action="store_true", help="Set to true to quicky run the client using only 10 datasamples. \
        Useful for testing purposes. Default: False")
    parser.add_argument("--use_cuda", type=bool, default=True, required=False, help="Set to true to use GPU. Default: False")
    parser.add_argument("--model", type=str, default="efficientnet", choices=["efficientnet", "alexnet"],
        help="Use either Efficientnet or Alexnet models. If you want to achieve differential privacy, please use the Alexnet model")
    parser.add_argument("--per_client_steps", type=int, help="the number of the traing steps of every client", default=100)

    '''=====================================================Split Arguments====================================================='''  
    parser.add_argument("--modelA_layers", type=int, choices=[1,2,3,4,5], default=5 )
    parser.add_argument("--modelC_layers", type=int, choices=[1,2,3,4,5], default=3 )
    '''=====================================================DP_Arguments====================================================='''  
    parser.add_argument("--add_DP_gradient", action="store_true", help='Whether to add DP on gradients.', default=False)
    # parser.add_argument("--add_DP_hidden", action="store_true", help='Whether to add DP on hidden states.', default=False)
    parser.add_argument("--add_DP_hidden", type=lambda x: x.lower() == 'true', help='Whether to add DP on hidden states.', default=False)
    parser.add_argument("--DP_grad_norm_clip", type=float, help='hidden_grad_norm', default=0.04)
    parser.add_argument("--grad_noise", type=float, help='grad_noise', default=0.02)
    '''=====================================================Attack_Arguments====================================================='''
    parser.add_argument("--Attack", action="store_true", help='Whether to attack to reconstruct dataset.', default=True)
    parser.add_argument("--Attack_load_lora", type=lambda x: x.lower() == 'true', help='Load lora to make client 1 differen from client 0 at first.', default=True)
    parser.add_argument("--Attack_lora_checkpoint", type=str, help="the number of the traing steps of every client", default="checkpoint-50")
    parser.add_argument("--Client1_VariedEmbeding", help='Whether to attack to reconstruct dataset.', default=False)
    args = parser.parse_args()

    return args

class Fed_Client(NumPyClient):
    def __init__(self, datalist, optimizer, schedule, models, tokenizer, FL_args, child_conn) -> None:
        super(Fed_Client, self).__init__()
        self.child_conn = child_conn
        self.FL_args = FL_args

        # 高斯噪声绝对值的数学期望与标准差的关系：E(|N(0,σ)|) = σ * sqrt(2/π)
        compensation_factor = math.sqrt(math.pi / 2)
        self.FL_args.grad_noise = self.FL_args.grad_noise * compensation_factor
        logger.info(f"补偿因子之后噪声标准差为:{self.FL_args.grad_noise}")

        self.datalist = datalist 

        if FL_args.do_train:
            self.len_dataset = len(datalist[0])
        elif FL_args.do_eval:
            self.len_dataset = len(datalist)
        elif FL_args.do_predict and not FL_args.do_inference:
            self.len_dataset = len(datalist)
        else:
            self.len_dataset = None
        logger.info(f"len_dataset:{self.len_dataset}")
        self.model = models  # [ModelA, ModelB]
        logger.info(f"modelA_dtype:{self.model[0].dtype},modelC_dtype:{self.model[1].dtype}")
        logger.warning(f"modelA_layers:{self.FL_args.modelA_layers}, modelC_layers:{self.FL_args.modelC_layers},modelB_layers:{32-self.FL_args.modelA_layers-self.FL_args.modelC_layers}")
        self.optim = optimizer
        self.schedule = schedule
        self.label = None
        self.torch_dtype = self.init_torch_dtype()
        if self.FL_args.do_predict or self.FL_args.do_inference:
            self.FL_args.batch_size = 1
            logger.info(f"batch_size强制设置为{self.FL_args.batch_size}")
            # 初始化模型A和C的缓存
            # 对pkv_modelA赋值的时候，不需要使用深拷贝，因为目的是pkv_modelA自动变化值
            self.pkv_modelA=DynamicCache()
            self.pkv_modelC=DynamicCache()
            for _ in range(32-self.FL_args.modelC_layers):
                self.pkv_modelC.key_cache.append(torch.tensor(0, dtype=torch.int8, device=self.FL_args.device))
                self.pkv_modelC.value_cache.append(torch.tensor(0, dtype=torch.int8, device=self.FL_args.device))

        # 在这里初始化为了不传递attention_mask，而是直接本地生成，同一个输入，不同层的attn_mask相同
        self.causal_mask = self.init_casual_mask()
        self.answer_turn=0 #记录当前答案的轮次

        #不同数据集不同配置
        if self.FL_args.dataset_name == "CoQA":
            self.id=[] #for CoQA 官方数据集
            self.turn_id = [] #for CoQA 官方数据集
            self.score_list=[] #测试集，记录每条数据的结果
        if self.FL_args.dataset_name == "MultiRC":
            self.pid=None
            self.qid=None
            self.aid=None
            self.debug_list=[] #记录yes/no转换成1/0失败的案例
            self.final_output_list = [] 
            self.pred_map = defaultdict(lambda: {"pid": "", "qid": "",  "scores_map": {}})
        if self.FL_args.dataset_name == "Xsum":
            self.id=[]
            self.all_predictions=[]
            self.all_labels=[]
        if self.FL_args.dataset_name == "Record":
            self.id=[] #for Record 官方数据集
            self.turn_id = [] #for Record 官方数据集
            self.score_list=[] #测试集，记录每条数据的结果

        self.f1 = None
        self.train_metricx = {'step':[], 'loss':[]}
        self.eval_output = {'pred_id':[]}
        self.device = torch.device(FL_args.device if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        self.client_id = torch.tensor(self.FL_args.client_id).to(self.device)
        self.data = None
        self.attn_mask = None #测试，模型C的attention_mask
        self.cache_position = None # 测试，指示pkv的位置
        self.raw_attn_mask = None # 测试，模型A第一次传递的原始的attention_mask，后续只传递token
        self.train_attn_mask = None # 训练，模型C的attention_mask
        self.real_input_num = None  # 训练，模型A传递这个代替传递attention_mask给模型B
        self.total_pred=[]

        if self.FL_args.do_train:
            self.model[0].train()
            self.model[1].train()
            self.tokenizer = tokenizer
            logger.info(f"下面是self.client1 模型A的可训练参数")
            logger.info("\n")  
            for name, param in self.model[0].named_parameters():
                logger.info(f"Parameter Name: {name}, Requires Grad: {param.requires_grad}")
            logger.info(f"下面是self.client1 模型C的可训练参数")
            logger.info("\n")  
            for name, param in self.model[1].named_parameters():
                logger.info(f"Parameter Name: {name}, Requires Grad: {param.requires_grad}")

        else:
            self.model[0].eval()
            self.model[1].eval()
            self.tokenizer = tokenizer
        # For 画图
        self.batch_norms_history = defaultdict(list)
        self.global_step = 0
        if self.FL_args.add_DP_gradient:
            logger.warning(f"在C->B梯度上加噪")
            if self.FL_args.DP_grad_norm_clip is None:
                logger.info(f"DP_grad_norm_clip is None")
            else:
                logger.info(f"DP_grad_norm_clip is {self.FL_args.DP_grad_norm_clip}")
        if self.FL_args.add_DP_hidden:
            logger.warning(f"在A->B hidden state上加噪")

    def set_parameters(self, parameters):
        """Receive parameters and apply them to the local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)

        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract model parameters and return them as a list of numpy arrays."""
        # for glm, the function of extracting model parameters should be written here
        return [val.cpu().numpy() for _, val in self.model[0].state_dict().items()] 
    
    def get_parameters_part3(self, config: Dict[str, Scalar]):
        """Extract model parameters and return them as a list of numpy arrays."""

        return [val.cpu().numpy() for _, val in self.model[1].state_dict().items()] 
    
    def init_torch_dtype(self):
        dtype_mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "int32": torch.int32,
        }
        # 检查FL_args.torch_dtype是否有效
        if self.FL_args.torch_dtype not in dtype_mapping:
            raise ValueError(f"Unsupported data type: {self.FL_args.torch_dtype}. Supported types: {list(dtype_mapping.keys())}")
        # 获取对应的数据类型
        dtype = dtype_mapping[self.FL_args.torch_dtype]
        return dtype

    def init_casual_mask(self):
        causal_mask = torch.full((4096, 4096), fill_value=1)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask.to(self.torch_dtype)   # torch.finfo(dtype)需要浮点数输入
        causal_mask = (
            causal_mask[None, None, :, :].repeat(self.FL_args.batch_size, 1, 1, 1).to(self.torch_dtype) * torch.finfo(self.torch_dtype).min
        )
        causal_mask = causal_mask.to(dtype=self.torch_dtype, device=self.FL_args.device)

        logger.info("初始化casual_mask完成")
        return causal_mask

    def update_causal_mask(self, attention_mask,position_ids):
        causal_mask = self.causal_mask.detach().clone()
        #attention_mask [1,102]
        # seq_length = position_ids.shape[0]
        # batch_size = self.FL_args.batch_size
        dtype = self.torch_dtype
        # device = attention_mask.device

        if attention_mask is not None and attention_mask.dim() == 2:
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
            causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(
                padding_mask, torch.finfo(dtype).min
            )

        if attention_mask is not None and torch.any(attention_mask != 1):
            causal_mask = causal_mask.mul(~torch.all(causal_mask == causal_mask.min(), dim=-1)[..., None]).to(
                dtype
            )

        return causal_mask

    def fit(self, parameters: NDArrays=None, config: Dict[str, Scalar]=None):
        feature = []
        num_sample = 0
        metrics = {}
        if config['type'] == 1:
            # print('fit client part A')
            if self.FL_args.do_train:
                feature = self.train_fit_partA(current_step=config['current_step'], need_test_data=config['need_test_data'])
            else:
                feature = self.predict_fit_partA(current_step=config['current_step'], need_test_data=config['need_test_data'])
            metrics = {}
        
        elif config['type']==2:
            # print('fit client part C')

            if self.FL_args.do_train:
                feature, num_sample, metrics = self.train_fit_partC(parameters, config)

            else : # 目前只有do_train
                feature, num_sample, metrics = self.predict_fit_partC(parameters, config, self.child_conn)
                

        elif config['type']==3:
            # print('backward client part A')
            feature, num_sample, metrics = self.backward_modelA(parameters, config)

        return feature, num_sample, metrics
    
    def train_fit_partA(self, parameters: NDArrays=None, current_step=0, need_test_data=0):
        current_epoch = current_step // self.len_dataset
        current_batch =current_step % self.len_dataset
        data = self.datalist[current_epoch][current_batch]['input_ids'].to(self.device) #torch.Size([2, 920])
        attn_mask = self.datalist[current_epoch][current_batch]['attention_mask'].to(self.device) #torch.Size([2, 920])
        self.train_attn_mask = attn_mask
        # att_mask.shape torch.Size([1, 1, 4096, 4096])
        self.real_input_num = torch.sum(attn_mask == 1, dim=1).to(torch.int64)#torch.Size([2]) [674,367]
        # data.type:torch.Size([1, 640])
        label = self.datalist[current_epoch][current_batch]['labels'].to(self.device)
        #label.type: torch.Size([2, 920])

        if current_epoch==0 and current_batch==0:
            #打印第一个数据
            decoded_data = self.tokenizer.decode(data[0,:].squeeze(), skip_special_tokens=True)
            label_example = label[0,:].squeeze()
            label_example = label_example[label_example != -100]
            decoded_label = self.tokenizer.decode(label_example, skip_special_tokens=True)
            logger.info(f"当前data:{data.shape},当前attn_mask:{attn_mask.shape},当前label:{label.shape}")
            logger.info(f"\n当前data:\n{decoded_data}\n当前label:\n{decoded_label}")

        # forward pass
        #start_1 = time.time() # 
        f1 = self.model[0](input_ids=data, attention_mask=attn_mask)  # data (batchsize,seq_len)
        # f1 = self.model[0](input_ids=data)  # data (batchsize,seq_len)

        #end_1 = time.time() #
        #cost_1 = end_1 - start_1 # 
        #logger.info(f"前向传播modelA_cost:{cost_1}") #
        #start_2 = time.time() # 
        hidden_A = f1.last_hidden_state.clone().detach() #.requires_grad_(True)
        # hidden_A.shape torch.Size([1, 352, 4096]) (batch_size, ,embedding_size)

        p_ids = f1.position_ids.clone().detach()

        self.label = label
        self.f1 = f1

        #以下在传输的hidden_A中加入噪声，可插拔
        if self.FL_args.add_DP_hidden:
            avg_hidden_A = torch.mean(torch.abs(hidden_A)).item()
            logger.info(f"Average hidden_A magnitude: {avg_hidden_A:.15f}")
            if self.FL_args.wandb:
                wandb.log({"Average_hidden_A_magnitude": avg_hidden_A},step= self.global_step)

            noise_std = torch.tensor(self.FL_args.grad_noise, dtype=self.torch_dtype)
            logger.info(f"噪声标准差为: {noise_std:.15f}")
            noise = torch.randn_like(hidden_A) * noise_std

            # # 计算噪声和梯度夹角
            # cos_sim = calculate_cosine_similarity(1e6 * hidden_gradient.detach().clone(), 1e6 * noise)
            # avg_cos_sim = torch.mean(cos_sim).item()
            # logger.info(f"Adjusted cosine similarity: {avg_cos_sim:.15f}")
            # wandb.log({"Avg_seq_cos": avg_cos_sim},step= self.global_step)
            
            # if avg_cos_sim < 0.1:
            #     logger.warning("夹角小于0.1,正在进行噪声投影！")
            #     noise = 1e-6 * orthogonalize_noise(1e6 * hidden_gradient.detach().clone(),1e6 *  noise)

            avg_noise = torch.mean(torch.abs(noise)).item()
            logger.info(f"Average noise magnitude: {avg_noise:.15f}")
            if self.FL_args.wandb:
                wandb.log({"Average_noise_magnitude": avg_noise},step= self.global_step)
            hidden_A.add_(noise)  # 原地操作直接修改梯度值  # torch.Size([2, 920, 4096])

            avg_hidden_A = torch.mean(torch.abs(hidden_A)).item()
            logger.info(f"添加噪声后Average hidden_A magnitude: {avg_hidden_A:.15f}")
            if self.FL_args.wandb:
                wandb.log({"添加噪声后Average_hidden_A_magnitude": avg_hidden_A},step= self.global_step)
        else:
            logger.info(f"前向传播hidden_A未添加噪声")            

        #以在在传输的hidden_A中加入噪声，可插拔


        # od = collections.OrderedDict([('hidden_stateA', hidden_A),
        #                 ('real_input_num', self.real_input_num),
        #                 ('position_ids', p_ids)])
        if self.FL_args.Attack:
            logger.warning("您正在发送您的数据到云端")
            od = collections.OrderedDict([('hidden_stateA', hidden_A),
                ('real_input_num', self.real_input_num),
                ('position_ids', p_ids),
                ('true_data',data)])
        else:
            od = collections.OrderedDict([('hidden_stateA', hidden_A),
                            ('real_input_num', self.real_input_num),
                            ('position_ids', p_ids)])

        feature = [val.cpu().numpy() for _, val in od.items()]

        #end_2 = time.time() #
        #cost_2 = end_2 - start_2 # 
        #logger.info(f"打包modelA传输的数据_cost:{cost_2},此时时间为:{time.time()}") #

        # 保存feature看传输大小
        # stored_path = "/home/zhangzishuai/SplitFederated-LLaMA/pickle-store/llama_fit_A.pickle"
        # with open(stored_path, 'wb') as f:
        #     pickle.dump(feature, f)
        #     logger.info(f"保存pickle数据完成")
        torch.cuda.empty_cache()
        return feature # 数据不包含梯度！

    def train_fit_partC(self, parameters: NDArrays=None, config: Dict[str, Scalar]=None , child_conn=None):       
        # ff2 = self.set_parameters(ff2) # recieve feture of model2 from server
        # print('there')

        # past_key_values = torch.from_numpy(parameters[3]).to(self.device) #  torch.Size([2, 5, 1, 32, 128])
        # # print('past_key_values size in fit C:',past_key_values.size())
        # past_key_values = past_key_values.clone().detach().requires_grad_(True)
        # pkv2 = ([past_key_values])
        '''=====================================================计算loss并更新模型C====================================================='''
        #start_3 = time.time() # 
        # if not self.FL_args.do_train:
        #     pkv=torch.from_numpy(parameters[3]).to(self.device) #  layers in one tensor
        #     pkv_tensors = torch.split(pkv, 1, dim=0)
        #     past_key = pkv_tensors[0:31]
        #     past_value = pkv_tensors[31:62]
        #     # past_key, past_value = torch.split(pkv, 31, dim=0)
        #     # past_key = [torch.split(past_key, 1, dim=0)]
        #     # past_value = [torch.split(past_value, 1, dim=0)]
        #     pkv2= DynamicCache.from_legacy_cache(None)
        #     for past_key in past_key:
        #         pkv2.key_cache.append(past_key.requires_grad_(True))
        #     for past_value in past_value:
        #         pkv2.value_cache.append(past_value.requires_grad_(True))
        #     pkv2.seen_tokens = torch.from_numpy(parameters[4])

        hidden = torch.from_numpy(parameters[0]).to(self.device).requires_grad_(True)
        # att_mask = torch.from_numpy(parameters[1]).to(self.device)
        p_ids = torch.from_numpy(parameters[1]).to(self.device)
        #end_3 = time.time() # 
        #cost_3 = end_3 - start_3 # 
        #logger.info(f"modelC解析接收到的数据准备前向传播_cost:{cost_3}") #
        #start_4 = time.time() # 
        # 将attn_mask从[0,1]转换成[4096,4096]大概是这样
        self.train_attn_mask =self.update_causal_mask(self.train_attn_mask, p_ids)
        final_output = self.model[1](position_ids = p_ids,
                            attention_mask = self.train_attn_mask,
                            hidden_states = hidden,
                            labels = self.label)

        #end_4 = time.time() # 
        #cost_4 = end_4 - start_4 # 
        #logger.info(f"modelC前向传播_cost:{cost_4}") # 
        # final_output.logits.size:torch.Size([1, 128, 130528])
        
        metrics = {}
        gradient = []
        # labels = []

        #start_5 = time.time() # 

        # log loss
        loss = final_output[0]
        # print(loss.item())
        log(INFO, 'step %s lr %s  loss %s:', config["current_step"], self.optim[1].param_groups[0]['lr'], loss.item())
        self.global_step += 1
        if self.FL_args.wandb:
            wandb.log({"loss": loss.item()},step = self.global_step)

        # for param_group in self.optim[1].param_groups:
        #     for param in param_group['params']:
        #         print(f"params shape:{param.shape}")
        #     # print(type(param_group))
        #     print([type(value) for value in param_group.values()])
        #     print('查看学习率: ',param_group['lr'])

        # backward        
        self.optim[1].zero_grad()
        loss.backward()  # model3 step
        # print(f1.grad)
    
        nn.utils.clip_grad_norm_(self.model[1].parameters(), self.FL_args.max_grad_norm)
        self.optim[1].step()
        self.schedule[1].step()

        hidden_gradient = hidden.grad # tensor

        #以下是模块，可插拔
        # 统计
        # batch_norms = []
        # batch_means = []
        # batch_variances = []

        # for i in range(hidden_gradient.size(0)):
        #     grad = hidden_gradient[i]  # 获取第i个批次的梯度，形状 [920, 4096]
            
        #     # 计算二范数（Frobenius 范数）
        #     norm = torch.norm(grad, p='fro')  # 或者 grad.view(-1).norm(p=2)
        #     logger.info(f"梯度范数为{norm.item()}")
        #     if i==1 and self.FL_args.wandb:
        #         wandb.log({"grad_norm": norm.item()},step = self.global_step)
        #     batch_norms.append(norm.item())
        #     self.batch_norms_history[i].append((self.global_step, norm.item()))
        #     # 计算均值
        #     mean = grad.mean()
        #     batch_means.append(mean.item())
            
        #     # 计算方差（使用样本方差，除以n）
        #     var = grad.var(unbiased=False)
        #     batch_variances.append(var.item())

        # # 打印统计结果
        # for i in range(len(batch_norms)):
        #     logger.info(f'Batch {i}:')
        #     logger.info(f'  Norm: {batch_norms[i]:.4e}')  # Norm: 1.9470e-02
        #     logger.info(f'  Mean: {batch_means[i]:.4e}')  # Mean: -0.0000e+00
        #     logger.info(f'  Variance: {batch_variances[i]:.4e}')  # Variance: 0.0000e+00


        # if self.global_step==self.FL_args.max_step:
        #     plt.figure(figsize=(10, 6))

        #     # 为每个批次绘制曲线
        #     for batch_idx in self.batch_norms_history:
        #         steps = [step for step, norm in self.batch_norms_history[batch_idx]]
        #         norms = [norm for step, norm in self.batch_norms_history[batch_idx]]
        #         plt.plot(steps, norms, label=f"Batch {batch_idx}", marker="o", markersize=4)

        #     # 添加图例和标签
        #     plt.xlabel("Training Step")
        #     plt.ylabel("Gradient Norm (Frobenius)")
        #     plt.title("Gradient Norm per Batch vs. Training Step")
        #     plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")  # 将图例放在图外
        #     plt.grid(True, linestyle="--", alpha=0.7)
        #     plt.tight_layout()
        #     plt.savefig("/home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module/output/DP_debug/DP_Norm_per_step.png")

        # 加DP
        bs_grad_norm={}
        if self.FL_args.add_DP_gradient:
            # 针对每一个batch梯度裁剪
            if self.FL_args.DP_grad_norm_clip is not None:
                for i in range(hidden_gradient.shape[0]):
                    # 计算梯度的范数
                    grad_norm = torch.norm(hidden_gradient[i,:,:], p=2)
                    avg_grad = torch.mean(torch.abs(hidden_gradient[i,:,:])).item()
                    logger.info(f"Average gradient magnitude: {avg_grad:.7f}")
                    if i==1 and self.FL_args.wandb:
                        wandb.log({"Average gradient_magnitude": avg_grad},step= self.global_step)
                    bs_grad_norm[f"{i}"]=grad_norm
                    # 如果范数超过阈值，则进行裁剪
                    if grad_norm > self.FL_args.DP_grad_norm_clip:
                        logger.warning(f"Gradient norm: {grad_norm:.4f},大于梯度裁剪阈值，进行梯度裁剪！")
                        hidden_gradient[i,:,:] = hidden_gradient[i,:,:] * (self.FL_args.DP_grad_norm_clip / grad_norm)
                        bs_grad_norm[f"{i}"]=self.FL_args.DP_grad_norm_clip
                        
            # 添加噪声

            # 梯度最小范数的1/10
            # min_grad_norm = min(bs_grad_norm.values())
            # noise_std = min_grad_norm/10  # 根据你的梯度范数设置的噪声标准差
            # 标准DP噪声大小算法
            # step_budget=self.FL_args.privacy_budget/self.FL_args.max_step
            # sample_rate = 1/self.len_dataset
            # noise_std = 2*self.FL_args.DP_grad_norm_clip *sample_rate* np.sqrt(2 * np.log(1.25/self.FL_args.privacy_delta)) / step_budget
            # 自定义noise大小
            noise_std = torch.tensor(self.FL_args.grad_noise, dtype=self.torch_dtype)
            logger.info(f"噪声标准差为: {noise_std:.15f}")
            noise = torch.randn_like(hidden_gradient) * noise_std

            # # 计算噪声和梯度夹角
            # cos_sim = calculate_cosine_similarity(1e6 * hidden_gradient.detach().clone(), 1e6 * noise)
            # avg_cos_sim = torch.mean(cos_sim).item()
            # logger.info(f"Adjusted cosine similarity: {avg_cos_sim:.15f}")
            # wandb.log({"Avg_seq_cos": avg_cos_sim},step= self.global_step)
            
            # if avg_cos_sim < 0.1:
            #     logger.warning("夹角小于0.1,正在进行噪声投影！")
            #     noise = 1e-6 * orthogonalize_noise(1e6 * hidden_gradient.detach().clone(),1e6 *  noise)

            avg_noise = torch.mean(torch.abs(noise)).item()
            logger.info(f"Average noise magnitude: {avg_noise:.15f}")
            if self.FL_args.wandb:
                wandb.log({"Average_noise_magnitude": avg_noise},step= self.global_step)
            hidden_gradient.add_(noise)  # 原地操作直接修改梯度值  # torch.Size([2, 920, 4096])
        else:
            logger.info(f"反向传播未添加DP噪声")

        #以上是模块，可插拔



        od = collections.OrderedDict([('hiddden_gradient', hidden_gradient),
                        ('client_id', self.client_id)])
        gradient = [val.cpu().numpy() for _, val in od.items()]

        metrics['loss'] = loss.item()

        self.train_metricx['step'].append(config['current_step'])
        self.train_metricx['loss'].append(loss.item())


        torch.cuda.empty_cache()
        return gradient, 0, metrics  # GetParametersRes

    def predict_fit_partA(self, parameters: NDArrays=None, current_step=0, need_test_data=0):
        if need_test_data ==1:  # 根据query前向传播
            if not self.FL_args.do_inference: #测试集
                data = self.datalist[current_step]['input_ids'].to(self.device)
                # 去掉pad符号
                data = data[data != self.tokenizer.pad_token_id]
                data=data.unsqueeze(0)
                logger.info(f"data.shape:{data.shape}")
                
                # #2025-2-24
                # #如果data.shape的第二个维度大于1024，则截断;实测跟截断没关系
                # if data.shape[1] > 1200:
                #     data = torch.cat([data[:, :600],data[:, -600:]],dim=1)
                #     logger.warning(f"已截断！")

                attn_mask = self.datalist[current_step]['attention_mask'].to(self.device)
                # 去掉pad符号
                attn_mask=attn_mask[attn_mask != 0]
                attn_mask=attn_mask.unsqueeze(0)
                # #如果attn_mask.shape的第二个维度大于1024，则截断
                # if attn_mask.shape[1] > 1200:
                #     attn_mask = attn_mask[:, -1200:]

                label = self.datalist[current_step]['labels'].to(self.device)  # 测试集有标签
                # label从[1,1] 降维成 [1]
                label=label.squeeze()

                # 不同数据集配置不同
                if self.FL_args.dataset_name == "CoQA":
                    id = self.datalist[current_step]['id'] 
                    turn_id = self.datalist[current_step]['turn_id']
                if self.FL_args.dataset_name == "MultiRC":
                    self.pid = self.datalist[current_step]['pid'] 
                    self.qid = self.datalist[current_step]['qid'] 
                    self.aid = self.datalist[current_step]['aid']
                if self.FL_args.dataset_name == "Xsum":
                    id = self.datalist[current_step]['id']
                if self.FL_args.dataset_name == "Record":
                    id = self.datalist[current_step]['id'] 
                    turn_id = self.datalist[current_step]['turn_id']
 
            else: # 用户query
                data = self.datalist['input_ids'].to(self.device)
                logger.info(data.shape)
                # 网页展示的时候第一个data从datalist里来，后续的从管道中得到
                if self.child_conn is not None:
                    if self.child_conn.poll():
                        max_target_length = self.FL_args.val_max_target_length
                        # 将prompt转换数据集的格式
                        prompt = self.child_conn.recv()
                        # inputs= []
                        # inputs.append(prompt)
                        model_inputs = self.tokenizer([prompt], max_length=self.FL_args.max_source_length, truncation=True, padding=True)
                        model_inputs = torch.tensor(model_inputs["input_ids"])
                        # # 获取输入的形状
                        # seq_len =  self.FL_args.max_output_length+2
                        # # 重新填充到原始长度
                        # pad_length = seq_len - model_inputs.size(1)
                        # attn_mask = torch.cat((torch.zeros(pad_length, dtype=torch.int64), torch.ones(model_inputs.size(1), dtype=torch.int64)),dim=0)
                        # attn_mask = attn_mask.to(self.device)
                        # attn_mask = attn_mask.unsqueeze(0)
                        # padded_input = torch.nn.functional.pad(model_inputs, ( pad_length, 0), value=3)
                        # data = padded_input.to(self.device)
                        # label = None # 用户推理没有标签
                        # 重新填充到原始长度
                        attn_mask = torch.ones(model_inputs.size(1), dtype=torch.int64)
                        attn_mask = attn_mask.to(self.device)
                        attn_mask = attn_mask.unsqueeze(0)
                        data = model_inputs.to(self.device)
                        label = None # 用户推理没有标签
                else:
                    # print(f"data:{data}")
                    attn_mask = self.datalist['attention_mask'].to(self.device)
                    # print(f"attn_mask:{attn_mask}")
                    label = None # 用户推理没有标签

            self.data = data
            self.raw_attn_mask = attn_mask
            self.attn_mask = attn_mask
            self.cache_position = torch.arange(attn_mask.shape[-1], dtype=torch.int64, device=self.device)
            
        else: # 拼接query前向传播
            data = self.data.to(self.device)
            attn_mask = self.attn_mask.to(self.device)
            self.cache_position = torch.tensor(attn_mask.shape[-1] - 1, dtype=torch.int64, device=self.device)
            self.cache_position = self.cache_position.unsqueeze(0)
            if not self.FL_args.do_inference:  # 进行测试，推理的时候没有label
                label = self.datalist[current_step]['labels'].to(self.device)

                if self.FL_args.dataset_name == "MultiRC":
                    self.pid = self.datalist[current_step]['pid'] 
                    self.qid = self.datalist[current_step]['qid'] 
                    self.aid = self.datalist[current_step]['aid'] 
                if self.FL_args.dataset_name == "CoQA":
                    id = self.datalist[current_step]['id'] 
                    turn_id = self.datalist[current_step]['turn_id']
                if self.FL_args.dataset_name == "Xsum":
                    id = self.datalist[current_step]['id']
                if self.FL_args.dataset_name == "Record":
                    id = self.datalist[current_step]['id'] 
                    turn_id = self.datalist[current_step]['turn_id']

            else:
                label = None
            #logger.info(f"label:{label}")

        # forward pass
        # start_1 = time.time() # 

        with torch.no_grad():
            f1 = self.model[0](input_ids=data, attention_mask=attn_mask, past_key_values=self.pkv_modelA,cache_position=self.cache_position, use_cache=True)
        # 第一次推理
        # data:  torch.Size([1, 102])
        # attn_mask: torch.Size([1, 102])
        # past_key_values : 仅仅初始化
        # cache_position :torch.Size([102])
        #拼接后
        # data:  torch.Size([1, 1])
        # attn_mask: torch.Size([1, 103])
        # past_key_values : 保存的pkv
        # cache_position : tensor([103], device='cuda:0')


        # end_1 = time.time() #
        # cost_1 = end_1 - start_1 # 
        # logger.info(f"前向传播modelA_cost:{cost_1}") #

        # start_2 = time.time() # 
        hidden_A = f1.last_hidden_state.clone().detach() #.requires_grad_(True)
        # hidden_A.shape torch.Size([1, 102, 4096]) (batch_size, ,embedding_size) #拼接后 torch.Size([1, 1, 4096])
        att_mask = f1.attention_mask.clone().detach()   
        # att_mask.shape torch.Size([1, 1, 4096, 4096])   #拼接后 torch.Size([1, 1, 4096, 4096])
        p_ids = f1.position_ids.clone().detach()
        # p_ids.shape  torch.Size([1, 102])   #拼接后 torch.Size([1, 1])

        #f1.past_key_values.key_cache[0].shape:torch.Size([1, 32, 102, 128])   [batchsize_num_heads,seq_len, d_model]
        # pkv_modelA.update(f1.past_key_values.key_cache[0],f1.past_key_values.value_cache[0],layer_idx=0)
        self.pkv_modelA = f1.past_key_values
        # torch.Size([2, 32, 168, 128])    [batchsize,num_heads, seq_len, dimension] 

        self.label = label
        if not self.FL_args.do_inference:
            if self.FL_args.dataset_name == "CoQA": #推理的时候才有这个,CQA数据集专用
                self.id = id
                self.turn_id = turn_id
            if self.FL_args.dataset_name == "Xsum":
                self.id = id
            if self.FL_args.dataset_name == "Record":
                self.id = id
                self.turn_id = turn_id


        # if self.att_mask is not None:
        #     if not torch.equal(self.att_mask, att_mask):
        #         logger.info("当前self.att_mask <不> 等上一个att_mask")
        #     else:
        #         logger.info("当前self.att_mask等于上一个att_mask")

        if need_test_data ==1:
            od = collections.OrderedDict([('hidden_stateA', hidden_A),
                                        ('attention_mask', att_mask),
                                        ('position_ids', p_ids),
                                        ('raw_attn_mask', self.raw_attn_mask)])
        else:
            od = collections.OrderedDict([('hidden_stateA', hidden_A),
                            ('position_ids', p_ids)])

        feature = [val.cpu().numpy() for _, val in od.items()]

        # end_2 = time.time() #
        # cost_2 = end_2 - start_2 # 
        # logger.info(f"打包modelA传输的数据_cost:{cost_2},此时时间为:{time.time()}") #

        # # 保存feature看传输大小
        # stored_path = "/home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module-autosplit/output/CoQA/debug/llama_fit_A.pickle"
        # with open(stored_path, 'wb') as f:
        #     pickle.dump(feature, f)
        #     logger.info(f"保存pickle数据完成")

        return feature # 数据不包含梯度！

    def predict_fit_partC(self, parameters: NDArrays=None, config: Dict[str, Scalar]=None , child_conn=None):       

        '''=====================================================前向传播模型C====================================================='''
        # start_3 = time.time() # 

        if config['need_test_data'] == 1:
            hidden = torch.from_numpy(parameters[0]).to(self.device)
            att_mask = torch.from_numpy(parameters[1]).to(self.device)
            p_ids = torch.from_numpy(parameters[2]).to(self.device)
        else:
            hidden = torch.from_numpy(parameters[0]).to(self.device)
            p_ids = torch.from_numpy(parameters[1]).to(self.device)
        # end_3 = time.time() # 
        # cost_3 = end_3 - start_3 # 
        # logger.info(f"modelC解析接收到的数据准备前向传播_cost:{cost_3}") #
        # start_4 = time.time() # 

        with torch.no_grad():
            if config['need_test_data'] == 1:
                final_output = self.model[1](position_ids = p_ids,
                        attention_mask = att_mask,
                        hidden_states = hidden,
                        #past_key_values = None, #必须是None ,否则不对
                        past_key_values = self.pkv_modelC,
                        cache_position=self.cache_position,
                        use_cache=True)                
            else:
                modelC_attn_mask =self.update_causal_mask(self.attn_mask, p_ids)   
                #每一层模型内部接触的attntion 应该是在第一层之前就生成好的，第一层生成时需要利用update_causal_mask函数更新attn
                final_output = self.model[1](position_ids = p_ids,
                        attention_mask = modelC_attn_mask,
                        hidden_states = hidden,
                        past_key_values = self.pkv_modelC,
                        cache_position=self.cache_position,
                        use_cache=True) 
        # pkv_modelA.update(final_output.past_key_values.key_cache[0],final_output.past_key_values.value_cache[0],layer_idx=0)
        self.pkv_modelC = final_output.past_key_values
        # end_4 = time.time() # 
        # cost_4 = end_4 - start_4 # 
        # logger.info(f"modelC前向传播_cost:{cost_4}") # 
        # final_output.logits.size:torch.Size([1, 128, 130528])
        
        metrics = {}
        gradient = []
        # labels = []
        # start_5 = time.time() # 

        metrics['pred_finished'] = 0
        #logger.info( 'DO_PREDICT')
        if len(self.total_pred)==0:
            print('DO_PREDICT')

        # 6/26新增
        # final_output: tuple(loss, logits)
        # final_output[1]是一个元组，长度为1，final_output[1][0]是一个元组，长度为2，每一个都是一个tensor, 形状都是torch.Size([256, 1, 32, 128])
        #print(f"final_output.logits:{final_output.logits}")
        # tensor(,batch_size,, vocab_size)

        # pred_ids = final_output.logits[:, -1:].argmax(-1)

        logits=final_output.logits[:,-1,:] #(B，vocab size)
        # probs = F.softmax(logits, dim=-1)
        # 此时有多种生成方法，这里使用采样生成方法
        # print("self.data:",self.data)
        # logger.info(f"logits:{logits[0]}")
        pred_ids = self.sample_text(logits)

        #argmax才需要unsqueeze(0)
        # pred_ids = torch.argmax(logits, dim=-1) 
        # pred_ids = pred_ids.unsqueeze(0)

        self.total_pred.append(int(pred_ids[0][0]))


        # 解码预测结果
        tokenizer = AutoTokenizer.from_pretrained("/home/zhangzishuai/SplitFederated-LLaMA/Models/Llama2-7B-chat-service", trust_remote_code=True)
        response = tokenizer.decode(int(pred_ids[0][0]), skip_special_tokens=True)
        logger.info(f"这是第{len(self.total_pred)}个pred_ids:{pred_ids},解码结果:{response}")

        #发送给网页
        if self.child_conn is not None:
            if int(pred_ids[0][0]) in [162,155,141]:
                pass
            elif int(pred_ids[0][0])==243:
                self.child_conn.send("笑脸emoji")
            else:
                self.child_conn.send(response)          

        # 找到padding的第一个位置，把data的第一个padding符号替换掉
        #确保pred_ids的shape(batch,seq)=[1,1]
        self.data, self.attn_mask, self.cache_position = self.attn_mask_update(self.attn_mask, pred_ids,self.cache_position)

        # end_5 = time.time() # 
        # cost_5 = end_5 - start_5 # 
        # logger.info(f"modelC得到pred_decode_cost:{cost_5}") # 
        # 如果输出结束符号，则告诉服务端停止
        if pred_ids[0][-1] == 2 or len(self.total_pred) == self.FL_args.max_output_length:
            #结束符2
            #self.FL_args.pred_finished = True
            #使用decoder输出所有的预测结果
            tokenizer = AutoTokenizer.from_pretrained("/home/zhangzishuai/SplitFederated-LLaMA/Models/Llama2-7B-chat-service", trust_remote_code=True)
            tokenizer.pad_token_id = 3
            output_text = tokenizer.decode(self.total_pred, skip_special_tokens=True)
            logger.info(f"response:{output_text}")

            if self.FL_args.dataset_name == "CoQA":
                # 用于CoQA数据官方测试集合保存预测结果
                pred_data = {
                    "id": tokenizer.decode(self.id[0], skip_special_tokens=True),          # 假设存在 current_id 变量存储当前样本的ID
                    "turn_id": int(self.turn_id), 
                    "answer": response
                }
                prediction_save_path = os.path.join(self.FL_args.lora_modelA_path,'prediction_kvcache.json')
                with open(prediction_save_path, 'a', encoding='utf-8') as f:
                    json.dump(pred_data, f, ensure_ascii=False, indent=4)
                    f.write('\n')  # 方便读取时逐行解析

                score = compute_metrics(self.total_pred, self.label, self.tokenizer, self.FL_args)
                logger.info(f"当前回答的score:{score}")
                self.score_list.append(score)
                correct_predictions = sum(self.score_list)
                accuracy = correct_predictions / len(self.score_list)
                logger.info(f"已回答{len(self.score_list)}个问题,准确率为:{accuracy:.6f}")

            if self.FL_args.dataset_name == "MultiRC":
                self.answer_turn += 1
                score = yes_no_to_score(output_text)
                pid_str = tokenizer.decode(self.pid[0], skip_special_tokens=True)
                qid_str = tokenizer.decode(self.qid[0], skip_special_tokens=True)
                aid_str = tokenizer.decode(self.aid[0], skip_special_tokens=True)

                if score is None:
                    decoded_labels = tokenizer.decode(self.label.squeeze(), skip_special_tokens=True)
                    # 既不包含 "yes" 也不包含 "no"
                    debug_info = {
                        "pid": pid_str,
                        "qid": qid_str,
                        "aid": aid_str,
                        "output_text": output_text,
                        "label": decoded_labels
                    }
                    logger.warning(f"output_text 未识别为 yes/no: {debug_info}")
                    self.debug_list.append(debug_info)

                if score is not None:
                    # 2) 确保 self.pred_map 里有 pid,qid
                    pred_map_key = (pid_str, qid_str)
                    self.pred_map[pred_map_key]["pid"] = pid_str
                    self.pred_map[pred_map_key]["qid"] = qid_str
                    aid_int = int(aid_str)
                    self.pred_map[pred_map_key]["scores_map"][aid_int] = score

                    pred_data = {
                        "pid": pid_str,
                        "qid": qid_str,
                        "aid": aid_str,                
                        # "turn_id": int(id), 
                        "answer": output_text
                    }
                    logger.info(f"第{self.answer_turn}个问题。pred_data:{pred_data}")
                    # prediction_save_path = os.path.join(self.FL_args.lora_modelA_path,'Fl-llama_prediction.json')
                    if self.FL_args.add_DP_hidden:
                        prediction_save_path = os.path.join(self.FL_args.lora_modelA_path, 'Foward_with_noise_prediction.json')
                    else:
                        prediction_save_path = os.path.join(self.FL_args.lora_modelA_path, 'Foward_no_noise_prediction.json')

                    with open(prediction_save_path, 'a', encoding='utf-8') as f:
                        json.dump(pred_data, f, ensure_ascii=False, indent=4)
                        f.write('\n')  # 方便读取时逐行解析
                    
                    if self.FL_args.ignore_pad_token_for_loss:
                    # where(a,b,c)  若a为真，则输出该位置上的原始值，如果条件为假，则输出第二个参数中的值c
                        self.label = torch.where(self.label != -100, self.label, tokenizer.pad_token_id)
                    decoded_labels = tokenizer.decode(self.label.squeeze(), skip_special_tokens=True)
                    logger.info(f"第{self.answer_turn}个问题。labels:{decoded_labels}")

                    if self.answer_turn==self.FL_args.max_predict_samples:
                        for (pid_str, qid_str), info in self.pred_map.items():
                            pid = info["pid"]
                            qid = info["qid"]
                            scores_map = info["scores_map"]  # dict, e.g. {10:1, 11:0, ...}

                            # 如果 aid_int 是纯数字，那么可以按数值排序
                            # 如果有字符串混入，你可能要区分数字key和字符串key或者自定义排序
                            sorted_aids = sorted(scores_map.keys(), key=lambda x: int(x) if isinstance(x, str) and x.isdigit() else x)

                            # 根据排序后得到对应的分值
                            # 例如 [ scores_map[10], scores_map[11], ... ]
                            scores_list = [scores_map[aid] for aid in sorted_aids]

                            self.final_output_list.append({
                                "pid": pid,
                                "qid": qid,
                                "scores": scores_list
                            })

                        # 把 self.final_output_list 写出到 JSON 文件
                        # 如果前向传播没有噪声
                        if self.FL_args.add_DP_hidden:
                            prediction_save_path = os.path.join(self.FL_args.lora_modelA_path, 'Foward_with_noise_prediction_for_eval.json')
                        else:
                            prediction_save_path = os.path.join(self.FL_args.lora_modelA_path, 'Foward_no_noise_prediction_for_eval.json')
                        with open(prediction_save_path, 'w', encoding='utf-8') as f:
                            json.dump(self.final_output_list, f, ensure_ascii=False, indent=2)

                        logger.info(f"已将 {len(self.final_output_list)} 条记录写入 {prediction_save_path}")

                        if self.debug_list:
                            debug_save_path = os.path.join(self.FL_args.lora_modelA_path, 'unexpected_outputs.json')
                            with open(debug_save_path, 'w', encoding='utf-8') as f:
                                json.dump(self.debug_list, f, ensure_ascii=False, indent=2)
                            logger.warning(f"共有 {len(self.debug_list)} 个回答既不包含yes也不包含no,已写入 {debug_save_path}")
                        else:
                            logger.info("全部回答解析成yes/no成功")

                        official_eval(prediction_save_path,logger)

            if self.FL_args.dataset_name == "Xsum":
                self.answer_turn += 1
                # for Compute Rouge
                if self.FL_args.ignore_pad_token_for_loss:
                    self.label = torch.where(self.label != -100, self.label, tokenizer.pad_token_id)
                decoded_labels = tokenizer.decode(self.label.squeeze(), skip_special_tokens=True)
                logger.info(f"第{self.answer_turn}个问题。labels:{decoded_labels}")

                # 用于数据官方测试集合保存预测结果
                pred_data = {
                    "id": tokenizer.decode(self.id[0], skip_special_tokens=True),          # 假设存在 current_id 变量存储当前样本的ID
                    "answer": output_text
                }
                logger.info(f"第{self.answer_turn}个问题。pred_data:{pred_data}")
                prediction_save_path = os.path.join(self.FL_args.lora_modelA_path,'prediction_kvcache.json')
                with open(prediction_save_path, 'a', encoding='utf-8') as f:
                    json.dump(pred_data, f, ensure_ascii=False, indent=4)
                    f.write('\n')  # 方便读取时逐行解析

                self.all_predictions.append(pred_data["answer"])
                self.all_labels.append(decoded_labels)

                if self.answer_turn==self.FL_args.max_predict_samples:
                    score_dict = compute_metrics_Xsum(self.all_predictions, self.all_labels)
                    logger.info(f"score_dict:{score_dict}")

            if self.FL_args.dataset_name == "Record":
                self.answer_turn += 1
                turn_id=tokenizer.decode(self.turn_id[0], skip_special_tokens=True)
                pred_data = {      # 假设存在 current_id 变量存储当前样本的ID
                    f"{turn_id}": output_text
                }
                logger.info(f"第{self.answer_turn}个问题。pred_data:{pred_data}")
                if self.FL_args.lora_modelA_path is None:
                    prediction_save_path=os.path.join(self.FL_args.output_dir,'distributed_prediction.json')
                else:
                    prediction_save_path = os.path.join(self.FL_args.lora_modelA_path,'distributed_prediction.json')
                with open(prediction_save_path, 'a', encoding='utf-8') as f:
                    json.dump(pred_data, f, ensure_ascii=False, indent=4)
                    f.write('\n')  # 方便读取时逐行解析

                score = compute_metrics_Record(output_text, self.label, tokenizer, self.FL_args)
                logger.info(f"当前回答的score:{score}")
                self.score_list.append(score)
                correct_predictions = sum(self.score_list)
                accuracy = correct_predictions / len(self.score_list)
                logger.info(f"已回答{len(self.score_list)}个问题,当前准确率为:{accuracy:.6f}")

            if self.child_conn is not None:
                self.child_conn.send("回答完毕")

            self.data = None
            self.cache_position = None
            self.raw_attn_mask = None
            self.attn_mask = None
            self.total_pred=[]
            del self.pkv_modelA
            del self.pkv_modelC
            torch.cuda.set_device(self.FL_args.device)
            torch.cuda.empty_cache() 
            self.pkv_modelA=DynamicCache()
            self.pkv_modelC=DynamicCache()
            for _ in range(32-self.FL_args.modelC_layers):
                self.pkv_modelC.key_cache.append(torch.tensor(0, dtype=torch.int8, device=self.FL_args.device))
                self.pkv_modelC.value_cache.append(torch.tensor(0, dtype=torch.int8, device=self.FL_args.device))
            metrics['pred_finished'] = 1


        return gradient, 0, metrics  # GetParametersRes

    #采样生成代码
    def sample_text(self, probs, temperature=0.1, top_p=0.95):
        # 应用temperature   probs的torch.Size([1, 32000])
        #如果 temperature 小于1，则使得高概率的token更加突出，低概率的token被压制；如果大于1，则反之。

        # # 按概率排序
        # sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        
        # # 计算累积概率,后一个概率是前一个概率加上它本身概率的和
        # cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # # 移除低于阈值的token
        # sorted_indices_to_remove = cumulative_probs > top_p
        # sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        # # sorted_indices_to_remove[..., 0] = 0  # 保留最高概率的token
        # sorted_indices_to_remove[..., 0] = False  # 保留最高概率的token
        
        # # 屏蔽掉被移除的indices
        # indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        # probs[indices_to_remove] = 0
    
        #温度采样
        probs = torch.softmax(probs/temperature, dim=-1)

        # 采样
        next_token = torch.multinomial(probs, num_samples=1)

        # 贪婪采样策略
        # next_token = probs.argmax(-1)
        
        return next_token

    def attn_mask_update(self, attn_mask, pred_id, cache_position):
        #region
        # 获取输入的形状
        # batch_size, seq_len = attn_mask.size()
        # 去掉padding部分
        # non_padding_input = attn_mask[attn_mask != 0]

        # 拼接输出
        # new_attn_mask_size = torch.cat((non_padding_input, [[1]]), dim=1)  #为多batchsize做准备
        #non_padding_input.size()类型是torch.Size，可能是tuple,不能直接转换成整数
        # new_attn_mask_size = int(non_padding_input.size()[-1])+1

        # # 重新填充到原始长度
        # pad_length = seq_len - new_attn_mask_size 
        # if pad_length >= 0:
        #     attn_mask = torch.cat((torch.zeros(pad_length, dtype=torch.int64), torch.ones(new_attn_mask_size, dtype=torch.int64)),dim=0)
        # else:
        #     attn_mask =  torch.ones(seq_len, dtype=torch.int64)
        #endregion

        # 不进行以上填充
        attn_mask = torch.cat([attn_mask, attn_mask.new_ones((attn_mask.shape[0], 1))], dim=-1) 
        padded_data = pred_id  #更新输入token
        cache_position = cache_position[-1:] + 1 

        return padded_data, attn_mask, cache_position

    def backward_modelA(self, parameters: NDArrays=None, config: Dict[str, Scalar]=None):
        # ff1_grad = self.set_parameters(args.ff1_grad) # recieve gradient of ff1 from server

        hidden_gradient = torch.from_numpy(parameters[0]).to(self.device)
        # print(hidden_gradient)

        # print(type(lastpkv_gradient))
        self.optim[0].zero_grad()
        # self.f1.last_hidden_state.backward(hidden_gradient, retain_graph=True)

        self.f1.last_hidden_state.backward(hidden_gradient)
        
        # 梯度剪裁
        nn.utils.clip_grad_norm_(self.model[0].parameters(), self.FL_args.max_grad_norm)
        self.optim[0].step()
        self.schedule[0].step()

        if (config['current_step']+1) % self.FL_args.save_step == 0:
            # self.save_model(config['current_step'], self.optim, self.schedule)
            if self.FL_args.do_BatchTrain:
                self.Batch_save_model(config['current_step'])
            else:
                self.save_model(config['current_step'])

        
        if config['current_step'] == self.FL_args.max_step-1 :
            print('Now saving training metrix!')
            # 指定保存路径
            json_save_path = os.path.join(self.FL_args.output_dir,'loss.json')

            # 保存字典到文件
            with open(json_save_path, 'w') as f:
                json.dump(self.train_metricx, f)

        return [], 0, {}

    # def save_model(self, step, optimizer, schedule):
    def save_model(self, step):
        logger.info("Saving Lora")  

        for i, model in enumerate(self.model):
            if i==0:
                check_fold = os.path.join(self.FL_args.output_dir, "model-A")
                model.save_pretrained(os.path.join(check_fold, 'checkpoint-{}'.format(str(step+1))))
            elif i==1: 
                check_fold = os.path.join(self.FL_args.output_dir, "model-C")
                model.save_pretrained(os.path.join(check_fold, 'checkpoint-{}'.format(str(step+1))))
        logger.info("Lora权重保存成功!") 

    def Batch_save_model(self, step):
        logger.info("Saving Lora")  

        for i, model in enumerate(self.model):
            if i==0:
                check_fold = os.path.join(self.FL_args.output_dir, "client_1_model-A")
                model.save_pretrained(os.path.join(check_fold, 'checkpoint-{}'.format(str(step+1))))
            elif i==1: 
                check_fold = os.path.join(self.FL_args.output_dir, "client_1_model-C")
                model.save_pretrained(os.path.join(check_fold, 'checkpoint-{}'.format(str(step+1))))
        logger.info("Lora权重保存成功!") 


#region
    # def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
    #     #self.set_parameters(parameters)

    #     #loss, accuracy = test_client(self.model, self.valloader, self.device)
    #     loss=0.00
    #     accuracy=0

    #     return float(loss), len(self.valloader), {"accuracy": accuracy}
# def client_dry_run(device: torch.device = "cpu"):
#     """Weak tests to check whether all client methods are working as expected."""

#     model = utils.load_efficientnet(classes=10)
#     trainset, testset = utils.load_partition(0)
#     trainset = trainset.select(range(10))
#     testset = testset.select(range(10))
#     client = Fed_Client(trainset, testset, device)
#     client.fit(
#         utils.get_model_params(model),
#         {"batch_size": 16, "local_epochs": 1},
#     )

#     client.evaluate(utils.get_model_params(model), {"val_steps": 32})

#     print("Dry Run Successful")


# def old_compute_metrics(preds, labels, tokenizer, args):
#         # preds, labels = eval_preds  # batched token list 

#         if isinstance(preds, tuple): 
#             preds = preds[0]
#         decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
#         if args.ignore_pad_token_for_loss:
#             # Replace -100 in the labels as we can't decode them.
#             labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#             logger.info(f"labels:{labels}")
#             logger.info(f"labels.shape:{labels.shape}")
            
#         decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

#          # FOR BOOLQ!!!
#         # for i in range(len(decoded_labels)):
#         #     if decoded_preds[i]=='':
#         #         decoded_preds[i]='Yes' if decoded_labels[i]=='No' else 'No'

#         print('predictions:', preds)
#         print('decode_predictions:', decoded_preds)
#         print('labels:', labels)
#         print('decode_labels:', decoded_labels)
        
#         # auc = 0
#         # for i in range(len(decoded_labels)):
#         #     if decoded_labels[i] == decoded_preds[i]:
#         #         auc + 1
        
#         # acc = auc/len(decoded_labels)

#         # preds_1 = []
#         # labels_1 = []
#         # # CB
#         # for item in decoded_preds:
#         #     if item=='Yes':
#         #         preds_1.append(0)
#         #     elif item=='No':
#         #         preds_1.append(1)
#         #     elif item=='Maybe':
#         #         preds_1.append(2)

#         # # for item in decoded_labels:
#         # #     labels_1.append(1 if item=='Yes' else 0)
#         # for item in decoded_labels:
#         #     if item=='Yes':
#         #         labels_1.append(0)
#         #     elif item=='No':
#         #         labels_1.append(1)
#         #     elif item=='Maybe':
#         #         labels_1.append(2)

#         # pred_1 = np.array(preds_1)
#         # label_1 = np.array(labels_1)

#         # print(pred_1)
#         # print(label_1)

#         acc = accuracy_score(label_1,pred_1)
#         cm = confusion_matrix(decoded_labels, decoded_preds, labels=['Yes','No','Maybe']) # CB
#         f1score = f1_score(label_1, pred_1, average='macro')

#         # print('metrics:')
#         # print('Accuracy:', acc)
#         # print('Confusion_matrix:', cm)
#         # print('F1-Score',f1score)

#         score_dict = {
#             # "rouge-1": [],
#             # "rouge-2": [],
#             # "rouge-l": [],
#             # "bleu-4": [] ,
#             "Accuracy": [],
#             "Confusion_matrix":[],
#             "F1-Score":[] 
#         }
#         # for pred, label in zip(decoded_preds, decoded_labels):
#         #     # ...input
#         #     hypothesis = list(jieba.cut(pred))
#         #     reference = list(jieba.cut(label))
#         #     rouge = Rouge()
#         #     scores = rouge.get_scores(' '.join(hypothesis) , ' '.join(reference))
#         #     result = scores[0]
            
#         #     for k, v in result.items():
#         #         score_dict[k].append(round(v["f"] * 100, 4))
#         #     bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
#         #     score_dict["bleu-4"].append(round(bleu_score * 100, 4))
#         #     # score_dict["Accuracy"].append(round(acc * 100, 4))

#         # for k, v in score_dict.items():
#         #     score_dict[k] = float(np.mean(v))

#         score_dict["Accuracy"].append(acc)
#         score_dict["Confusion_matrix"].append(cm.tolist)
#         score_dict["F1-Score"]= round(f1score* 100, 4)
                
#         return score_dict
#endregion
    '''=====================================================计算指标====================================================='''
def compute_metrics(preds, labels, tokenizer, args):
        # preds, labels = eval_preds  # 仅一个回答的 token list 

        decoded_preds = tokenizer.decode(preds, skip_special_tokens=True)
        if args.ignore_pad_token_for_loss:
            # where(a,b,c)  若a为真，则输出该位置上的原始值，如果条件为假，则输出第二个参数中的值c
            labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)
            #logger.info(f"labels:{labels}")

        decoded_labels = tokenizer.decode(labels.squeeze(), skip_special_tokens=True) #其他测试集的label是token
        # 按照 ;;;;分割不同的答案
        decoded_labels = decoded_labels.split(";;;;")
        # 去除分隔符后，得到多个版本的答案
        decoded_labels = [answer.strip() for answer in decoded_labels if answer.strip()]

        logger.info(f'decode_predictions:{decoded_preds}')
        logger.info(f'decode_labels:{decoded_labels}')
        
        # if decoded_preds == decoded_labels: #精确匹配才算回答正确
        if decoded_preds in decoded_labels: #模糊匹配就算回答正确
            logger.info(f"回答正确!decode_labels=decode_predictions={decoded_preds}")
            score = 1
        else:
            logger.info(f"回答错误!decode_labels={decoded_labels} ;  decode_predictions={decoded_preds}")
            score = 0
        
        return score

def compute_metrics_MultiRC(preds, labels):

    score_dict = {
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": [],}
    for pred, label in zip(preds, labels):
        # ...input
        prediction = list(jieba.cut(pred))
        abstract = list(jieba.cut(label))
        rouge = Rouge()
        scores = rouge.get_scores(' '.join(prediction) , ' '.join(abstract))
        result = scores[0]
    for k, v in result.items():
        score_dict[k].append(round(v["f"] * 100, 4))
    for k, v in score_dict.items():
        score_dict[k] = float(np.mean(v))

    return score_dict

def compute_metrics_Xsum(preds, labels):

    score_dict = {
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": [],}
    for pred, label in zip(preds, labels):
        # ...input
        prediction = list(jieba.cut(pred))
        abstract = list(jieba.cut(label))
        rouge = Rouge()
        scores = rouge.get_scores(' '.join(prediction) , ' '.join(abstract))
        result = scores[0]
    for k, v in result.items():
        score_dict[k].append(round(v["f"] * 100, 4))
    for k, v in score_dict.items():
        score_dict[k] = float(np.mean(v))

    return score_dict

def yes_no_to_score(generated_text: str):
    """
    检查生成文本是否包含 'yes' 或 'no'。
    - 若包含 'yes'，返回 1
    - 若包含 'no'，返回 0
    - 若都不包含，返回 None
    """
    txt_lower = generated_text.strip().lower()
    if "yes" in txt_lower:
        return 1
    elif "no" in txt_lower:
        return 0
    else:
        return None

def compute_metrics_Record(output_text, labels, tokenizer, args):
    # decoded_preds = tokenizer.decode(preds, skip_special_tokens=True)

    if args.ignore_pad_token_for_loss:
        # where(a,b,c)  若a为真，则输出该位置上的原始值，如果条件为假，则输出第二个参数中的值c
        labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)
        #logger.info(f"labels:{labels}")
    # decoded_labels = tokenizer.decode(labels.squeeze(), skip_special_tokens=True)

    decoded_labels = tokenizer.decode(labels.squeeze(), skip_special_tokens=True) #其他测试集的label是token
    #logger.info('predictions:', preds)
    # 按照 ;;;;分割不同的答案
    decoded_labels = decoded_labels.split(";;;;")
    # 去除分隔符后，得到多个版本的答案
    decoded_labels = [answer.strip() for answer in decoded_labels if answer.strip()]        

    logger.info(f'decode_predictions:{output_text}')
    logger.info(f'decode_labels:{decoded_labels}')
    
    # if decoded_preds == decoded_labels:
    if output_text in decoded_labels: #模糊匹配就算回答正确
        logger.info(f"回答正确!decode_labels=decode_predictions={output_text}")
        score = 1
    else:
        logger.info(f"回答错误!decode_labels={decoded_labels} ;  decode_predictions={output_text}")
        score = 0

    return score

def main(prompt: str =None , child_conn=None) -> None:
    '''=====================================================设置参数====================================================='''
    FL_args = FLparser()

    if prompt is not None:
        FL_args.do_inference = True
        FL_args.do_predict = True

    FL_args.modelA_config_name = FL_args.modelA_name_or_path
    FL_args.modelC_config_name = FL_args.modelC_name_or_path
    if type(FL_args.target_modules)==str:
        FL_args.target_modules = FL_args.target_modules.split(',')
        if len(FL_args.target_modules)==8:
            FL_args.Client1_VariedEmbeding=True
            logger.warning("多客户端训练时客户端1可变词表!")
        elif len(FL_args.target_modules)==4:
            FL_args.Client1_VariedEmbeding=False
            logger.warning("多客户端训练时客户端1不可变词表!")
        else:
            logger.warning("请检查target_modules!")
    # device
    device = torch.device(FL_args.device if torch.cuda.is_available() and FL_args.use_cuda else "cpu")

    # 确保实验可复现
    torch.manual_seed(FL_args.seed)
    random.seed(FL_args.seed)
    np.random.seed(FL_args.seed)
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True) 

    # FL_args.data_fold = '/home/zhangzishuai/SplitFederated-LLaMA/Dataset/Friends/ReadComprehension' #华佗数据集
    # FL_args.data_fold = "/home/zhangzishuai/SplitFederated-LLaMA/Dataset/CoQA" # CoQA数据集

    # # For huatuo
    # FL_args.premise_column = 'premise'
    # FL_args.question_column = 'question'
    # FL_args.passage_column = 'question'
    # FL_args.answer_column = 'answer'

    # # For Friends
    # FL_args.question_column = 'query'
    # FL_args.passage_column = 'utterances'
    # FL_args.answer_column = 'answer'

    # For CoQA
    # FL_args.question_column = 'questions'
    # FL_args.passage_column = 'story'
    # FL_args.answer_column = 'answers'

    FL_args.val_max_target_length = 200
    FL_args.prompt = prompt

    '''=====================================================聚合函数====================================================='''
    def aggregate_models(model1, model2):
        """聚合两个模型的参数"""
        aggregated_model = {}
        
        # 遍历模型1的参数
        for key in model1.state_dict().keys():
            if key in model2.state_dict():
                # 对应参数相加并取平均
                aggregated_model[key] = (model1.state_dict()[key] + model2.state_dict()[key]) / 2
            else:
                raise ValueError(f"参数 {key} 在两个模型中不匹配")
        
        # 创建一个新的模型并加载聚合后的参数
        new_model = model1.__class__(model1.config)  # 假设两个模型是同一类型
        new_model.load_state_dict(aggregated_model)

        return new_model

    '''=====================================================配置模型参数====================================================='''
    start_set_model=time.time()
    # 删除初始化模型后还需对状态字典进行量化
    dtype_mapping = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int32": torch.int32,
    }
    # 检查FL_args.torch_dtype是否有效
    if FL_args.torch_dtype not in dtype_mapping:
        raise ValueError(f"Unsupported data type: {FL_args.torch_dtype}. Supported types: {list(dtype_mapping.keys())}")
    # 获取对应的数据类型
    model_torch_dtype = dtype_mapping[FL_args.torch_dtype]

    def adjust_modelA_state_dict(original_dict, target_layers):
        """调整ModelA的状态字典键名"""
        new_dict = OrderedDict()
        # 提取非layer的参数（如embed_tokens, norm等）
        for k in original_dict:
            if not k.startswith("model.layers"):
                new_dict[k] = original_dict[k]
        
        # 动态匹配layer参数
        for layer_idx in range(target_layers):
            src_prefix = f"model.layers.{layer_idx}."
            tgt_prefix = f"model.layers.{layer_idx}."
            
            # 匹配所有该层的参数
            for k in original_dict:
                if k.startswith(src_prefix):
                    new_dict[k] = original_dict[k]
                    
        return new_dict
        
    def adjust_modelC_state_dict(original_dict, target_layers, total_layers=5):
        """调整ModelC的状态字典键名"""
        new_dict = OrderedDict()

        # 动态映射layer参数
        for new_layer_idx in range(target_layers):
            original_layer_idx = total_layers + new_layer_idx - target_layers
            src_prefix = f"model.layers.{original_layer_idx}."
            tgt_prefix = f"model.layers.{new_layer_idx}."
            
            # 需要双重验证：既要匹配原始层号，又要确保在目标范围内
            for k in original_dict:
                if k.startswith(src_prefix):
                    # 替换层号部分
                    new_key = k.replace(src_prefix, tgt_prefix, 1)
                    new_dict[new_key] = original_dict[k]

        # 提取非layer的参数,norm,head
        for k in original_dict:
            if not k.startswith("model.layers"):
                new_dict[k] = original_dict[k]                    
        return new_dict

    def adjust_modelA_Lora_state_dict(original_dict, target_layers):
        """调整ModelA的状态字典键名"""
        new_dict = OrderedDict()
        # 提取非layer的参数（如embed_tokens, norm等）
        for k in original_dict:
            if not k.startswith("base_model.model.model.layers"):
                new_dict[k] = original_dict[k]
        
        # 动态匹配layer参数
        for layer_idx in range(target_layers):
            src_prefix = f"base_model.model.model.layers.{layer_idx}."
            tgt_prefix = f"base_model.model.model.layers.{layer_idx}."
            
            # 匹配所有该层的参数
            for k in original_dict:
                if k.startswith(src_prefix):
                    parts = k.split('.')
                    if 'lora_A' in parts:
                        index = parts.index('lora_A')  # 找到目标位置
                        parts.insert(index + 1, 'default')  # 在目标后插入
                        new_key = '.'.join(parts)
                        new_dict[new_key] = original_dict[k]
                    if 'lora_B' in parts:
                        index = parts.index('lora_B')  # 找到目标位置
                        parts.insert(index + 1, 'default')  # 在目标后插入
                        new_key = '.'.join(parts)
                        new_dict[new_key] = original_dict[k]
                    
                    
        return new_dict

    def adjust_modelC_Lora_state_dict(original_dict, target_layers, total_layers=5):
        """调整ModelC的状态字典键名"""
        new_dict = OrderedDict()

        # 动态映射layer参数
        for new_layer_idx in range(target_layers):
            original_layer_idx = total_layers + new_layer_idx - target_layers
            src_prefix = f"base_model.model.model.layers.{original_layer_idx}."
            tgt_prefix = f"base_model.model.model.layers.{new_layer_idx}."
            
            # 需要双重验证：既要匹配原始层号，又要确保在目标范围内
            for k in original_dict:
                if k.startswith(src_prefix):
                    # 替换层号部分
                    new_key = k.replace(src_prefix, tgt_prefix, 1)
                    parts = new_key.split('.')
                    if 'lora_A' in parts:
                        index = parts.index('lora_A')  # 找到目标位置
                        parts.insert(index + 1, 'default')  # 在目标后插入
                        new_key = '.'.join(parts)
                        new_dict[new_key] = original_dict[k]
                    if 'lora_B' in parts:
                        index = parts.index('lora_B')  # 找到目标位置
                        parts.insert(index + 1, 'default')  # 在目标后插入
                        new_key = '.'.join(parts)
                        new_dict[new_key] = original_dict[k]

                    new_dict[new_key] = original_dict[k]

        # 提取非layer的参数,norm,head
        for k in original_dict:
            if not k.startswith("base_model.model.model.layers"):
                new_dict[k] = original_dict[k]    

        return new_dict

    if FL_args.do_train:
        # Set seed before initializing model.
        set_seed(FL_args.seed)

        config_kwargs = {
            "cache_dir": FL_args.cache_dir,
            "revision": FL_args.model_revision,
            "use_auth_token": True if FL_args.use_auth_token else None,
        }
        if FL_args.modelA_config_name and FL_args.modelC_config_name:   # 从这里进入
            configA = AutoConfig.from_pretrained(FL_args.modelA_config_name, **config_kwargs)
            configC = AutoConfig.from_pretrained(FL_args.modelC_config_name, **config_kwargs)
        elif FL_args.modelA_name_or_path and FL_args.modelC_name_or_path:
            configA = AutoConfig.from_pretrained(FL_args.modelA_name_or_path, **config_kwargs)
            configC = AutoConfig.from_pretrained(FL_args.modelC_name_or_path, **config_kwargs)
        else:
            configA = CONFIG_MAPPING[FL_args.model_type]()
            configC = CONFIG_MAPPING[FL_args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")
            # if FL_args.config_overrides is not None:
            #     logger.info(f"Overriding config: {FL_args.config_overrides}")
            #     config.update_from_string(FL_args.config_overrides)
            #     logger.info(f"New config: {config}")
            

        tokenizer_kwargs = {
            "cache_dir": FL_args.cache_dir,
            "use_fast": FL_args.use_fast_tokenizer,
            "revision": FL_args.model_revision,
            "use_auth_token": True if FL_args.use_auth_token else None,
            "padding_side":'left'
        }
        if FL_args.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(FL_args.tokenizer_name, **tokenizer_kwargs)
        elif FL_args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(FL_args.model_name_or_path, **tokenizer_kwargs)
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
                "You can do it from another script, save it, and load it from here, using --tokenizer_name."
            )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = 3

        logger.info(f"FL_args.target_modules:{FL_args.target_modules}")

        # lora 配置
        clientA_layer_id = list(range(FL_args.modelA_layers))

        # #仅微调q、k、v、o
        # t_m_c = []
        # for index in FL_args.target_modules:
        #     logger.info(f"index:{index}")
        #     for i in clientA_layer_id:
        #         t_m_c.append(str(f'model.layers.{i}.self_attn' + '.' + index))

        # 同时微调多个模块
        t_m_c = []
        self_attn_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        other_layer_modules = ['gate_proj', 'up_proj', 'down_proj']
        global_modules = ['embed_tokens']  # 不属于某一层（如模型顶层）
        for index in FL_args.target_modules:
            logger.info(f"index: {index}")
            if index in global_modules:
                # 顶层模块，如 embed_tokens
                t_m_c.append(f'model.{index}')
            elif index in self_attn_modules:
                # 属于 self_attn 的模块
                for i in clientA_layer_id:
                    t_m_c.append(f'model.layers.{i}.self_attn.{index}')
            elif index in other_layer_modules:
                # 属于每层的其他模块，比如 feed_forward
                for i in clientA_layer_id:
                    t_m_c.append(f'model.layers.{i}.mlp.{index}')
            else:
                logger.warning(f"未知模块类型: {index}，未添加到 target_modules")

        #logger.info(f"t_m_c:{t_m_c}")
        ModelA_target_modules = t_m_c
        lora_configA = LoraConfig(
            r=FL_args.lora_r,
            lora_alpha=FL_args.lora_alpha,
            # target_modules=["query_key_value"],
            # target_modules =  ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            target_modules =  ModelA_target_modules,
            fan_in_fan_out = False,
            lora_dropout=0.05,
            inference_mode=False,
            bias="none",
            task_type="CAUSAL_LM",
        )
        logger.info(f"lora_A_config:{lora_configA}")

        clientC_layer_id = list(range(FL_args.modelC_layers))

        # # 仅微调q、k、v、o
        # t_m_c = []
        # for index in FL_args.target_modules:
        #     logger.info(f"index:{index}")
        #     for i in clientC_layer_id:
        #         t_m_c.append(str(f'model.layers.{i}.self_attn' + '.' + index))
        # #logger.info(f"t_m_c:{t_m_c}")

        # 同时微调多个模块
        t_m_c = []
        self_attn_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        other_layer_modules = ['gate_proj', 'up_proj', 'down_proj']
        global_modules = ['embed_tokens']  # 不属于某一层（如模型顶层）
        for index in FL_args.target_modules:
            logger.info(f"index: {index}")
            if index in global_modules:
                # 顶层模块，如 embed_tokens
                logger.info("模型C没有 embed_tokens ")
            elif index in self_attn_modules:
                # 属于 self_attn 的模块
                for i in clientC_layer_id:
                    t_m_c.append(f'model.layers.{i}.self_attn.{index}')
            elif index in other_layer_modules:
                # 属于每层的其他模块，比如 feed_forward
                for i in clientC_layer_id:
                    t_m_c.append(f'model.layers.{i}.mlp.{index}')
            else:
                logger.warning(f"未知模块类型: {index}，未添加到 target_modules")

        ModelC_target_modules = t_m_c
        lora_configC = LoraConfig(
            r=FL_args.lora_r,
            lora_alpha=FL_args.lora_alpha,
            # target_modules=["query_key_value"],
            # target_modules =  ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            target_modules =  ModelC_target_modules,
            fan_in_fan_out = False,
            lora_dropout=0.05,
            inference_mode=False,
            bias="none",
            task_type="CAUSAL_LM",
        )
        logger.info(f"lora_C_config:{lora_configC}")


        # model_A config and initial
        if FL_args.modelA_name_or_path:

            torch_dtype = (
                FL_args.torch_dtype
                if FL_args.torch_dtype in ["auto", None]
                else getattr(torch, FL_args.torch_dtype)
            )
            logger.info(f"modelA torch_dtype:{torch_dtype}")

            model_A = LlamaForCausalLM(config=configA, modelA_layers=FL_args.modelA_layers).to("cpu")
            clientA_state_dict = torch.load(f'{FL_args.modelA_name_or_path}/pytorch_model_A.bin', map_location=FL_args.device)
            adjusted_A_dict = adjust_modelA_state_dict(clientA_state_dict, FL_args.modelA_layers)
            model_A.load_state_dict(adjusted_A_dict)
            del adjusted_A_dict
            del clientA_state_dict
            model_A = model_A.to(device)

        else:
            model_A = LlamaForCausalLM.from_config(configA)
            n_params = sum({p.data_ptr(): p.numel() for p in model_A.parameters()}.values())
            logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

        # model_C config and initial
        if FL_args.modelC_name_or_path:

            torch_dtype = (
                FL_args.torch_dtype
                if FL_args.torch_dtype in ["auto", None]
                else getattr(torch, FL_args.torch_dtype)
            )
            logger.info(f"modelC torch_dtype:{torch_dtype}")

            model_C = LlamaForCausalLMC(config=configC, modelC_layers=FL_args.modelC_layers).to("cpu")
            clientC_state_dict = torch.load(f'{FL_args.modelC_name_or_path}/pytorch_model_C.bin', map_location=FL_args.device)
            adjusted_C_dict = adjust_modelC_state_dict(clientC_state_dict, FL_args.modelC_layers)
            model_C.load_state_dict(adjusted_C_dict)
            del adjusted_C_dict
            del clientC_state_dict
            model_C = model_C.to(device)
            torch.cuda.set_device(FL_args.device)
            torch.cuda.empty_cache()

            
        else:
            model_C = LlamaForCausalLM.from_config(configC)
            n_params = sum({p.data_ptr(): p.numel() for p in model_C.parameters()}.values())
            logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

        torch.cuda.empty_cache()
        if FL_args.Attack and FL_args.Attack_load_lora:
            logger.warning("加载lora检查点使得客户端1成为不同的模型!")

            match = re.search(r"-(\d+)$", FL_args.Attack_lora_checkpoint)
            if match:
                number_str = match.group(1)
                checkpoint_number = int(number_str)
                logger.info(f"匹配到的数字为: {checkpoint_number}")
            else:
                logger.warning("未匹配到数字")

            if checkpoint_number ==500:
                A_lora_path="/home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module-autosplit/output/Safety/version_3_Lr2e-5_Alayers5_Blayers5/model-A"
                logger.info(f"A_lora_path:{A_lora_path}")
            elif checkpoint_number ==4000:
                A_lora_path="/home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module-autosplit/output/Safety/version_1_Lr2e-5_Alayers5_Blayers5/model-A"
                logger.info(f"A_lora_path:{A_lora_path}")
            else:
                logger.warning("请检查A_lora_path路径!")

            # C_lora_path="/home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module-autosplit/output/Safety/version_1_Lr2e-5_Alayers5_Blayers5/model-C"
            attack_lora_path_A = os.path.join(A_lora_path, FL_args.Attack_lora_checkpoint)
            # attack_lora_path_C = os.path.join(C_lora_path, FL_args.Attack_lora_checkpoint)
            # 加载lora检查点
            model_A = PeftModel.from_pretrained(model_A, attack_lora_path_A)
            model_A = model_A.merge_and_unload()
            # model_C = PeftModel.from_pretrained(model_C, attack_lora_path_C)
            # model_C = model_C.merge_and_unload()
            if hasattr(model_A, 'peft_config'):
                del model_A.peft_config
            # if hasattr(model_C, 'peft_config'):
            #     del model_C.peft_config
         
            logger.info("检查点加载成功: model-A")
        elif FL_args.Attack and not FL_args.Attack_load_lora:
            logger.warning("客户端1和客户端0初始相同!")
        else:
            logger.info("客户端1不会发送数据当做攻击的测试集")

        # 重新加载lora模型
        model_A = get_peft_model(model_A, lora_configA)
        # model_A.print_trainable_parameters()
        model_C = get_peft_model(model_C, lora_configC)
        # model_C.print_trainable_parameters()

        # logger.info(f"下面是modelA的参数")
        # logger.info("\n")  
        # for name, param in model_A.named_parameters():
        #     if param.requires_grad:
        #         logger.info(f"Parameter Name: {name}, value: {param.data}, Requires Grad: {param.requires_grad}") 
        # logger.info(f"下面是modelC的参数")
        # logger.info("\n")  
        # for name, param in model_C.named_parameters():
        #     if param.requires_grad:
        #         logger.info(f"Parameter Name: {name}, value: {param.data}, Requires Grad: {param.requires_grad}") 

        if FL_args.custom_lora:
            logger.warning(f"Initial lora weights from bin")
            logger.warning(f"You need to check lora r and lora alpha")

            Lora_A_state_dict = torch.load("/home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module-autosplit/lora_r8_alpha16_weights/lora_A/adapter_model.bin")
            adjusted_LoraA_dict = adjust_modelA_Lora_state_dict(Lora_A_state_dict, FL_args.modelA_layers)
            model_A.load_state_dict(adjusted_LoraA_dict, strict=False)
            del Lora_A_state_dict
            del adjusted_LoraA_dict
            Lora_C_state_dict = torch.load("/home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module-autosplit/lora_r8_alpha16_weights/lora_C/adapter_model.bin")
            adjusted_LoraC_dict = adjust_modelC_Lora_state_dict(Lora_C_state_dict, FL_args.modelC_layers)
            model_C.load_state_dict(adjusted_LoraC_dict, strict=False)
            del Lora_C_state_dict
            del adjusted_LoraC_dict

            torch.cuda.empty_cache()
        else:
            logger.warning(f"Initial lora weights from automation")

        # logger.info(f"下面是修改后modelA的参数")
        # logger.info("\n")  
        # for name, param in model_A.named_parameters():
        #     if param.requires_grad:
        #         logger.info(f"Parameter Name: {name}, value: {param.data}, Requires Grad: {param.requires_grad}")  
        # logger.info(f"下面是修改后modelC的参数")
        # logger.info("\n")  
        # for name, param in model_C.named_parameters():
        #     if param.requires_grad:
        #         logger.info(f"Parameter Name: {name}, value: {param.data}, Requires Grad: {param.requires_grad}") 

        # 如果model_A的数据类型与目标数据类型不同，则进行转换
        if model_A.dtype != model_torch_dtype:
            logger.info(f"Converting model_A from {model_A.dtype} to {model_torch_dtype}")
            model_A = model_A.to(model_torch_dtype)

        # 如果model_C的数据类型与目标数据类型不同，则进行转换
        if model_C.dtype != model_torch_dtype:
            logger.info(f"Converting model_C from {model_C.dtype} to {model_torch_dtype}")
            model_C = model_C.to(model_torch_dtype)


        # optimizer初始化
        optimA = AdamW(filter(lambda p: p.requires_grad, model_A.parameters()), lr=FL_args.lr, betas=FL_args.betas, eps=FL_args.eps, weight_decay=FL_args.weight_decay)
        optimC = AdamW(filter(lambda p: p.requires_grad, model_C.parameters()), lr=FL_args.lr, betas=FL_args.betas, eps=FL_args.eps, weight_decay=FL_args.weight_decay)
        # schedule 初始化
        scheduleA = LinearLR(optimA, start_factor=1.0, end_factor=0.0, total_iters=FL_args.max_step)
        scheduleC = LinearLR(optimC, start_factor=1.0, end_factor=0.0, total_iters=FL_args.max_step)
        schedule = [scheduleA,scheduleC]
        optimizer = [optimA, optimC]

        # #最简单的版本
        # # 改用 SGD 优化器，并保持学习率始终不变
        # optimA = torch.optim.SGD(
        #     filter(lambda p: p.requires_grad, model_A.parameters()),
        #     lr=FL_args.lr
        # )
        # optimC = torch.optim.SGD(
        #     filter(lambda p: p.requires_grad, model_C.parameters()),
        #     lr=FL_args.lr
        # )

        # # 恒定学习率调度器（实际不改变学习率）
        # from torch.optim.lr_scheduler import LambdaLR
        # scheduleA = LambdaLR(optimA, lr_lambda=lambda epoch: 1.0)  # lambda 始终返回1.0
        # scheduleC = LambdaLR(optimC, lr_lambda=lambda epoch: 1.0)
        # schedule = [scheduleA, scheduleC]
        # optimizer = [optimA, optimC]


    else: # 测试的时候加载lora模型
        set_seed(FL_args.seed)
        config_kwargs = {
            "cache_dir": FL_args.cache_dir,
            "revision": FL_args.model_revision,
            "use_auth_token": True if FL_args.use_auth_token else None,
        }
        if FL_args.modelA_config_name and FL_args.modelC_config_name:   # 从这里进入
            configA = AutoConfig.from_pretrained(FL_args.modelA_config_name, **config_kwargs)
            configC = AutoConfig.from_pretrained(FL_args.modelC_config_name, **config_kwargs)
        # 加载基座模型A
        if FL_args.modelA_name_or_path:
            torch_dtype = (
                FL_args.torch_dtype
                if FL_args.torch_dtype in ["auto", None]
                else getattr(torch, FL_args.torch_dtype)
            )
            model_A = LlamaForCausalLM(config=configA, modelA_layers=FL_args.modelA_layers).to('cpu')
            clientA_state_dict = torch.load(f'{FL_args.modelA_name_or_path}/pytorch_model_A.bin', map_location=FL_args.device)
            #根据modelA_layers动态加载模型
            adjusted_A_dict = adjust_modelA_state_dict(clientA_state_dict, FL_args.modelA_layers)
            model_A.load_state_dict(adjusted_A_dict)
            del clientA_state_dict
            del adjusted_A_dict
            model_A = model_A.to(device)
        # 加载基座模型C
        if FL_args.modelC_name_or_path:
            torch_dtype = (
                FL_args.torch_dtype
                if FL_args.torch_dtype in ["auto", None]
                else getattr(torch, FL_args.torch_dtype)
            )
            model_C = LlamaForCausalLMC(config=configC, modelC_layers=FL_args.modelC_layers).to('cpu')
            clientC_state_dict = torch.load(f'{FL_args.modelC_name_or_path}/pytorch_model_C.bin', map_location=FL_args.device)
            adjusted_C_dict = adjust_modelC_state_dict(clientC_state_dict, FL_args.modelC_layers)
            model_C.load_state_dict(adjusted_C_dict)
            del clientC_state_dict
            del adjusted_C_dict
            model_C = model_C.to(device)
            torch.cuda.set_device(FL_args.device)
            torch.cuda.empty_cache()

        if FL_args.AggregatedClients:
            logger.info("正在聚合两个客户端的模型A和模型C")
            if FL_args.lora_modelA_path is not None or FL_args.lora_modelC_path is not None:
                raise ValueError("当 FL_args.AggregatedClients 为 True 时, lora_modelA_path 和 lora_modelC_path 应为 None")
            logger.info(f"正加载{FL_args.cleint0_lora_modelA_path},{FL_args.cleint0_lora_modelC_path},{FL_args.cleint1_lora_modelA_path}和{FL_args.cleint1_lora_modelC_path}的checkpoint")

            aggregate_start = time.time()
            if FL_args.AggregatedAvg:
                model_A_cleint0 = PeftModel.from_pretrained(model_A, FL_args.cleint0_lora_modelA_path, device_map=FL_args.device,trust_remote_code=True)
                model_A_cleint0 = model_A_cleint0.merge_and_unload()
                model_C_cleint0 = PeftModel.from_pretrained(model_C, FL_args.cleint0_lora_modelC_path, device_map=FL_args.device,trust_remote_code=True)
                model_C_cleint0 = model_C_cleint0.merge_and_unload()
                model_A_cleint1 = PeftModel.from_pretrained(model_A, FL_args.cleint1_lora_modelA_path, device_map=FL_args.device,trust_remote_code=True)
                model_A_cleint1 = model_A_cleint1.merge_and_unload()
                model_C_cleint1 = PeftModel.from_pretrained(model_C, FL_args.cleint1_lora_modelC_path, device_map=FL_args.device,trust_remote_code=True)
                model_C_cleint1 = model_C_cleint1.merge_and_unload()
                model_A = aggregate_models(model_A_cleint0, model_A_cleint1).to(FL_args.device)
                model_C = aggregate_models(model_C_cleint0, model_C_cleint1).to(FL_args.device)
                logger.info("聚合方式为AggregatedAvg")
            elif FL_args.AggregatedIter: 
                model_A_cleint0 = PeftModel.from_pretrained(model_A, FL_args.cleint0_lora_modelA_path, device_map=FL_args.device,trust_remote_code=True)
                model_A_cleint0 = model_A_cleint0.merge_and_unload()
                model_A_cleint1 = PeftModel.from_pretrained(model_A_cleint0, FL_args.cleint1_lora_modelA_path, device_map=FL_args.device,trust_remote_code=True)
                model_A = model_A_cleint1.merge_and_unload()
                model_C_cleint0 = PeftModel.from_pretrained(model_C, FL_args.cleint0_lora_modelC_path, device_map=FL_args.device,trust_remote_code=True)
                model_C_cleint0 = model_C_cleint0.merge_and_unload()
                model_C_cleint1 = PeftModel.from_pretrained(model_C_cleint0, FL_args.cleint1_lora_modelC_path, device_map=FL_args.device,trust_remote_code=True)
                model_C = model_C_cleint1.merge_and_unload()
                logger.info("聚合方式为AggregatedIter")
            else:
                raise ValueError("请指定聚合方式!")
            aggregate_end=time.time()
            aggregate_cost=aggregate_end-aggregate_start
            logger.info(f"聚合两个客户端的模型A和模型C完成! 耗时:{aggregate_cost:.2f}秒")
            # 删除冗余显存
            del model_A_cleint0
            del model_A_cleint1
            del model_C_cleint0
            del model_C_cleint1
            
        elif FL_args.lora_modelA_path and FL_args.lora_modelC_path:
            model_A = PeftModel.from_pretrained(model_A, FL_args.lora_modelA_path, device_map=FL_args.device,trust_remote_code=True)
            model_A = model_A.merge_and_unload()
            model_C = PeftModel.from_pretrained(model_C, FL_args.lora_modelC_path, device_map=FL_args.device,trust_remote_code=True)
            model_C = model_C.merge_and_unload()
            logger.warning(f"正加载{FL_args.lora_modelA_path}和{FL_args.lora_modelC_path}的checkpoint")
        else:
            logger.warning("没有加载任何checkpoint!")

        # 如果model_A的数据类型与目标数据类型不同，则进行转换
        if model_A.dtype != model_torch_dtype:
            logger.info(f"Converting model_A from {model_A.dtype} to {model_torch_dtype}")
            model_A = model_A.to(model_torch_dtype)
        # 如果model_C的数据类型与目标数据类型不同，则进行转换
        if model_C.dtype != model_torch_dtype:
            logger.info(f"Converting model_C from {model_C.dtype} to {model_torch_dtype}")
            model_C = model_C.to(model_torch_dtype)

        tokenizer_kwargs = {
            "cache_dir": FL_args.cache_dir,
            "use_fast": FL_args.use_fast_tokenizer,
            "revision": FL_args.model_revision,
            "use_auth_token": True if FL_args.use_auth_token else None,
            "padding_side":'left'
        }
        if FL_args.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(FL_args.tokenizer_name, **tokenizer_kwargs)
        elif FL_args.modelB_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(FL_args.modelB_name_or_path, **tokenizer_kwargs)
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
                "You can do it from another script, save it, and load it from here, using --tokenizer_name."
            )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = 3
        # optimizer初始化
        schedule = None
        # schedule 初始化
        optimizer = None

    models = [model_A, model_C]

    end_set_model = time.time()
    model_time_cost = end_set_model - start_set_model
    logger.info( f"client加载模型耗时{model_time_cost:.2f} 秒。")

    if FL_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 2048:
            block_size = 2048
    else:
        block_size = min(FL_args.block_size, tokenizer.model_max_length)

    #加载数据集
    if not FL_args.from_pickle:
        # if FL_args.do_BatchTrain: #BatchTarin的时候要进行pad，否则不用
        #     datalist = get_dataset(FL_args, tokenizer)
        #     logger.info("正在进行do_BatchTrain,需要进行pad")
        # else: 
        #     datalist = get_dataset_no_pad(FL_args, tokenizer)
        #     logger.info("不需要进行pad")
        datalist = get_dataset_no_pad(FL_args, tokenizer)


        # 保存dataset方便调试
        if FL_args.data2pic:
            logger.info(f"正在保存pickle数据...")
            if FL_args.do_train:
                stored_path = f"/home/zhangzishuai/SplitFederated-LLaMA/pickle-store/Distributed_{FL_args.dataset_name}_train_bs{FL_args.batch_size}_step{FL_args.max_step}.pickle"
            if FL_args.do_predict:
                if FL_args.iter_test:
                    stored_path = f"/home/zhangzishuai/SplitFederated-LLaMA/pickle-store/Distributed_{FL_args.dataset_name}_IterTest.pickle"
                else:
                    stored_path = f"/home/zhangzishuai/SplitFederated-LLaMA/pickle-store/Distributed_{FL_args.dataset_name}_FullTest.pickle"
            with open(stored_path, 'wb') as f:
                pickle.dump(datalist, f)
                logger.info(f"已将pickle数据保存到{stored_path}")
    else:
        logger.info(f"从pickle加载数据...")
        if FL_args.dataset_name == "CoQA":
            if FL_args.do_train:
                stored_path = f"/home/zhangzishuai/SplitFederated-LLaMA/pickle-store/CoQA_Training_50000.pickle"
            if FL_args.do_predict:
                stored_path = f"/home/zhangzishuai/SplitFederated-LLaMA/pickle-store/CoQA_TestDataset.pickle"
        if FL_args.dataset_name == "Xsum":
            if FL_args.do_train:
                if FL_args.Attack:
                    stored_path = "/home/zhangzishuai/SplitFederated-LLaMA/pickle-store/Xsum_bs2_step10000_central_train_client1.pickle"
                else:
                    stored_path=f"/home/zhangzishuai/SplitFederated-LLaMA/Models/central_model/finetune_code/pickle-store/Xsum_bs{FL_args.batch_size}_step{FL_args.max_step}_central_train.pickle"
            if FL_args.do_predict:
                if FL_args.iter_test:
                    stored_path="/home/zhangzishuai/SplitFederated-LLaMA/Models/central_model/finetune_code/Xsum/Xsum_central_IterTest.pickle"
                else:
                    stored_path="/home/zhangzishuai/SplitFederated-LLaMA/Models/central_model/finetune_code/Xsum/Xsum_central_FullTest.pickle"
        if FL_args.dataset_name == "MultiRC":
            if FL_args.do_train:
                if FL_args.Attack:
                    stored_path = f"/home/zhangzishuai/SplitFederated-LLaMA/pickle-store/MultiRC_bs{FL_args.batch_size}_central_train_client1.pickle"
                else:
                    stored_path = f"/home/zhangzishuai/SplitFederated-LLaMA/pickle-store/MultiRC_bs{FL_args.batch_size}_central_train.pickle"
            if FL_args.do_predict:
                if FL_args.iter_test:
                    stored_path = "/home/zhangzishuai/SplitFederated-LLaMA/pickle-store/MultiRC_central_IterTest.pickle"
                else:
                    stored_path = "/home/zhangzishuai/SplitFederated-LLaMA/pickle-store/MultiRC_central_FullTest.pickle"
        if FL_args.dataset_name == "Record":
            if FL_args.do_train:
                # 从中心化哪里来的数据
                # stored_path = f"/home/zhangzishuai/SplitFederated-LLaMA/Models/central_model/finetune_code/pickle-store/Record_bs{FL_args.batch_size}_step{FL_args.max_step}_central_train.pickle"
                stored_path = f"/home/zhangzishuai/SplitFederated-LLaMA/pickle-store/Record_bs2_step20000_central_train_client1.pickle"
            if FL_args.do_predict:
                if FL_args.iter_test:
                    stored_path = "/home/zhangzishuai/SplitFederated-LLaMA/Models/central_model/finetune_code/Record/Record_Central_IterTest.pickle"
                else:
                    stored_path = "/home/zhangzishuai/SplitFederated-LLaMA/pickle-store/Record_central_FullTest.pickle"
        with open(stored_path, 'rb') as f:
            datalist = pickle.load(f)
            logger.info(f"从{stored_path}加载pickle数据完成")

    if FL_args.do_train:
        FL_args.status = 'train'
    if FL_args.do_BatchTrain:
        FL_args.status = 'batch_train'
    if FL_args.do_eval:
        FL_args.status = 'eval'
    if FL_args.do_predict and not args.do_inference:
        FL_args.status = 'predict'
    if FL_args.do_inference:
        FL_args.status = 'inference'

    logger.info(f"len_datalist:{len(datalist)},len代表有多少个epoch")
    logger.info(f"client_args:{FL_args}")
    logger.info(f"正在进行的任务:{FL_args.status}")

    torch.cuda.empty_cache()
    # Start Flower client
    fl_client = Fed_Client(datalist, optimizer, schedule, models, tokenizer, FL_args, child_conn).to_client()
    fl.client.start_client(server_address=FL_args.server_ip, client=fl_client)

if __name__ == "__main__":
    args = FLparser()
    '''=====================================================设置日志====================================================='''
    # 设置日志文件路径
    if (args.add_DP_gradient or args.add_DP_hidden ) and not args.Attack:
        log_path = os.path.join(args.output_dir, f'DP_no_attack_client_1_noise{args.grad_noise}_Alayers{args.modelA_layers}_Clayers{args.modelC_layers}.log')
    elif (args.add_DP_gradient or args.add_DP_hidden ) and args.Attack:
        log_path = os.path.join(args.output_dir, f'Attack_with_DP_client_1_noise{args.grad_noise}_Alayers{args.modelA_layers}_Clayers{args.modelC_layers}.log')
    elif not (args.add_DP_gradient or args.add_DP_hidden ) and args.Attack:
        log_path = os.path.join(args.output_dir, f'Attack_no_DP_client_1_Alayers{args.modelA_layers}_Clayers{args.modelC_layers}.log')
    else:
        log_path = os.path.join(args.output_dir, f'client_1_Alayers{args.modelA_layers}_Clayers{args.modelC_layers}.log')
    logging.basicConfig(filename=log_path,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                        level=logging.INFO,
                        filemode='w')

    # create a logger object
    logger = logging.getLogger(__name__)

    # wandb初始化
    if args.wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project=f"DP_Distributed_6_common_lora",
            name=f"noise{args.grad_noise}_Alayers{args.modelA_layers}_Clayers{args.modelC_layers}",
            # track hyperparameters and run metadata
            config={
            "learning_rate": args.lr,
            "architecture": "Llama",
            "dataset": "MultiRC",
            "total_steps": args.max_step,
            }
        )

    if args.do_train:
        logger.info("正在进行客户端训练")
        assert not (args.do_eval or args.do_predict or args.do_inference), "当 do_train 为 True 时，其它标志必须为 False"
        main()
    if args.do_BatchTrain:
        logger.info("正在进行客户端批量训练")
        assert not (args.do_eval or args.do_predict or args.do_inference), "当 do_BatchTrain 为 True 时，其它标志必须为 False"
    if args.do_eval:
        logger.info("正在进行客户端验证")
        assert not (args.do_train or args.do_predict or args.do_inference), "当 do_eval 为 True 时，其它标志必须为 False"
        main()
    if args.do_predict and not args.do_inference:
        logger.info("正在进行客户端测试")
        assert not (args.do_train or args.do_eval or args.do_inference), "当 do_predict 为 True 时，其它标志必须为 False"
        main()
    if args.do_inference:
        logger.info("正在进行客户端推理")
        args.batch_size = 1
        logger.info(f"batch_size强制设置为{args.batch_size}")
        assert not (args.do_train or args.do_eval), "当 do_inference 为 True 时, do_train, do_BatchTrain, do_eval 必须为 False"

        if args.do_netdisplay:
            # 以下是为了网页展示推理
            args = FLparser()
            prompt = args.prompt  #接受命令行参数
            pipe_fd = int(args.pipe)
            child_conn = connection.Connection(pipe_fd)
            main(str(prompt), child_conn)
            child_conn.close()  # 关闭子管道
        else:
            B_INST, E_INST = "[INST]", "[/INST]"
            B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
            raw_prompt = "what can you do?"
            TemplatePrompt=f"{B_INST}{B_SYS}System: You are a helpful assistant. Always answer as helpfully as possible.{E_SYS}User: {raw_prompt}{E_INST}"
            # 不加载checkpoint的prompt格式,中心化加<s>，这里不用因为dataloader自动加了
            # TemplatePrompt="能帮我测试ChatGLM和FedLLAMA传输数据大小吗"
            main(TemplatePrompt)