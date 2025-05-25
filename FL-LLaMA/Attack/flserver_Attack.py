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
"""Flower server."""
"""client 2 part, server 1 part """
#region 导入包
import argparse
import json
import sys 
import os
import torch
import torch.nn as nn
from transformers.optimization import AdamW
from torch.optim.lr_scheduler import LinearLR, StepLR
import collections
import flwr as fl
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('./server') # 相对导入server文件夹
# 为了比较超长的句子的rouge
sys.setrecursionlimit(1430 * 1430 + 10)
from server.modeling_llama_B import LlamaForCausalLMB
import concurrent.futures
import timeit
from collections import OrderedDict
from logging import DEBUG, INFO
import logging
from typing import Dict, List, Optional, Tuple, Union
import yagmail #邮箱提醒
from datetime import datetime
import traceback
from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    ReconnectIns,
    Scalar,
)
from flwr.common.logger import log
from flwr.common.typing import GetParametersIns
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import FedAvg, Strategy
from flwr.common import FitRes, NDArray, NDArrays, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server import Server
from sfglm_strategy import SplitFed
# from  sfglm_strategy import SplitFed
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
from transformers.cache_utils import Cache, DynamicCache, StaticCache
import safetensors.torch
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
import wandb
import pickle
from multiprocessing import connection
import random
import numpy as np
#重建攻击使用官方的
from modeling_llama_official import LlamaForCausalLM
#重建攻击评估效果
# from rouge_chinese import Rouge
from rouge import Rouge 
import rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import jieba


# 设置CUBLAS_WORKSPACE_CONFIG环境变量
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
#endregion

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# FitResultsAndFailures = Tuple[
#     List[Tuple[ClientProxy, FitRes]],
#     List[Union[Tuple[ClientProxy, FitRes], BaseException]],
# ]
# EvaluateResultsAndFailures = Tuple[
#     List[Tuple[ClientProxy, EvaluateRes]],
#     List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
# ]
# ReconnectResultsAndFailures = Tuple[
#     List[Tuple[ClientProxy, DisconnectRes]],
#     List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]],
# ]

def FLparser(): # use glm arguments with argument.py
    parser = argparse.ArgumentParser(description="Flower")
    '''=====================================================model args====================================================='''                    
    # 
    parser.add_argument("--modelB_name_or_path", type=str, default="./server",
                        help="The model checkpoint for weights initialization. Don't set if you want to train a model from scratch.")

    parser.add_argument("--model_type", type=str, default=None,
                        help="If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES))
    parser.add_argument("--config_overrides", type=str, default=None,
                        help="Override some existing default config settings when a model is trained from scratch. Example: n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index")
    parser.add_argument("--modelB_config_name", type=str, default=None,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", type=str, default="/home/zhangzishuai/SplitFederated-LLaMA/Models/Llama2-7B-chat-service",
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Where do you want to store the pretrained models downloaded from huggingface.co")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--target_modules", type=str, default='q_proj,v_proj,k_proj,o_proj',
                        help="List of module names or regex expression of the module names to replace with Lora. For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'")
    parser.add_argument("--use_fast_tokenizer", default=False,
                        help="Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.")
    parser.add_argument("--model_revision", type=str, default="main",
                        help="The specific model version to use (can be a branch name, tag name or commit id).")
    parser.add_argument("--use_auth_token", default=False,
                        help="Will use the token generated when running `huggingface-cli login` (necessary to use this script with private models).")
    parser.add_argument("--torch_dtype", type=str, default="float16",
                        choices=["auto", "bfloat16", "float16", "float32"],
                        help="Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the dtype will be automatically derived from the model's weights.")
    # parser.add_argument("--lora_modelB_path", type=str, default="./output/Record/version_3_r8_al32_Lr0.00003/model-B/checkpoint-33000", help="The pretrained model checkpoint for lora weights.")
    parser.add_argument("--lora_modelB_path", type=str, default="./output/AttackExperiments/Version_0/model-B/checkpoint-8000", help="The pretrained model checkpoint for lora weights.")
    parser.add_argument("--quantization_bit", type=int, help="quantization bit",  default=4)

    '''=====================================================training args====================================================='''  
    parser.add_argument("--batch_size", type=int, help="traing batch size", default=2) 
    parser.add_argument("--max_train_samples", type=int, default=None)
    # 这两个参数会影响batch时pad的长度，因此要想完全复刻某一实验，需要保持相同
    parser.add_argument("--max_source_length", type=int, default=700)
    parser.add_argument("--max_target_length", type=int, default=20)
    parser.add_argument("--max_eval_samples", type=int, default=None)  
    parser.add_argument("--seed", type=int, help="random seed", default=42)
    parser.add_argument("--device",type=str, default = 'cuda:0')
    parser.add_argument("--max_grad_norm", type=float, help='max grad_clipping norm', default=1.0)
    parser.add_argument("--lr", type=float, help='learning rate', default=1e-5)
    parser.add_argument("--betas", type=tuple, help='(adamhf)optimizer betas', default=(0.9,0.999))
    parser.add_argument("--eps", type=float, help='(adamhf)optimizer eps', default=1e-5)
    parser.add_argument("--weight_decay", type=float, help='(adamhf)optimizer weight decay', default=0.01)
    parser.add_argument("--output_dir", type=str, help = 'output folder path', default="./output/BatchTrain")
    parser.add_argument("--save_step", type=int, help = 'step to save the prefix encoder', default=1000)
    parser.add_argument("--overwrite_output_dir", type=bool, help = 'whether to overwrite the output folder', default=False)
    parser.add_argument("--block_size", type=int, help = 'I dont know yet', default=None)
    parser.add_argument("--max_step", type=int, help='number of max training steps, should be same with server side!', default=20000)
    parser.add_argument("--mixed_precision", action="store_true", help='Whether to use mixed precision instead of full precision', default=False)
    parser.add_argument("--custom_lora", action="store_true", help='Whether to customize lora weights.', default=True)
    '''=====================================================Fl Arguments====================================================='''  
    parser.add_argument("--server_ip", type=str, help='ip4_address:port.', default="10.143.12.74:7030")
    parser.add_argument("--prompt", type=str, default=None) 
    parser.add_argument("--pipe", type=str, default=None)    
    parser.add_argument("--max_predict_samples", type=int, default=1000)
    parser.add_argument("--val_max_target_length", type=int, default=None)
    parser.add_argument("--do_train", action="store_true", help='Whether to run training.', default=False)
    parser.add_argument("--do_BatchParallel_train", action="store_true", help='Whether to run BatchParallel training.', default=False)
    parser.add_argument("--do_eval", action="store_true", help='Whether to run eval on the dev set.', default=False)
    parser.add_argument("--do_predict", action="store_true", help='Whether to run predictions on the test set.', default=True)
    parser.add_argument("--do_inference", action="store_true", help='Whether to inference using prompt from users.', default=False)
    parser.add_argument("--do_netdisplay", action="store_true", help='Whether to Netdisplay.', default=False)
    parser.add_argument("--dry", type=bool, default=False, required=False, help="Do a dry-run to check the client")
    parser.add_argument("--client_id", type=int, default=1, choices=range(0, 10), required=False, help="Specifies the artificial data partition of CIFAR10 to be used. \
        Picks partition 0 by default")
    parser.add_argument(  "--toy", action="store_true", help="Set to true to quicky run the client using only 10 datasamples. \
        Useful for testing purposes. Default: False")
    parser.add_argument("--use_cuda", type=bool, default=True, required=False, help="Set to true to use GPU. Default: False")
    parser.add_argument("--model", type=str, default="efficientnet", choices=["efficientnet", "alexnet"],
        help="Use either Efficientnet or Alexnet models. If you want to achieve differential privacy, please use the Alexnet model")
    parser.add_argument("--client_count", type=int, help='number of clients', default=1)
    parser.add_argument("--per_client_steps", type=int, help="the number of the traing steps of every client", default=1)
    parser.add_argument("--max_output_length", type=int, help='max_output_length.', default=10)
    parser.add_argument("--pred_finished", type=bool, help='pred_finished.', default=False)
    parser.add_argument("--start_inference", type=str, help='display answer when get prompt.', default="not OK")

    '''=====================================================Split Arguments====================================================='''  
    parser.add_argument("--modelA_layers", type=int, choices=[1,2,3,4,5], default=1 )
    parser.add_argument("--modelC_layers", type=int, choices=[1,2,3,4,5], default=1 )
    '''=====================================================DP_Arguments====================================================='''  
    parser.add_argument("--add_DP_gradient", action="store_true", help='Whether to add DP on gradients.', default=False)
    # parser.add_argument("--add_DP_hidden", action="store_true", help='Whether to add DP on hidden states.', default=False)
    parser.add_argument("--add_DP_hidden", help='Whether to add DP on hidden states.', default=False)
    parser.add_argument("--DP_grad_norm_clip", type=float, help='hidden_grad_norm', default=0.04)
    parser.add_argument("--grad_noise", type=float, help='grad_noise', default=0.02)
    '''=====================================================Attack_Arguments====================================================='''
    parser.add_argument("--Attack", action="store_true", help='Whether to attack to reconstruct dataset.', default=True)
    parser.add_argument("--Attack_Lora", action="store_true", help='Whether to use lora to reconstruct dataset.', default=False)
    parser.add_argument("--Attack_model_path", type=str, default="/home/zhangzishuai/SplitFederated-LLaMA/Models/central_model/model_path")
    parser.add_argument("--Attack_decoder_path", type=str, default="./output/Attack_decoder_checkpoint/version_0_step-8000.pth")
    parser.add_argument("--Attack_lora_r", type=int, default=8)
    parser.add_argument("--Attack_lora_alpha", type=int, default=16)
    parser.add_argument("--Attack_layers", type=int, default=1)
    parser.add_argument("--Attack_lr", type=float, help='learning rate', default=2e-5)
    parser.add_argument("--Attack_load_decoder", help='Whether to attack to reconstruct dataset.', default=True)
    parser.add_argument("--Attack_train", help='Whether to attack to reconstruct dataset.', default=False)
    parser.add_argument("--Attack_random_initial", help='Whether to attack to reconstruct dataset.', default=False)
    parser.add_argument("--Attack_wandb", action="store_true", help='Whether to attack to reconstruct dataset.', default=False)
    parser.add_argument("--Attack_predict_next", action="store_true", help='Whether to attack to reconstruct dataset.', default=True)
    parser.add_argument("--Attack_no_mask", action="store_true", help='Whether to attack to reconstruct dataset.', default=False)
    # truncation在训练攻击的时候为True ，测试的时候为False
    parser.add_argument("--Attack_truncation", action="store_true", help='Whether to attack to reconstruct dataset.', default=False)
    parser.add_argument("--Attack_lora_checkpoint", type=str, help="the number of the traing steps of every client", default="checkpoint-4000")
    parser.add_argument("--Client1_VariedEmbeding", help='Whether to attack to reconstruct dataset.', default=False)

    args = parser.parse_args()

    return args

class FL_Server(Server):
    """Flower server for Split fed learning."""
    def __init__(self, optimizer,schedule,ServerModel, args, logger, client_manager: ClientManager, strategy: Strategy = None, child_conn_server=None,attack_decoder_model=None,attack_optimizer=None,attack_tokenizer=None) -> None:
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.servermodel = ServerModel
        self.logger = logger
        self.logger.info(f"modelB_dtype:{self.servermodel.dtype}")
        self.logger.warning(f"modelA_layers:{args.modelA_layers}, modelC_layers:{args.modelC_layers},modelB_layers:{32-args.modelA_layers-args.modelC_layers}")
        self.optim = optimizer
        self.schedule = schedule
        self.model_args = args
        if self.model_args.do_inference or self.model_args.do_predict:
            self.model_args.batch_size = 1
            logger.info(f"batch_size强制设置为{self.model_args.batch_size}")
        self.child_conn_server = child_conn_server
        self.torch_dtype = self.init_torch_dtype()
        global self_device
        self.device = torch.device(self_device)
        self.causal_mask = self.init_casual_mask() #训练和测试都需要用到，本地生成attn_mask

        if self.model_args.do_train or self.model_args.do_BatchParallel_train:
            self.servermodel.train()
            self.train_attn_mask = None
        else:
            self.servermodel.eval()
            self.pkv_modelB = DynamicCache()
            for _ in range(self.model_args.modelA_layers):
                self.pkv_modelB.key_cache.append(torch.tensor(0, dtype=torch.int8, device=self.model_args.device))
                self.pkv_modelB.value_cache.append(torch.tensor(0, dtype=torch.int8, device=self.model_args.device))
            self.cache_position = None
            self.predict_attn_mask = None

        self.attack_decoder_model=attack_decoder_model
        self.attack_optimizer=attack_optimizer
        self.attack_tokenizer=attack_tokenizer
        if self.attack_decoder_model is not None:
            logger.warning("正在实施重建数据攻击")
            logger.info(f"下面是self.attack_decoder_model的参数")
            logger.info("\n")  
            for name, param in self.attack_decoder_model.named_parameters():
                logger.info(f"Parameter Name: {name}, Requires Grad: {param.requires_grad}") 
        else:
            logger.info("没有实施攻击")
        # for wandb 画图    
        self.global_step = 0
        self.rouge_2=[]
        self.bleu=[]

    def train_server(self, ServerModel, position_ids, attention_mask, hidden_state):

        feature2 = ServerModel(position_ids = position_ids,
            attention_mask = attention_mask,
            hidden_state = hidden_state) # dict
        
        return feature2

    def predict_server(self, ServerModel, attention_mask, hidden_state, past_key_values, cache_position):

        feature2 = ServerModel(attention_mask = attention_mask,
            hidden_state = hidden_state,            
            past_key_values = past_key_values,
            cache_position = cache_position)
        
        return feature2
    
    def init_torch_dtype(self):
        dtype_mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "int32": torch.int32,
        }
        # 检查FL_args.torch_dtype是否有效
        if self.model_args.torch_dtype not in dtype_mapping:
            raise ValueError(f"Unsupported data type: {self.model_args.torch_dtype}. Supported types: {list(dtype_mapping.keys())}")
        # 获取对应的数据类型
        dtype = dtype_mapping[self.model_args.torch_dtype]
        return dtype

    def init_casual_mask(self):
        causal_mask = torch.full((4096, 4096), fill_value=1)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask.to(self.torch_dtype)   # torch.finfo(dtype)需要浮点数输入
        causal_mask = (
            causal_mask[None, None, :, :].repeat(self.model_args.batch_size, 1, 1, 1).to(self.torch_dtype) * torch.finfo(self.torch_dtype).min
        )
        causal_mask = causal_mask.to(dtype=self.torch_dtype, device=self.model_args.device)

        self.logger.info("初始化casual_mask完成")
        return causal_mask
        
    def update_causal_mask(self, attention_mask,position_ids):
        causal_mask=self.causal_mask.detach().clone()
        #attention_mask [1,102]
        dtype = self.torch_dtype

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

        if self.model_args.Attack and self.model_args.Attack_truncation:
            self.logger.warning("update_causal_mask中对attention_mask做截断!")
            causal_mask = causal_mask[:, :, :mask_length, :mask_length]

        return causal_mask
    
    def save_model(self, step):
        self.logger.info("Saving Lora")  
        # filtered_state_dict = self.model[0].state_dict()['transformer.prefix_encoder.embedding.weight']

        check_fold = os.path.join(self.model_args.output_dir, "model-B")

        # out_dir = os.path.join(check_fold, 'lora-adapter.bin')

        self.servermodel.save_pretrained(os.path.join(check_fold, 'checkpoint-{}'.format(str(step+1))))
        self.logger.info("Lora权重保存成功!") 

    def fit(self, num_rounds: int, timeout: Optional[float]=None): 
        """Run split federated learning with clients."""
        history = History()
        if self.model_args.client_count>1:
            histories = {}
            # Create the first history object
            histories['history0'] = history
            # Create additional history objects based on client_count
            for i in range(1, self.model_args.client_count):
                histories[f'history{i}'] = History()
        # Initialize parameters
        self.logger.info("Initializing fed-split learning!")

        # get number of clients
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=num_rounds,
            parameters=None,
            client_manager=self._client_manager,
        ) 

        if not client_instructions:
            self.logger.info( "Caution! No clients selected, cancel and check again!")
            return None
        log(
            DEBUG,
            "strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )
        self.logger.info( "Total of {} clients participating.".format(self._client_manager.num_available()))

        # Run split federated learning for num_rounds
        self.logger.info( "Split-FL starting!")
        start_time = timeit.default_timer()
        if self.model_args.do_train:
            self.train(client_instructions, num_rounds, timeout, start_time, history)
        elif self.model_args.do_BatchParallel_train:
            self.BatchParallel_train(client_instructions, num_rounds, timeout, start_time, histories)
        elif self.model_args.do_eval:
            self.evaluate(client_instructions, num_rounds, timeout, start_time, history)
        elif self.model_args.do_predict:
            self.predict(client_instructions, num_rounds, timeout, start_time, history)


        # all finished
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        self.logger.info( "Fed-split learning finished in %s", elapsed)

        # save model
        # torch.save(self.servermodel, 'checkpoint/testcheckmlp/server.pt')

        # aggregate
        # self.logger.info( "Start aggregate clients' models parameters!")
        # paras_part1 = []
        # paras_part3 = []
        # for id, (client, ins) in client_instructions:
        #     paras_part1.append(client.get_parameters())
        #     paras_part3.append(client.get_parameters_part3())

        return history
    
    def train(self, client_instructions, num_rounds, timeout, start_time, history):
        # default with serial training
        client_id = 1
        for client_proxy, ins in client_instructions:          
            for current_round in range(num_rounds):

                '''=====================================================train client part A 前向传播模型A====================================================='''
                step_time = timeit.default_timer()
                
                ins.config['type'] = 1
                fitres_partA = fit_client_model1(client_proxy, current_round, ins, timeout) # fitres          
                featureA = parameters_to_ndarrays(fitres_partA.parameters)
                
                # #查看精度损失
                # stored_path = "/home/zhangzishuai/SplitFederated-LLaMA/pickle-store/featureA_ndarrays_server.pickle"
                # with open(stored_path, 'wb') as f:
                #     pickle.dump(featureA, f)

                '''=====================================================train server model B 前向传播模型B====================================================='''
                hidden_stateA = torch.from_numpy(featureA[0]).to(self.device)
                hidden_stateA = hidden_stateA.clone().detach().requires_grad_(True)
                real_input_num = torch.from_numpy(featureA[1]).to(self.device)
                p_ids = torch.from_numpy(featureA[2]).to(self.device)
                # 训练前向传播的时候不需要pkv
                
                # #适配batch为1的情形
                # pad_len = self.model_args.max_source_length + self.model_args.max_target_length - real_input_num #虽然在batchsize为1的时候原本不需要pad，但是为了不传递pkv，因此还是pad
                # att_mask= [0]* pad_len + [1]*real_input_num 
                # att_mask=att_mask.unsqueeze(0)
                # 适配多batch的情形
                max_len = self.model_args.max_source_length + self.model_args.max_target_length
                pad_len = max_len - real_input_num
                # expand用于张量维度扩展
                indices = torch.arange(max_len, device=self.device).expand(len(pad_len), -1)
                att_mask = (indices >= pad_len.unsqueeze(1)).to(torch.int64) #这里 unsqueeze(1) 将 pad_len 的维度从 [batch_size] 调整为 [batch_size, 1] 后，PyTorch 会自动将其 广播 到 [batch_size, max_len] 的形状

                att_mask=torch.tensor(att_mask, dtype=self.torch_dtype,device=self.device)
                self.att_mask =self.update_causal_mask(att_mask, p_ids) #torch.Size([2, 1, 4096, 4096])

                featureB = self.train_server(self.servermodel, 
                                             position_ids = p_ids,
                                             attention_mask = self.att_mask,
                                             hidden_state = hidden_stateA)
                                             #past_key_values=None) # dict

                hidden_stateB = featureB.last_hidden_state.clone().detach()#.requires_grad_(True)
                # att_mask = featureB.attention_mask.clone().detach()
                p_ids = featureB.position_ids.clone().detach()

                # # pkv2 = featureB.past_key_values[0].clone().detach()# .requires_grad_(True)
                # pkv2 = []
                # past_key= featureB.past_key_values.key_cache[0:31]
                # past_value = featureB.past_key_values.value_cache[0:31]
                # for past_key in past_key:
                #     pkv2.append(past_key.clone().detach()) 
                # for past_value in past_value:    
                #     pkv2.append(past_value.clone().detach()) 
                # del past_key
                # del past_value
                # pkv2 = torch.cat(pkv2, dim=0) 
                # seen_tokens = torch.tensor(featureB.past_key_values.seen_tokens).to(self.device)

                featureB_od = collections.OrderedDict([('hidden_state', hidden_stateB),
                                                    #    ('attention_mask', att_mask),
                                                       ('position_ids', p_ids)])
                feature_array = [val.cpu().numpy() for _, val in featureB_od.items()]
                ins.parameters = ndarrays_to_parameters(feature_array)

                # print('fitins for client part C', ins)
                
                '''=====================================================train model C 前向传播模型C并梯度反传====================================================='''
                ins.config['type'] = 2
                fitres_partC = fit_client_model2(client_proxy, ins, timeout) # featureB's gradient
                #print(fitres_partC)
                gradient_server = parameters_to_ndarrays(fitres_partC.parameters)
                loss = fitres_partC.metrics['loss']

                gradient_server_hidden = torch.from_numpy(gradient_server[0]).to(self.device)
                #gradient_last_pkv = torch.from_numpy(gradient_server[1]).to(self.device)
                # print("server side gradient:", gradient_server)
                # loss = fitres_partC.metrics['loss']

                '''=====================================================更新server model====================================================='''
                # step server model
                self.optim.zero_grad()
                featureB.last_hidden_state.backward(gradient_server_hidden)

                nn.utils.clip_grad_norm_(self.servermodel.parameters(), self.model_args.max_grad_norm)
                self.optim.step()
                self.schedule.step()

                if (current_round+1) % self.model_args.save_step == 0:
                    # self.save_model(config['current_step'], self.optim, self.schedule)
                    self.save_model(current_round)

                '''=====================================================更新 client model partA ====================================================='''
                ins.config['type'] = 3
                ins.config['current_step'] = current_round
                # gradient_od = collections.OrderedDict([('hidden_gradient', hidden_stateA.grad),
                #                                        ('past_key_value_gradient',pkv1_grad)]) # it's a tuple of tensor need change to tensor!
                gradient_od = collections.OrderedDict([('hidden_gradient', hidden_stateA.grad)]) # it's a tuple of tensor need change to tensor!
                gradient =  [val.cpu().numpy() for _, val in gradient_od.items()]
                ins.parameters = ndarrays_to_parameters(gradient)
                # print(ins) # FitIns tensor:list[bytes]

                # print('fitins for backward client model A')

                _ = back_client_model1(client_proxy, ins, timeout)
                step_end = timeit.default_timer()
                step_elapsed = step_end - step_time
                # self.logger.info( "steps end in %s", step_elapsed)
                # self.logger.info( "steps %s: loss %s", current_round, fitres_partC.metrics['loss'])
                self.logger.info("steps %s: loss %s", current_round, loss)
                # wandb.log({"loss": loss})
                # reset fit_ins
                ins.parameters = Parameters(tensors=[], tensor_type="")

                # print('fit ins for next step:',ins) 
                # print(fitres_partC) # FitRes, metrics{}
                # it seams that fitresA=fitresC
                #history.add_loss_distributed(current_round, fitres_partC.metrics['loss'])
                history.add_loss_distributed(current_round, loss)

            # Bookkeeping
            end_time = timeit.default_timer()
            elapsed = end_time - start_time
            self.logger.info("client %d finished in %s", client_id, elapsed)
            client_id += 1
            
        # wandb.finish()

    def BatchParallel_train(self, client_instructions, num_rounds, timeout, start_time, histories):
        """
        正常训练流程
        开启嵌入重建数据攻击：
            - 客户端0与服务器串通,将其私有数据提供给服务器用于训练攻击 decoder
            - 服务器利用攻击 decoder 对客户端1的隐藏状态进行重建
        """
        for current_round in range(num_rounds):
            # Initialize tensors to store data from all clients
            hidden_statesA_list = []
            att_masks = []
            p_ids_list = []
            pkvs1_list = []
            order = 0
            target_real_input_num = 0
            order_client_instructions={}
            colluding_labels = None
            for client_proxy, ins in client_instructions:
                '''=====================================================train client part A 前向传播模型A====================================================='''
                step_time = timeit.default_timer()
                
                ins.config['type'] = 1
                fitres_partA = fit_client_model1(client_proxy, current_round, ins, timeout) # fitres               
                featureA = parameters_to_ndarrays(fitres_partA.parameters)
                del fitres_partA
                '''=====================================================train server model B 前向传播模型B=====================================================''' 
                # feature1: list of array
                hidden_stateA = torch.from_numpy(featureA[0]).to(self.device)
                hidden_stateA = hidden_stateA.clone().detach().requires_grad_(True)
                # hidden_stateA: torch.Size([2, 920, 4096])
                real_input_num = torch.from_numpy(featureA[1]).to(self.device)
                p_ids = torch.from_numpy(featureA[2]).to(self.device)
                # p_ids: torch.Size([1, 920])

                if self.model_args.Attack: 
                    if len(featureA)==5:
                        colluding_labels =  torch.from_numpy(featureA[3]).to(self.device)  # 需补充 client0_labels 的获取方式
                        # self.logger.info(f"client_proxy.cid {client_proxy.cid}是Client0")
                        order_client_instructions["Client0_cid"] = client_proxy.cid
                        order_client_instructions["Client0_order"] = order
                        order += 1
                    else:
                        Client_1_labels =  torch.from_numpy(featureA[3]).to(self.device) 
                        # self.logger.info(f"client_proxy.cid {client_proxy.cid}是Client1")
                        order_client_instructions["Client1_cid"] = client_proxy.cid
                        order_client_instructions["Client1_order"] = order
                        target_real_input_num = real_input_num
                        order += 1
                    if len(order_client_instructions)==4 and current_round==0:
                        self.logger.info(f"order_client_instructions: {order_client_instructions}")

                # 适配多batch的情形
                max_len = self.model_args.max_source_length + self.model_args.max_target_length
                pad_len = max_len - real_input_num
                # expand用于张量维度扩展
                indices = torch.arange(max_len, device=self.device).expand(len(pad_len), -1)
                att_mask = (indices >= pad_len.unsqueeze(1)).to(torch.int64) 
                att_mask = att_mask.to(dtype=self.torch_dtype, device=self.device)

                self.att_mask =self.update_causal_mask(att_mask, p_ids) #torch.Size([2, 1, 4096, 4096])

                hidden_statesA_list.append(hidden_stateA)
                att_masks.append(self.att_mask)
                p_ids_list.append(p_ids)
                # pkvs1_list.append(pkv1)
                del hidden_stateA
                del att_mask
                del p_ids

                
            '''=====================================================train server model B 前向传播模型B====================================================='''
            # Concatenate collected data into batch tensors
            hidden_stateA = torch.cat(hidden_statesA_list, dim=0)
            hidden_stateA.retain_grad()
            # hidden_stateA:torch.Size([4, 920, 4096]) dim 0是bacth_size
            att_mask = torch.cat(att_masks, dim=0)
            # att_mask:torch.Size([2, 1, 640, 768]) dim 0是bacth_size
            p_ids = torch.cat(p_ids_list, dim=0)
            # p_ids:torch.Size([2, 2, 640]) dim 0是bacth_size
            del featureA
            
            featureB = self.train_server(self.servermodel, 
                                            position_ids = p_ids,
                                            attention_mask = att_mask,
                                            hidden_state = hidden_stateA)
            
            '''=====================================================重建数据攻击 ====================================================='''
            # 利用客户端0（colluding client）数据训练攻击 decoder，并对客户端1进行重建
            if self.model_args.Attack and (self.attack_decoder_model is not None):
                if colluding_labels is None:
                    self.logger.warning("没有收到客户端的数据！")

                assert not torch.equal(colluding_labels, Client_1_labels), "传统标签和私有标签张量完全一致"

                # 假设client_instructions中的第一个客户端为客户端0（串通方），第二个为客户端1（目标）
                batch_size = self.model_args.batch_size
                if order_client_instructions["Client0_order"] == 0:
                    # 客户端0数据（colluding client）：取第一个batch_size的样本
                    colluding_hidden = hidden_stateA[0:batch_size, :, :]
                    # 注意：这里假设 colluding 客户端的原始标签已提前准备好，形状为 [batch_size, seq_len]
                    colluding_att_mask = att_mask[0:batch_size,:,:,:]
                    # 利用训练好的攻击模型对客户端1的数据进行重建
                    target_hidden = hidden_stateA[batch_size:batch_size*self.model_args.client_count, :, :]
                    target_att_mask = att_mask[batch_size:batch_size*self.model_args.client_count,:,:,:]
                else:
                    # 客户端0数据（colluding client）：取第一个batch_size的样本
                    colluding_hidden = hidden_stateA[batch_size:batch_size*self.model_args.client_count, :, :]
                    # 注意：这里假设 colluding 客户端的原始标签已提前准备好，形状为 [batch_size, seq_len]
                    colluding_att_mask = att_mask[batch_size:batch_size*self.model_args.client_count,:,:,:]
                    target_hidden = hidden_stateA[0:batch_size, :, :]
                    target_att_mask = att_mask[0:batch_size,:,:,:]

                colluding_hidden = colluding_hidden.to(torch.device(self.model_args.device))
                colluding_att_mask = colluding_att_mask.to(torch.device(self.model_args.device))
                target_hidden = target_hidden.to(torch.device(self.model_args.device))
                target_att_mask = target_att_mask.to(torch.device(self.model_args.device))

                # 前向通过攻击 decoder 模型
                if self.model_args.Attack_Train:
                    self.attack_decoder_model.train()
                    if self.model_args.Attack_no_mask:
                        attack_logits = self.attack_decoder_model(input_ids=colluding_hidden)
                    else:
                        attack_logits = self.attack_decoder_model(input_ids=colluding_hidden, attention_mask = colluding_att_mask)
                    # 预测下一个token
                    if self.model_args.Attack_predict_next:
                        shift_logits = attack_logits[..., :-1, :].contiguous()
                        shift_labels = colluding_labels[..., 1:].contiguous()
                        # Flatten the tokens
                        shift_logits=shift_logits.view(-1, shift_logits.size(-1))
                        shift_labels=shift_labels.view(-1)
                        attack_loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=3)
                    else:
                        # 预测当前token
                        # 计算交叉熵损失
                        attack_loss = torch.nn.functional.cross_entropy(
                            attack_logits.view(-1, self.attack_decoder_model.vocab_size), # torch.Size([2, 920, 32000])
                            colluding_labels.view(-1), # torch.Size([2, 920])
                            ignore_index=3
                        )

                    self.attack_optimizer.zero_grad()
                    attack_loss.backward()
                    self.attack_optimizer.step()

                self.attack_decoder_model.eval()
                with torch.no_grad():
                    if self.model_args.Attack_no_mask:
                        attack_logits_target = self.attack_decoder_model(input_ids=target_hidden) # torch.Size([2, 920, 32000])
                    else:
                        attack_logits_target = self.attack_decoder_model(target_hidden, target_att_mask) # torch.Size([2, 920, 32000])
                    # 取每个位置上概率最大的token id作为预测结果
                    attack_pred_ids = attack_logits_target.argmax(dim=-1)  # shape: [batch_size, seq_len]

                # 对每个样本，从 real_input_num 开始保留有效 token
                filtered_ids = []
                for i, real_input_num in enumerate(target_real_input_num):
                    filtered_ids.append(attack_pred_ids[i, -real_input_num:])
                reconstructed_texts = self.attack_tokenizer.batch_decode(filtered_ids, skip_special_tokens=True)

                self.logger.info("step %d, attack_loss %.6f, Reconstructed text for client1: %s", current_round, attack_loss.item(), reconstructed_texts)
                if self.model_args.Attack and self.model_args.Attack_wandb:
                    wandb.log({"attack_loss": attack_loss.item()},step= self.global_step)
                # 计算rouge指标
                def compute_attack_metrics(preds, labels):
                    # score_dict = {
                    #     "rouge-1": [],
                    #     "rouge-2": [],
                    #     "rouge-l": [],}
                    # for pred, label in zip(preds, labels):
                    #     # ...input
                    #     # prediction = list(jieba.cut(pred))
                    #     # abstract = list(jieba.cut(label))
                    #     rouge = Rouge()
                    #     # scores = rouge.get_scores(' '.join(prediction) , ' '.join(abstract))
                    #     scores = rouge.get_scores(preds, labels,avg=True)
                    #     result = scores[0]
                    # for k, v in result.items():
                    #     score_dict[k].append(round(v["f"] * 100, 4))
                    # for k, v in score_dict.items():
                    #     score_dict[k] = float(np.mean(v))
                    rouge = Rouge()
                        # scores = rouge.get_scores(' '.join(prediction) , ' '.join(abstract))
                    scores = rouge.get_scores(preds, labels,avg=True)
                    # return score_dict
                    return scores
                
                def compute_bleu_sentence(hypothesis, reference):
                    """
                    计算单个句子对的 BLEU-4 指标。
                    
                    参数:
                    hypothesis (str): 模型生成的句子，作为假设句子。
                    reference (str): 真实参考句子。
                    
                    返回:
                    bleu_score (float): 该句子的 BLEU-4 分数。
                    """
                    # 简单地以空格分词，可根据需要使用更复杂的分词方法（如 nltk.word_tokenize）
                    hypothesis_tokens = hypothesis.split()
                    reference_tokens = reference.split()
                    # 设置 BLEU-4 的权重，即 1-gram 到 4-gram 的权重均为 0.25
                    weights = (0.25, 0.25, 0.25, 0.25)
                    bleu_score = sentence_bleu([reference_tokens], hypothesis_tokens, weights=weights)
                    return bleu_score

                True_label = self.attack_tokenizer.batch_decode(Client_1_labels.cpu().numpy(), skip_special_tokens=True)
                self.logger.info(f"step {current_round}, True_label {True_label} ")
                # score_dict = compute_attack_metrics(reconstructed_texts,True_label)
                # if self.model_args.Attack and self.model_args.Attack_wandb:
                #     wandb.log({"rouge-2": score_dict['rouge-2']["f"]})
                # self.logger.info(f"step {current_round}, attack_metircs {score_dict} ")
                try:
                    score_dict = compute_attack_metrics(reconstructed_texts, True_label)
                    self.rouge_2.append(score_dict['rouge-2']["f"])
                    if self.model_args.Attack and self.model_args.Attack_wandb:
                        wandb.log({"rouge-2": score_dict['rouge-2']["f"]},step= self.global_step)
                    self.logger.info(f"step {current_round}, attack_metircs {score_dict} ")
                    bleu_4 = compute_bleu_sentence(reconstructed_texts, True_label)
                    self.bleu.append(bleu_4)
                    self.logger.info(f"Sentence-level BLEU-4 Score:{bleu_4}")

                    # 计算当前所有样本的平均指标
                    avg_rouge_2 = np.mean(self.rouge_2) if self.rouge_2 else 0
                    avg_bleu = np.mean(self.bleu) if self.bleu else 0
                    self.logger.info(f"当前平均的 rouge-2: {avg_rouge_2}")
                    self.logger.info(f"当前平均的 bleu-4: {avg_bleu}")
                except Exception as e:
                    self.logger.warning(f"An error occurred while computing attack metrics: {e}. Skipping this step.")
                    
            # 用完即清理
            del p_ids
            del att_mask

            hidden_stateB = featureB.last_hidden_state.clone().detach()#.requires_grad_(True)
            # torch.Size([640, 2, 4096])
            # att_mask = featureB.attention_mask.clone().detach()
            # torch.Size([2, 1, 640, 768])
            p_ids = featureB.position_ids.clone().detach()
            # torch.Size([2, 2, 640])
            # pkv2 = featureB.past_key_values

            # print(pkv2.size(0))
            # torch.Size([2, 128, 2, 32, 128])
            # pkv2 = tuple(pkv2)

            gradient_server_list=[]

            '''=====================================================train model C 前向传播模型C并梯度反传====================================================='''
            client_id = 0 # 不一定对应真实的id，这里指的是client_instructions的第一个client
            for client_proxy, ins in client_instructions:
                # 准备隐藏层
                # hidden_stateB_i = hidden_stateB[client_id:client_id+1,:,  :]
                hidden_stateB_i = hidden_stateB[client_id*self.model_args.batch_size:(client_id+1)*self.model_args.batch_size, :,:]
                # torch.Size([640, 1, 4096])

                # p_ids_i = p_ids[client_id:client_id+1, :]
                p_ids_i = p_ids[client_id*self.model_args.batch_size:(client_id+1)*self.model_args.batch_size, :]

                featureB_od = collections.OrderedDict([('hidden_state', hidden_stateB_i),
                                                        ('position_ids', p_ids_i)])
                feature_array = [val.cpu().numpy() for _, val in featureB_od.items()]
                ins.parameters = ndarrays_to_parameters(feature_array) 
                del feature_array
                '''=====================================================batch train model C 前向传播模型C并梯度反传====================================================='''
                ins.config['type'] = 2
                fitres_partC = fit_client_model2(client_proxy, ins, timeout) # featureB's gradient
                #print(fitres_partC)
                gradient_server = parameters_to_ndarrays(fitres_partC.parameters)
                loss = fitres_partC.metrics['loss']

                gradient_server_hidden = torch.from_numpy(gradient_server[0]).to(self.device).requires_grad_(True)
                gradient_server_hidden.retain_grad() #7/23新增
                # torch.Size([640, 1, 4096])
                # gradient_last_pkv = torch.from_numpy(gradient_server[1]).to(self.device).requires_grad_(True)
                # gradient_last_pkv.retain_grad()  # 7/23新增
                # torch.Size([2, 128, 1, 32, 128])
                gradient_server_list.append(gradient_server_hidden)
                client_real_id = int(torch.from_numpy(gradient_server[1])) #client_real_id是flclient的真实id，对应parser.add_argument("--client_id", type=int, default=0)的值
                # client_id和这里的client_real_id一一对应

                # del gradient_server_hidden
                # del gradient_last_pkv
                # torch.cuda.empty_cache()
               
                # print("server side gradient:", gradient_server)
                # loss = fitres_partC.metrics['loss']
                #self.logger.info( "steps %s client %d loss %s", current_round, client_id, fitres_partC.metrics['loss'])
                # logger.info("steps %s client %d loss %s", current_round, client_id, fitres_partC.metrics['loss'])
                self.logger.info("steps %s client %s loss %s", current_round, client_real_id, loss)
                if self.model_args.Attack and self.model_args.Attack_wandb:
                    if client_real_id==0:
                        wandb.log({"client0_loss": loss},step= self.global_step)
                    else:
                        wandb.log({"client1_loss": loss},step= self.global_step)
                client_id += 1

            gradient_server_hidden = torch.cat(gradient_server_list, dim=0).requires_grad_(True)


            del gradient_server_list
            torch.cuda.empty_cache()

            '''=====================================================更新server model====================================================='''
            self.optim.zero_grad()
            featureB.last_hidden_state.backward(gradient_server_hidden)
            #梯度裁剪
            nn.utils.clip_grad_norm_(self.servermodel.parameters(), self.model_args.max_grad_norm)
            self.optim.step()
            self.schedule.step()

            if (current_round+1) % self.model_args.save_step == 0:
                # self.save_model(config['current_step'], self.optim, self.schedule)
                self.save_model(current_round)

            client_id = 0
            for client_proxy, ins in client_instructions:
                '''=====================================================更新 client model partA ====================================================='''
                ins.config['type'] = 3
                ins.config['current_step'] = current_round
                hidden_stateA_i = hidden_stateA.grad[client_id*self.model_args.batch_size:(client_id+1)*self.model_args.batch_size, :,:]
                # torch.Size([ 1,688,4096])

                gradient_od = collections.OrderedDict([('hidden_gradient', hidden_stateA_i)]) # it's a tuple of
                # tensor need change to tensor!
                gradient =  [val.detach().cpu().numpy() for _, val in gradient_od.items()]
                ins.parameters = ndarrays_to_parameters(gradient)

                # del gradient_od
                # del gradient
                # torch.cuda.empty_cache()

                # print('fitins for backward client model A')
                _ = back_client_model1(client_proxy, ins, timeout)

                step_end = timeit.default_timer()
                step_elapsed = step_end - step_time
                # self.logger.info( "steps end in %s", step_elapsed)
                # self.logger.info( "steps %s: loss %s", current_round, loss)
                ins.parameters = Parameters(tensors=[], tensor_type="")
                # print('fit ins for next step:',ins)
                # print(fitres_partC) # FitRes, metrics{}
                # it seams that fitresA=fitresC
                # histories[f'history{client_id}'].add_loss_distributed(current_round, fitres_partC.metrics['loss'])
                histories[f'history{client_id}'].add_loss_distributed(current_round, loss)
                client_id += 1

            # Bookkeeping
            end_time = timeit.default_timer()
            elapsed = end_time - start_time
            elapsed_formatted = "{:.2f}".format(elapsed)
            self.logger.info( "step %d  %d clients finished in %s s",current_round, client_id, elapsed_formatted)
            self.global_step += 1    
            # reset fit_ins
            ins.parameters = Parameters(tensors=[], tensor_type="")

    def evaluate(self, client_instructions, num_rounds, timeout, start_time, history):
        # default with serial trainging
        
        # with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
        client_id = 1
        self.model_args.max_output_length = 128
        for client_proxy, ins in client_instructions:
            start_time = timeit.default_timer()      
            for current_round in range(num_rounds):  # 有多少个num_rounds应该就回答多少个问题
                #TODO num_rounds= 指定的num_rounds和测试集数目的最小值
                need_test_data =True
                step_time = timeit.default_timer()
                for _ in range(self.model_args.max_output_length):                
                    ins.config['type'] = 1
                    fitres_partA = fit_client_model1(client_proxy, current_round, ins, timeout, need_test_data)          
                    featureA = parameters_to_ndarrays(fitres_partA.parameters)
                    hidden_stateA = torch.from_numpy(featureA[0]).to(self.device)
                    hidden_stateA = hidden_stateA.clone().detach().requires_grad_(True)
                    att_mask = torch.from_numpy(featureA[1]).to(self.device)
                    p_ids = torch.from_numpy(featureA[2]).to(self.device)
                    pkv=torch.from_numpy(featureA[3]).to(self.device)   # 当batch_size为1 torch.Size([2, 32, 352, 128])
                    past_key, past_value = torch.split(pkv, 1, dim=0) 
                    pkv1= DynamicCache.from_legacy_cache(None)
                    pkv1.key_cache.append(past_key)
                    pkv1.value_cache.append(past_value)
                    pkv1.seen_tokens = torch.from_numpy(featureA[4])  
                    # predict server model
                    featureB = self.predict_server(self.servermodel,
                                                position_ids = p_ids,
                                                attention_mask = att_mask,
                                                hidden_state = hidden_stateA,
                                                past_key_values = pkv1) # dict

                    hidden_stateB = featureB.last_hidden_state.clone().detach()#.requires_grad_(True)
                    att_mask = featureB.attention_mask.clone().detach()
                    p_ids = featureB.position_ids.clone().detach()
                    pkv2 = []
                    past_key= featureB.past_key_values.key_cache[0:31]
                    past_value = featureB.past_key_values.value_cache[0:31]
                    for past_key in past_key:
                        pkv2.append(past_key.clone().detach()) 
                    for past_value in past_value:    
                        pkv2.append(past_value.clone().detach()) 
                    del past_key
                    del past_value
                    pkv2 = torch.cat(pkv2, dim=0) 
                    seen_tokens = torch.tensor(featureB.past_key_values.seen_tokens).to(self.device)

                    featureB_od = collections.OrderedDict([('hidden_state', hidden_stateB),
                                                        ('attention_mask', att_mask),
                                                        ('position_ids', p_ids),
                                                        ('past_key_values', pkv2),
                                                        ('seen_tokens',seen_tokens)])
                    feature_array = [val.cpu().numpy() for _, val in featureB_od.items()]
                    ins.parameters = ndarrays_to_parameters(feature_array)
                    # print('fitins for client part C', ins)

                    # predict client partC
                    ins.config['type'] = 2
                    need_test_data = False
                    #print("得到最终输出token:")
                    fitres_partC = fit_client_model2(client_proxy, ins, timeout)  # We should return the prediction result
                    if fitres_partC.metrics['pred_finished']==1:
                        step_end = timeit.default_timer()
                        time_cost = step_end - step_time
                        self.logger.info(f"第{current_round+1}个问题回答结束!耗时{time_cost:.2f} 秒。Loss为{fitres_partC.metrics['loss']:.4f}")
                        history.add_loss_distributed(current_round, fitres_partC.metrics['loss'])  
                        self.model_args.start_inference = "NO"               
            # 回答完所有问题
            end_time = timeit.default_timer()
            elapsed = end_time - start_time
            self.logger.info("client %d finished in %s", client_id, elapsed)
            client_id += 1

    def predict(self, client_instructions, num_rounds, timeout, start_time, history):
        # torch.cuda.memory._record_memory_history()
        client_id = 1
        for client_proxy, ins in client_instructions:
            current_round=0
            # current_round=416 在这里指定从哪个问题开始回答
            while True:
                # current_round表示已经回答了多少个问题
                if current_round==self.model_args.max_predict_samples:
                    break
                if self.child_conn_server is not None:
                    if self.child_conn_server.poll():
                        tmp = self.child_conn_server.recv()
                        self.model_args.start_inference = "OK"
                else:
                    self.model_args.start_inference = "OK"
                if self.model_args.start_inference == "OK":
                    print("开始前向传播")
                    need_test_data =True
                    start_time = time.time()
                    # for _ in range(self.model_args.max_output_length):
                    for inference_step in range(self.model_args.max_output_length):
                        ins.config['type'] = 1
                        # fitres_partA_start = time.time() #
                        fitres_partA = fit_client_model1(client_proxy, current_round, ins, timeout, need_test_data)
                        # fitres_partA_end = time.time() #
                        # fitres_partA_cost = fitres_partA_end - fitres_partA_start # 
                        # self.logger.info(f"接收到fitres_partA_cost:{fitres_partA_cost},此时时间为:{time.time()}") #
                        featureA = parameters_to_ndarrays(fitres_partA.parameters)
                        if need_test_data:
                            hidden_stateA = torch.from_numpy(featureA[0]).to(self.device)
                            hidden_stateA = hidden_stateA.clone().detach().requires_grad_(True)
                            att_mask = torch.from_numpy(featureA[1]).to(self.device)
                            p_ids = torch.from_numpy(featureA[2]).to(self.device)
                            raw_attn_mask = torch.from_numpy(featureA[3]).to(self.device)
                            if self.model_args.Attack: 
                                Client_1_labels =  torch.from_numpy(featureA[4]).to(self.device) 
                                # self.logger.info(f"client_proxy.cid {client_proxy.cid}是Client1")
                                real_input_num=torch.from_numpy(featureA[5]).to(self.device)
                                target_real_input_num = real_input_num

                        else:
                            hidden_stateA = torch.from_numpy(featureA[0]).to(self.device)
                            hidden_stateA = hidden_stateA.clone().detach().requires_grad_(True)
                            p_ids = torch.from_numpy(featureA[1]).to(self.device)



                        '''=====================================================重建数据攻击 ====================================================='''
                        # 利用客户端0（colluding client）数据训练攻击 decoder，并对客户端1进行重建
                        if self.model_args.Attack and (self.attack_decoder_model is not None) and need_test_data:

                            # 利用训练好的攻击模型对客户端1的数据进行重建
                            target_hidden = hidden_stateA
                            target_att_mask = raw_attn_mask
                            target_hidden = target_hidden.to(torch.device(self.model_args.device))
                            target_att_mask = target_att_mask.to(torch.device(self.model_args.device))

                            self.attack_decoder_model.eval()
                            with torch.no_grad():
                                if self.model_args.Attack_no_mask:
                                    attack_logits_target = self.attack_decoder_model(input_ids=target_hidden) # torch.Size([2, 920, 32000])
                                else:
                                    attack_logits_target = self.attack_decoder_model(target_hidden, target_att_mask) # torch.Size([2, 920, 32000])
                                # 取每个位置上概率最大的token id作为预测结果
                                attack_pred_ids = attack_logits_target.argmax(dim=-1)  # shape: [batch_size, seq_len]

                            # 对每个样本，从 real_input_num 开始保留有效 token
                            filtered_ids = []
                            for i, real_input_num in enumerate(target_real_input_num):
                                filtered_ids.append(attack_pred_ids[i, -real_input_num:])
                            reconstructed_texts = self.attack_tokenizer.batch_decode(filtered_ids, skip_special_tokens=True)

                            self.logger.info("step %d, Reconstructed text for client1: %s", current_round, reconstructed_texts)

                            # 计算rouge指标
                            def compute_attack_metrics(preds, labels):
                                # score_dict = {
                                #     "rouge-1": [],
                                #     "rouge-2": [],
                                #     "rouge-l": [],}
                                # for pred, label in zip(preds, labels):
                                #     # ...input
                                #     # prediction = list(jieba.cut(pred))
                                #     # abstract = list(jieba.cut(label))
                                #     rouge = Rouge()
                                #     # scores = rouge.get_scores(' '.join(prediction) , ' '.join(abstract))
                                #     scores = rouge.get_scores(preds, labels,avg=True)
                                #     result = scores[0]
                                # for k, v in result.items():
                                #     score_dict[k].append(round(v["f"] * 100, 4))
                                # for k, v in score_dict.items():
                                #     score_dict[k] = float(np.mean(v))
                                rouge = Rouge()
                                    # scores = rouge.get_scores(' '.join(prediction) , ' '.join(abstract))
                                scores = rouge.get_scores(preds, labels,avg=True)
                                # return score_dict
                                return scores
                            
                            def compute_bleu_sentence(hypothesis, reference):
                                """
                                计算单个句子对的 BLEU-4 指标。
                                
                                参数:
                                hypothesis (str): 模型生成的句子，作为假设句子。
                                reference (str): 真实参考句子。
                                
                                返回:
                                bleu_score (float): 该句子的 BLEU-4 分数。
                                """
                                hypothesis = hypothesis[0]
                                reference = reference[0]
                                # 简单地以空格分词，可根据需要使用更复杂的分词方法（如 nltk.word_tokenize）
                                hypothesis_tokens = hypothesis.split()
                                reference_tokens = reference.split()
                                # 设置 BLEU-4 的权重，即 1-gram 到 4-gram 的权重均为 0.25
                                weights = (0.25, 0.25, 0.25, 0.25)
                                bleu_score = sentence_bleu([reference_tokens], hypothesis_tokens, weights=weights)
                                return bleu_score

                            True_label = self.attack_tokenizer.batch_decode(Client_1_labels.cpu().numpy(), skip_special_tokens=True)
                            self.logger.info(f"step {current_round}, True_label {True_label} ")
                            # score_dict = compute_attack_metrics(reconstructed_texts,True_label)
                            # if self.model_args.Attack and self.model_args.Attack_wandb:
                            #     wandb.log({"rouge-2": score_dict['rouge-2']["f"]})
                            # self.logger.info(f"step {current_round}, attack_metircs {score_dict} ")
                            try:
                                score_dict = compute_attack_metrics(reconstructed_texts, True_label)
                                self.rouge_2.append(score_dict['rouge-2']["f"])
                                if self.model_args.Attack and self.model_args.Attack_wandb:
                                    wandb.log({"rouge-2": score_dict['rouge-2']["f"]},step= self.global_step)
                                self.logger.info(f"step {current_round}, attack_metircs {score_dict} ")
                                bleu_4 = compute_bleu_sentence(reconstructed_texts, True_label)
                                self.bleu.append(bleu_4)
                                self.logger.info(f"Sentence-level BLEU-4 Score:{bleu_4}")

                                # 计算当前所有样本的平均指标
                                avg_rouge_2 = np.mean(self.rouge_2) if self.rouge_2 else 0
                                avg_bleu = np.mean(self.bleu) if self.bleu else 0
                                self.logger.info(f"当前平均的 rouge-2: {avg_rouge_2}")
                                self.logger.info(f"当前平均的 bleu-4: {avg_bleu}")
                            except Exception as e:
                                self.logger.warning(f"An error occurred while computing attack metrics: {e}. Skipping this step.")

                        # predict server model
                        # featureB_start = time.time()
                        del fitres_partA
                        del featureA

                        if need_test_data:
                            self.cache_position = torch.arange(p_ids.shape[-1], dtype=torch.int64, device=self.model_args.device)
                            featureB = self.predict_server(self.servermodel,
                                                        attention_mask = att_mask, #torch.Size([1, 1, 4096, 4096])
                                                        hidden_state = hidden_stateA, #torch.Size([1, 102, 4096])
                                                        past_key_values = self.pkv_modelB,
                                                        cache_position=self.cache_position) # torch.Size([102]) 
                            self.att_mask =self.update_causal_mask(raw_attn_mask, p_ids)

                        else:
                            self.cache_position = self.cache_position[-1:] + 1 
                            featureB = self.predict_server(self.servermodel,
                                                        attention_mask = self.att_mask,
                                                        hidden_state = hidden_stateA,
                                                        past_key_values = self.pkv_modelB,
                                                        cache_position=self.cache_position) # dict
                            self.att_mask =self.update_causal_mask(self.att_mask, p_ids)

                        # featureB_end = time.time() #
                        # featureB_cost = featureB_end - featureB_start # 
                        # self.logger.info(f"featureB训练_cost:{featureB_cost}") #
                        hidden_stateB = featureB.last_hidden_state.clone().detach()#.requires_grad_(True)
                        att_mask = featureB.attention_mask.clone().detach()
                        p_ids = featureB.position_ids.clone().detach()

                        if self.model_args.do_inference or self.model_args.do_predict:
                            # self.logger.info(id(self.pkv_modelB))
                            # self.logger.info(id(featureB.past_key_values)) 
                            # 他们两个共享内存，所以pkv_modelB的值会随着featureB.past_key_values的值变化而变化
                            self.pkv_modelB = featureB.past_key_values

                        # if self.att_mask is not None:
                        #     if not torch.equal(self.att_mask, att_mask):
                        #         self.logger.info("当前self.att_mask <不> 等上一个att_mask")
                        #     else:
                        #         self.logger.info("当前self.att_mask等于上一个att_mask")

                        del featureB

                        if need_test_data:
                            featureB_od = collections.OrderedDict([('hidden_state', hidden_stateB), # 第一次 torch.Size([1, 102, 4096])
                                                                ('attention_mask', att_mask), # 第一次 torch.Size([1, 1, 4096, 4096])
                                                                ('position_ids', p_ids)]) # 第一次 torch.Size([1, 102]) 
                        else:
                            featureB_od = collections.OrderedDict([('hidden_state', hidden_stateB), # 后续 torch.Size([1, 1, 4096])
                                        ('position_ids', p_ids)]) # 后续 torch.Size([1, 1])

                        feature_array = [val.cpu().numpy() for _, val in featureB_od.items()]
                        # 保存feature看传输大小
                        # stored_path = "./output/CoQA/debug/llama_fit_B.pickle"
                        # with open(stored_path, 'wb') as f:
                        #     pickle.dump(feature_array, f)
                        #     self.logger.info(f"保存pickle数据完成")
                        ins.parameters = ndarrays_to_parameters(feature_array)

                        # predict client partC
                        ins.config['type'] = 2
                        need_test_data = False
                        #print("得到最终输出token:")
                        # fitres_partC_start = time.time() #
                        fitres_partC = fit_client_model2(client_proxy, ins, timeout)  # We should return the prediction result
                        # fitres_partC_end = time.time() #
                        # fitres_partC_cost = fitres_partC_end - fitres_partC_start # 
                        # self.logger.info(f"接收到fitres_partC_cost:{fitres_partC_cost}") #
                        torch.cuda.set_device(self.model_args.device)
                        torch.cuda.empty_cache()

                        if fitres_partC.metrics['pred_finished']==1:

                            self.att_mask = None
                            del self.pkv_modelB
                            torch.cuda.empty_cache() 
                            self.pkv_modelB = DynamicCache()
                            for _ in range(self.model_args.modelA_layers):
                                self.pkv_modelB.key_cache.append(torch.tensor(0, dtype=torch.int8, device=self.model_args.device))
                                self.pkv_modelB.value_cache.append(torch.tensor(0, dtype=torch.int8, device=self.model_args.device))
                            self.cache_position = None
                            
                            end_time = time.time()
                            time_cost = end_time - start_time
                            self.logger.info( f"第{current_round+1}个问题回答结束!耗时{time_cost:.2f} 秒。")
                            current_round +=1
                            if self.model_args.do_inference:
                                self.model_args.start_inference = "NO"   # 当是测试集时,不变成NO,以便继续测试
                            break


def fit_client_model1(
    client: ClientProxy, server_round: int, ins: FitIns, timeout: Optional[float]=None, need_test_data: Optional[bool]= False
) -> FitRes:
    """Refine parameters on a single client."""
    ins.config['current_step'] = server_round
    if need_test_data:
        ins.config['need_test_data']=1
    else:
        ins.config['need_test_data']=0
    fit_res = client.fit(ins, timeout)
    return fit_res

def fit_client_model2(
    client: ClientProxy, ins: FitIns, timeout: Optional[float]=None
) -> FitRes:
    """Refine parameters on a single client."""
    fit_res = client.fit(ins, timeout)
    # print("fit_res:", fit_res)
    return fit_res

def back_client_model1(
    client: ClientProxy, ins: FitIns, timeout: Optional[float]=None
) -> Tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    # ins.parameter = feature gradient
    
    fit_res = client.fit(ins, timeout) # 非抽象方法
    return fit_res


def save_client(client: ClientProxy) -> None:
    client.save_model()

def get_client_params_part1(client: ClientProxy):
    param = client.get_parameters({})
    return param

def get_client_params_part3(client: ClientProxy):
    param = client.get_parameters_part3({})
    return param


def _handle_finished_future_after_fit(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, FitRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)

def config_client(perclient_step):
    return {'perclient_step': perclient_step}


def main(child_conn_server=None):
    '''=====================================================设置参数====================================================='''
    FL_args = FLparser()
    # 设置日志文件,logger 的作用域是局限于 main() 函数内部
    # log_path = os.path.join(FL_args.output_dir, 'train_server.log')
    log_path = os.path.join(FL_args.output_dir, f'server_Alayers{FL_args.modelA_layers}_Clayers{FL_args.modelC_layers}.log')
    logging.basicConfig(filename=log_path,
                        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                        level=logging.INFO,
                        filemode='w') # 添加这个参数以覆盖已有的日志文件
    # 创建一个logger实例
    logger = logging.getLogger(__name__)

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
            
    if FL_args.Attack and FL_args.Attack_wandb:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project=f"Record_Distributed_Attack",
            name=f"noise_{FL_args.grad_noise}_Alayers_{FL_args.modelA_layers}_Attacklora_{FL_args.Attack_lora_checkpoint}_VariedEmbeding_{FL_args.Client1_VariedEmbeding}",
            # track hyperparameters and run metadata
            config={
            "learning_rate": FL_args.lr,
            "architecture": "Llama",
            "dataset": "Record",
            "total_steps": FL_args.max_step,
            }
        )

    global self_device
    self_device = FL_args.device
    device = torch.device(FL_args.device)
    print(f"device:{device},self_device:{self_device}")

    # 确保实验可复现
    torch.manual_seed(FL_args.seed)
    random.seed(FL_args.seed)
    np.random.seed(FL_args.seed)
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True) 


    FL_args.per_client_steps = FL_args.max_step

    if FL_args.do_train:
        logger.info("正在进行服务器端训练")
        assert not (FL_args.do_BatchParallel_train or FL_args.do_eval or FL_args.do_predict or FL_args.do_inference), "当 do_train 为 True 时，其它标志必须为 False"
    if FL_args.do_BatchParallel_train:
        FL_args.client_count = 2
        logger.info(f"正在进行服务器端批训练,客户端数: {FL_args.client_count}")
        assert not (FL_args.do_train or FL_args.do_eval or FL_args.do_predict or FL_args.do_inference), "当 do_BatchParallel_train 为 True 时，其它标志必须为 False"
    if FL_args.do_eval:
        logger.info("正在进行服务器端验证")
        assert not (FL_args.do_train or FL_args.do_BatchParallel_train or FL_args.do_predict or FL_args.do_inference), "当 do_eval 为 True 时，其它标志必须为 False"
    if FL_args.do_predict and not FL_args.do_inference:
        logger.info("正在进行服务器端测试")
        assert not (FL_args.do_train or FL_args.do_BatchParallel_train or FL_args.do_eval or FL_args.do_inference), "当 do_predict 为 True 时，其它标志必须为 False"
    if FL_args.do_inference:
        FL_args.do_predict = True
        logger.info("正在进行服务器端推理")
        assert not (FL_args.do_train or FL_args.do_BatchParallel_train or FL_args.do_eval), "当 do_inference 为 True 时,do_train, do_BatchParallel_train, do_eval 必须为 False"
    if FL_args.Attack:
        # assert FL_args.do_BatchParallel_train, "当攻击时,必须开启批量训练(do_BatchParallel_train=True)"
        logger.warning(f"正在实施重建数据攻击")
        #初始化decoder
        class LlamaAttackDecoder(nn.Module):
            def __init__(self, base_model_name_or_path: str, num_decoder_layers: int,config):
                super().__init__()
                if FL_args.Attack_random_initial:
                    logger.warning("随机初始化decoder参数")
                    base_model = LlamaForCausalLM(config=config)
                else:
                    logger.warning("加载预训练的 LLaMA 模型获取权重")
                    base_model = LlamaForCausalLM.from_pretrained(base_model_name_or_path,config=config)
                self.config = config 
                # 截取 LLaMA 模型的后 num_decoder_layers 层Transformer层
                # base_model.model 是 LlamaModel，包含 embed_tokens, layers, norm
                self.layers = nn.ModuleList([layer for layer in base_model.model.layers[-num_decoder_layers:]])
                # 保存隐藏维度和词表大小等信息
                self.hidden_size = self.config.hidden_size
                self.vocab_size = self.config.vocab_size
                
                # 最后的归一化层 (RMSNorm) 和输出词汇投影层 (lm_head)
                self.norm = base_model.model.norm  # 最终层归一化
                self.lm_head = base_model.lm_head  # 词汇投影层 (通常与 embed_tokens 权重 tie)
                # lm_head Linear(in_features=4096, out_features=32000, bias=False)
                
                # **注意**: 上述操作默认将 lm_head 权重与词嵌入权重 tie 在一起，假设服务器知道词表。
                # 如果服务器无权访问原始嵌入，可将 tie 解除并随机初始化 lm_head。
                
                # 将不再需要的部分删除以节省内存（如前面的层）
                del base_model  # 删除完整模型，释放内存

            def forward(
                self, 
                hidden_states=None, 
                attention_mask=None, 
                input_ids=None, 
                inputs_embeds=None, 
                past_key_value: Optional[Tuple[torch.Tensor]] = None,
                output_attentions: Optional[bool] = False,
                output_hidden_states=None,
                use_cache: Optional[bool] = False,
                cache_position: Optional[torch.LongTensor] = None,
                labels=None, 
                return_dict=False,
                **kwargs):
                # 如果 kwargs 中存在 labels，则将其删除，避免传递到下层
                if "labels" in kwargs:
                    kwargs.pop("labels")
                    
                # 优先使用 inputs_embeds，其次 input_ids，最后 hidden_states
                if inputs_embeds is not None:
                    hidden_states = inputs_embeds
                elif input_ids is not None: # 从这里进入
                    hidden_states = input_ids
                if hidden_states is None:
                    raise ValueError("必须提供 hidden_states, input_ids 或 inputs_embeds。")
                
                # 这是3.31日之前实际运行的
                # # 如果没有提供 attention_mask，则默认全部为1
                # if attention_mask is None:
                #     attention_mask = torch.ones(hidden_states.shape[:2], device=hidden_states.device)

                # 3.31日修改后的代码
                # 如果没有显式提供 attention_mask，则构造 causal mask
                if attention_mask is None:
                    seq_len = hidden_states.size(1)
                    attention_mask = torch.ones((1, 1, seq_len, seq_len), device=hidden_states.device)
                    attention_mask = torch.tril(attention_mask)  # 下三角矩阵，确保仅看到当前和之前的位置

                if attention_mask is not None and attention_mask.dtype != hidden_states.dtype:
                    attention_mask = attention_mask.to(hidden_states.dtype)

                # 自动生成 position_ids（这里假设有效 token 的位置从 0 开始）
                if "position_ids" in kwargs and kwargs["position_ids"] is not None:
                    position_ids = kwargs["position_ids"]
                else:
                    bs, seq, _ = hidden_states.shape
                    position_ids = torch.arange(seq, device=hidden_states.device).unsqueeze(0).expand(bs, seq)
                    kwargs["position_ids"] = position_ids

                # 逐层前向传播，将 attention_mask 和其它 kwargs 一并传递
                for layer in self.layers:
                    hidden_states = layer(hidden_states=hidden_states, attention_mask=attention_mask, **kwargs)[0]
                
                hidden_states = self.norm(hidden_states)
                logits = self.lm_head(hidden_states)
                return logits

            def prepare_inputs_for_generation(
                self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
            ):
                past_length = 0
                if past_key_values is not None:
                    if isinstance(past_key_values, Cache):
                        cache_length = past_key_values.get_seq_length()
                        past_length = past_key_values.seen_tokens
                        max_cache_length = past_key_values.get_max_length()
                    else:
                        cache_length = past_length = past_key_values[0][0].shape[2]
                        max_cache_length = None

                    # Keep only the unprocessed tokens:
                    # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
                    # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
                    # input)
                    if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                        input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
                    # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
                    # input_ids based on the past_length.
                    elif past_length < input_ids.shape[1]:
                        input_ids = input_ids[:, past_length:]
                    # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

                    # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
                    if (
                        max_cache_length is not None
                        and attention_mask is not None
                        and cache_length + input_ids.shape[1] > max_cache_length
                    ):
                        attention_mask = attention_mask[:, -max_cache_length:]

                position_ids = kwargs.get("position_ids", None)
                if attention_mask is not None and position_ids is None:
                    # create position_ids on the fly for batch generation
                    position_ids = attention_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 1)
                    if past_key_values:
                        position_ids = position_ids[:, -input_ids.shape[1] :]

                if past_key_value := getattr(self.model.layers[0].self_attn, "past_key_value", None):
                    # generation with static cache
                    past_length = past_key_value.get_seq_length()
                    input_ids = input_ids[:, past_length:]
                    position_ids = position_ids[:, past_length:]

                # TODO @gante we should only keep a `cache_position` in generate, and do +=1.
                # same goes for position ids. Could also help with continued generation.
                cache_position = kwargs.get("cache_position", None)
                if cache_position is None:
                    cache_position = torch.arange(
                        past_length, past_length + position_ids.shape[-1], device=position_ids.device
                    )

                # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
                if inputs_embeds is not None and past_key_values is None:
                    model_inputs = {"inputs_embeds": inputs_embeds}
                else:
                    model_inputs = {"input_ids": input_ids}

                model_inputs.update(
                    {
                        "position_ids": position_ids,
                        "cache_position": cache_position,
                        "past_key_values": past_key_values,
                        "use_cache": kwargs.get("use_cache"),
                        "attention_mask": attention_mask,
                    }
                )
                return model_inputs

        Num_attack_layers = FL_args.modelA_layers
        logger.info(f"Num_attack_layers: {Num_attack_layers}")
        config_attack = AutoConfig.from_pretrained(FL_args.Attack_model_path)
        attack_decoder_model = LlamaAttackDecoder(FL_args.Attack_model_path, num_decoder_layers=Num_attack_layers,config=config_attack)

        if FL_args.Attack_load_decoder and not FL_args.Attack_train:
            state_dict = torch.load(FL_args.Attack_decoder_path)
            attack_decoder_model.load_state_dict(state_dict)
            logger.warning(f"从{FL_args.Attack_decoder_path}加载训练好的权重")
            attack_decoder_model.eval()
            attack_optimizer=None
        else:
            attack_optimizer = AdamW(filter(lambda p: p.requires_grad, attack_decoder_model.parameters()), lr=FL_args.Attack_lr, betas=FL_args.betas, eps=FL_args.eps, weight_decay=FL_args.weight_decay)

        # if FL_args.Attack_Lora:
        #     logger.info("正在使用lora微调攻击模型")
        #     Attack_t_m_s = []
        #     for index in FL_args.target_modules:
        #         for i in list(range(Num_attack_layers)):
        #             Attack_t_m_s.append(str(f'layers.{i}.self_attn' + '.' + index))

        #     Attack_lora_config = LoraConfig(
        #         r=FL_args.Attack_lora_r,
        #         lora_alpha=FL_args.Attack_lora_alpha,
        #         target_modules =  Attack_t_m_s,
        #         fan_in_fan_out = False,
        #         lora_dropout=0.05,
        #         inference_mode=False,
        #         bias="none",
        #         task_type="CAUSAL_LM",
        #     )

        #     attack_decoder_model = get_peft_model(attack_decoder_model, Attack_lora_config)

        # else:
        #     logger.info("正在使用全参数微调攻击模型")


        attack_decoder_model=attack_decoder_model.to(device)
        attack_decoder_model=attack_decoder_model.to(torch.float16)
        tokenizer_kwargs = {
            "cache_dir": FL_args.cache_dir,
            "use_fast": FL_args.use_fast_tokenizer,
            "revision": FL_args.model_revision,
            "use_auth_token": True if FL_args.use_auth_token else None,
            "padding_side":'left'
        }
        attack_tokenizer = AutoTokenizer.from_pretrained(FL_args.Attack_model_path, **tokenizer_kwargs)
        attack_tokenizer.pad_token_id = 3


    else:
        logger.warning(f"没有实施重建数据攻击")
        attack_decoder_model=None
        attack_optimizer=None
        attack_tokenizer=None
    
    '''=====================================================配置Server Model====================================================='''
    start_set_model = time.time()
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

    def adjust_modelB_state_dict(original_dict, modelA_layers, modelC_layers, total_layers=32):
        """动态调整ModelB的状态字典键名"""
        logger.info(f"Adjusting ModelB state dict for {32-modelA_layers - modelC_layers} layers")
        new_dict = OrderedDict()
        # 计算实际需要加载的层数范围
        start_layer = modelA_layers
        end_layer = total_layers - modelC_layers
        target_layers = end_layer - start_layer
        
        # 验证参数有效性
        if target_layers <= 0:
            raise ValueError(f"Invalid layer split: modelA_layers({modelA_layers}) + modelC_layers({modelC_layers}) >= total_layers({total_layers})")
        
        # 处理非layer参数（假设ModelB不包含embedding等参数）
        for k in original_dict:
            if not k.startswith("model.layers"):
                new_dict[k] = original_dict[k]
        
        # 动态映射layer参数
        for new_layer_idx in range(target_layers):
            original_layer_idx = start_layer + new_layer_idx - 1
            src_prefix = f"model.layers.{original_layer_idx}."
            tgt_prefix = f"model.layers.{new_layer_idx}."
            
            # 需要双重验证：既要匹配原始层号，又要确保在目标范围内
            for k in original_dict:
                if k.startswith(src_prefix):
                    # 替换层号部分
                    new_key = k.replace(src_prefix, tgt_prefix, 1)
                    new_dict[new_key] = original_dict[k]
                    
        return new_dict

    def adjust_modelB_Lora_state_dict(original_dict, modelA_layers, modelC_layers, total_layers=32):
        """动态调整ModelB的状态字典键名"""
        logger.info(f"Adjusting ModelB Lora state dict for {32-modelA_layers - modelC_layers} layers")
        new_dict = OrderedDict()
        # 计算实际需要加载的层数范围
        start_layer = modelA_layers
        end_layer = total_layers - modelC_layers
        target_layers = end_layer - start_layer
        
        # 验证参数有效性
        if target_layers <= 0:
            raise ValueError(f"Invalid layer split: modelA_layers({modelA_layers}) + modelC_layers({modelC_layers}) >= total_layers({total_layers})")
        
        # 处理非layer参数（假设ModelB不包含embedding等参数）
        for k in original_dict:
            if not k.startswith("base_model.model.model.layers"):
                new_dict[k] = original_dict[k]
        
        # 动态映射layer参数
        for new_layer_idx in range(target_layers):
            original_layer_idx = start_layer + new_layer_idx - 1
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
                    
        return new_dict


    if FL_args.do_train or FL_args.do_BatchParallel_train:   
        config_kwargs = {
            "cache_dir": FL_args.cache_dir,
            "revision": FL_args.model_revision,
            "use_auth_token": True if FL_args.use_auth_token else None,
        }
        if FL_args.modelB_config_name:
            configB = AutoConfig.from_pretrained(FL_args.modelB_config_name, **config_kwargs)
        elif FL_args.modelB_name_or_path:  # 从这里进入
            configB = AutoConfig.from_pretrained(FL_args.modelB_name_or_path, **config_kwargs)
        else:
            configB = CONFIG_MAPPING[FL_args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")
            if FL_args.config_overrides is not None:
                logger.info(f"Overriding config: {FL_args.config_overrides}")
                configB.update_from_string(FL_args.config_overrides)
                logger.info(f"New config: {configB}")

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
        tokenizer.pad_token = tokenizer.unk_token

        # # logger.info(f"FL_args.target_modules:{FL_args.target_modules}")
        # t_m_s = []
        # for index in FL_args.target_modules:
        #     for i in list(range(32-FL_args.modelA_layers-FL_args.modelC_layers)):
        #         t_m_s.append(str(f'model.layers.{i}.self_attn' + '.' + index))
        # # logger.info(f"t_m_s:{t_m_s}")

        # 同时微调多个模块
        server_layer_id=list(range(32-FL_args.modelA_layers-FL_args.modelC_layers))
        t_m_s = []
        self_attn_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        other_layer_modules = ['gate_proj', 'up_proj', 'down_proj']
        global_modules = ['embed_tokens']  # 不属于某一层（如模型顶层）
        for index in FL_args.target_modules:
            logger.info(f"index: {index}")
            if index in global_modules:
                # 顶层模块，如 embed_tokens
                logger.info("模型B没有 embed_tokens ")
            elif index in self_attn_modules:
                # 属于 self_attn 的模块
                for i in server_layer_id:
                    t_m_s.append(f'model.layers.{i}.self_attn.{index}')
            elif index in other_layer_modules:
                # 属于每层的其他模块，比如 feed_forward
                for i in server_layer_id:
                    t_m_s.append(f'model.layers.{i}.{index}')
            else:
                logger.warning(f"未知模块类型: {index}，未添加到 target_modules")

        FL_args.target_modules = t_m_s
        logger.info(f"FL_args.target_modules:{FL_args.target_modules}")

        lora_config = LoraConfig(
            r=FL_args.lora_r,
            lora_alpha=FL_args.lora_alpha,
            # target_modules=["query_key_value"],
            # target_modules =  ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            target_modules =  FL_args.target_modules,
            fan_in_fan_out = False,
            lora_dropout=0.05,
            inference_mode=False,
            bias="none",
            task_type="CAUSAL_LM",
        )
        logger.info(f"lora_config:{lora_config}")

        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16
        # )

        if FL_args.modelB_name_or_path:

            torch_dtype = (
                FL_args.torch_dtype
                if FL_args.torch_dtype in ["auto", None]
                else getattr(torch, FL_args.torch_dtype)
            )
            logger.info(f"torch_dtype:{torch_dtype}") # torch.float16
            model_B = LlamaForCausalLMB(config=configB, modelA_layers=FL_args.modelA_layers, modelC_layers=FL_args.modelC_layers).to('cpu')
            #加载原状态字典
            server_state_dict = torch.load(f'{FL_args.modelB_name_or_path}/pytorch_model_B.bin', map_location=FL_args.device)
            #动态调整
            adjusted_B_dict = adjust_modelB_state_dict(server_state_dict, FL_args.modelA_layers, FL_args.modelC_layers)
            model_B.load_state_dict(adjusted_B_dict)
            del server_state_dict
            del adjusted_B_dict
            model_B.to(device)
            torch.cuda.set_device(FL_args.device)
            torch.cuda.empty_cache()

            # 如果model_B的数据类型与目标数据类型不同，则进行转换
            if model_B.dtype != model_torch_dtype:
                logger.info(f"Converting model_B from {model_B.dtype} to {model_torch_dtype}")
                model_B = model_B.to(model_torch_dtype)

        else:
            model_B = LlamaForCausalLMB.from_config(configB)
            n_params = sum({p.data_ptr(): p.numel() for p in model_B.parameters()}.values())
            logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

        model_B = get_peft_model(model_B, lora_config)

        # logger.info(f"下面是modelB的参数")
        # logger.info("\n")  
        # for name, param in model_B.named_parameters():
        #     if param.requires_grad:
        #         logger.info(f"Parameter Name: {name}, value: {param.data}, Requires Grad: {param.requires_grad}") 

        if FL_args.custom_lora:
            logger.warning(f"Initial lora weights from bin")
            logger.warning(f"You need to check lora r and lora alpha")
            Lora_B_state_dict = torch.load("./lora_r16_alpha32_weights/lora_B/adapter_model.bin")
            adjusted_LoraB_dict = adjust_modelB_Lora_state_dict(Lora_B_state_dict, FL_args.modelA_layers, FL_args.modelC_layers)
            model_B.load_state_dict(adjusted_LoraB_dict, strict=False)
            del Lora_B_state_dict
            del adjusted_LoraB_dict
            torch.cuda.empty_cache()
        else:
            logger.warning(f"Initial lora weights from automation")

        # logger.info(f"下面修改后是modelB的参数")
        # logger.info("\n")  
        # for name, param in model_B.named_parameters():
        #     if param.requires_grad:
        #         logger.info(f"Parameter Name: {name}, value: {param.data}, Requires Grad: {param.requires_grad}") 


        # model_B.print_trainable_parameters()
        # optimizer初始化 
        optimB = AdamW(filter(lambda p: p.requires_grad, model_B.parameters()), lr=FL_args.lr, betas=FL_args.betas, eps=FL_args.eps, weight_decay=FL_args.weight_decay)
        # schedule 初始化
        scheduleB = LinearLR(optimB, start_factor=1.0, end_factor=0.0, total_iters=FL_args.max_step)

        # #最简单的版本：
        # optimB = torch.optim.SGD(
        #     filter(lambda p: p.requires_grad, model_B.parameters()),
        #     lr=FL_args.lr
        # )
        # # 恒定学习率调度器（实际不改变学习率）
        # from torch.optim.lr_scheduler import LambdaLR
        # scheduleB = LambdaLR(optimB, lr_lambda=lambda epoch: 1.0)  # lambda 始终返回1.0


    else: # 测试模型的时候加载lora模型
        config_kwargs = {
            "cache_dir": FL_args.cache_dir,
            "revision": FL_args.model_revision,
            "use_auth_token": True if FL_args.use_auth_token else None,
        }
        if FL_args.modelB_name_or_path:  # 从这里进入
            configB = AutoConfig.from_pretrained(FL_args.modelB_name_or_path, **config_kwargs)
        if FL_args.modelB_name_or_path:
            torch_dtype = (
                FL_args.torch_dtype
                if FL_args.torch_dtype in ["auto", None]
                else getattr(torch, FL_args.torch_dtype)
            )
            logger.info(f"torch_dtype:{torch_dtype}") # torch.float16
            model_B = LlamaForCausalLMB(config=configB, modelA_layers=FL_args.modelA_layers, modelC_layers=FL_args.modelC_layers).to('cpu') # 初始化在CPU

            server_state_dict = torch.load(f'{FL_args.modelB_name_or_path}/pytorch_model_B.bin', map_location=FL_args.device)
            adjusted_B_dict = adjust_modelB_state_dict(server_state_dict, FL_args.modelA_layers, FL_args.modelC_layers)
            model_B.load_state_dict(adjusted_B_dict)
            del server_state_dict
            del adjusted_B_dict
            model_B.to(device)
            torch.cuda.empty_cache()

        if FL_args.lora_modelB_path:
            model_B = PeftModel.from_pretrained(model_B, FL_args.lora_modelB_path, device_map=FL_args.device,trust_remote_code=True)
            model_B = model_B.merge_and_unload()
            logger.warning(f"正加载{FL_args.lora_modelB_path}的checkpoint")
        else:
            logger.warning("没有加载checkpoint!")

        # 加载lora后改变精度，如果model_B的数据类型与目标数据类型不同，则进行转换
        if model_B.dtype != model_torch_dtype:
            logger.info(f"Converting model_B from {model_B.dtype} to {model_torch_dtype}")
            model_B = model_B.to(model_torch_dtype)

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
        tokenizer.pad_token = tokenizer.unk_token
        optimB = None
        scheduleB = None

    end_set_model = time.time()
    model_time_cost = end_set_model - start_set_model
    logger.info(f"server加载模型耗时{model_time_cost:.2f} 秒。")

    '''=====================================================配置FL框架====================================================='''    
    # get client manager
    clientmanager = SimpleClientManager()

    # Define strategy 
    flglm_strategy = SplitFed(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients = FL_args.client_count,  # number of clients
        min_evaluate_clients = FL_args.client_count,
        min_available_clients = FL_args.client_count, # Minimum number of clients that need to be connected to the server before a training round can start,
        on_fit_config_fn = config_client
    )

    logger.info(f"server_args:{FL_args}")

    # get server
    flserver = FL_Server(optimizer= optimB, schedule = scheduleB, ServerModel=model_B, args=FL_args, logger=logger, client_manager=clientmanager, strategy=flglm_strategy, child_conn_server=child_conn_server, attack_decoder_model = attack_decoder_model,attack_optimizer=attack_optimizer,attack_tokenizer=attack_tokenizer)

    # Start server 
    fl.server.start_server(
        server_address=FL_args.server_ip,
        server=flserver,
        config=fl.server.ServerConfig(num_rounds=FL_args.per_client_steps),
        strategy=flglm_strategy,
        client_manager=clientmanager,
        )

    '''=====================================================配置邮箱提醒====================================================='''
    #region
    # # 如果不需要配置邮箱提醒，这小节可以直接用上面这一行来代替
    # sender_mail = 'zzs45622660@qq.com'
    # sender_password = 'rboieudsrpjhddga'
    # # 获取当前脚本的文件名
    # script_name = os.path.basename(__file__)
    # script_abs_path = os.path.abspath(__file__)
    # # 得保证服务器能联网
    # yag = yagmail.SMTP(user = sender_mail, password = sender_password, host = 'smtp.qq.com')
    # time_format = "%Y-%m-%d %H:%M:%S"
    # starttime = time.strftime(time_format, time.localtime())
    # yag.send(to = [sender_mail], subject = f'【{script_name }开始运行】', contents = [f'服务器{FL_args.server_ip}\n{script_abs_path}程序开始运行\n时间:\n{starttime}\n此条消息确保程序能够发邮件提醒'])
    # try:

    #     fl.server.start_server(
    #         server_address=FL_args.server_ip,
    #         server=flserver,
    #         config=fl.server.ServerConfig(num_rounds=FL_args.per_client_steps),
    #         strategy=flglm_strategy,
    #         client_manager=clientmanager,
    #         )

    #     # 登录邮箱提醒程序完成
    #     endtime = time.strftime(time_format, time.localtime())
    #     starttime_cost = datetime.strptime(starttime, time_format)
    #     endtime_cost = datetime.strptime(endtime, time_format)
    #     costtime = endtime_cost - starttime_cost
    #     yag.send(to = [sender_mail], subject = f'【{script_name }运行完成】', contents = [f'服务器{FL_args.server_ip}\n{script_abs_path}程序运行完成啦!\n时间:{endtime},耗时:{costtime}'])
    # except Exception as ex:
    #     print(traceback.format_exc())
    #     endtime = time.strftime(time_format, time.localtime())
    #     error_name=repr(ex)
    #     error_detail=traceback.format_exc()
    #     yag.send(to = [sender_mail], subject = f'【{script_name }异常中断】', contents = [f'服务器{FL_args.server_ip}\n{script_abs_path}程序运行错误!\n错误名称:\n{error_name}\n\n详细错误:\n{error_detail}\n时间:\n{endtime}'])
    #endregion

if __name__ == "__main__":
    # 直接运行
    args = FLparser()
    if args.do_inference and args.do_netdisplay:
        print("正在进行网页推理展示")
        pipe_fd = int(args.pipe)
        child_conn_server = connection.Connection(pipe_fd)
        main(child_conn_server)
        child_conn_server.close()  # 关闭子管道
    else:
        main() 
