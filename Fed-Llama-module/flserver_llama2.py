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
from torch.optim.lr_scheduler import LinearLR
import collections
import flwr as fl
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append('/home/zhangzishuai/TFed-GLM/chatglm-6b')
sys.path.append('./server')
from server.modeling_llama_B import LlamaForCausalLMB

import concurrent.futures
import timeit
from logging import DEBUG, INFO
import logging
from typing import Dict, List, Optional, Tuple, Union

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
from get_dataloader import get_dataset
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
import pickle
#endregion

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# 设置日志文件
logging.basicConfig(filename='/home/zhangzishuai/SplitFederated-LLaMA/log/FedLlama/train_server.log',
                    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                    level=logging.INFO,
                    filemode='w') # 添加这个参数以覆盖已有的日志文件
# 创建一个logger实例
logger = logging.getLogger(__name__)

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]],
    List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, DisconnectRes]],
    List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]],
]

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.is_world_process_zero:
            print('+++++++++++++++++save call back++++++++++++++++')
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )
            kwargs["model"].save_pretrained(checkpoint_folder)

            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)
            return control


def FLparser(): # use glm arguments with argument.py
    parser = argparse.ArgumentParser(description="Flower")
    '''=====================================================model args====================================================='''                    
    # 
    parser.add_argument("--modelB_name_or_path", type=str, default="/home/zhangzishuai/SplitFederated-LLaMA/For_Open_Source/Fed-Llama-module/server",
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
    parser.add_argument("--load_in_bits", type=int, default=8)
    parser.add_argument("--model_revision", type=str, default="main",
                        help="The specific model version to use (can be a branch name, tag name or commit id).")
    parser.add_argument("--use_auth_token", default=False,
                        help="Will use the token generated when running `huggingface-cli login` (necessary to use this script with private models).")
    parser.add_argument("--torch_dtype", type=str, default="float16",
                        choices=["auto", "bfloat16", "float16", "float32"],
                        help="Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the dtype will be automatically derived from the model's weights.")
    parser.add_argument("--lora_modelB_path", type=str, default=None,
                        help="The pretrained model checkpoint for lora weights.")
    parser.add_argument("--quantization_bit", type=int, help="quantization bit",  default=4)


    '''=====================================================training args====================================================='''  
    parser.add_argument("--batch_size", type=int, help="traing batch size", default=1) 
    parser.add_argument("--max_train_samples", type=Optional[int], default=None)
    parser.add_argument("--max_eval_samples", type=Optional[int], default=None)  
    parser.add_argument("--seed", type=int, help="random seed", default=42)
    parser.add_argument("--device",type=str, default = 'cuda')
    parser.add_argument("--max_grad_norm", type=float, help='max grad_clipping norm', default=1.0)
    parser.add_argument("--lr", type=float, help='learning rate', default=1e-6)
    parser.add_argument("--betas", type=tuple, help='(adamhf)optimizer betas', default=(0.9,0.999))
    parser.add_argument("--eps", type=float, help='(adamhf)optimizer eps', default=1e-7)
    parser.add_argument("--weight_decay", type=float, help='(adamhf)optimizer weight decay', default=0.0)
    parser.add_argument("--output_dir", type=str, help = 'output folder path', default=None)
    parser.add_argument("--save_step", type=int, help = 'step to save the prefix encoder', default=20)
    parser.add_argument("--overwrite_output_dir", type=bool, help = 'whether to overwrite the output folder', default=False)
    parser.add_argument("--block_size", type=int, help = 'I dont know yet', default=None)
    parser.add_argument("--max_step", type=int, help='number of max training steps, should be same with server side!', default=10)

    '''=====================================================Fl Arguments====================================================='''  

    parser.add_argument("--prompt", type=str, default=None) 
    parser.add_argument("--pipe", type=str, default=None)    
    parser.add_argument("--max_predict_samples", type=Optional[int], default=None)
    parser.add_argument("--val_max_target_length", type=Optional[int], default=None)
    parser.add_argument("--do_train", action="store_true", help='Whether to run training.', default=False)
    parser.add_argument("--do_BatchParallel_train", action="store_true", help='Whether to run BatchParallel training.', default=False)
    parser.add_argument("--do_eval", action="store_true", help='Whether to run eval on the dev set.', default=False)
    parser.add_argument("--do_predict", action="store_true", help='Whether to run predictions on the test set.', default=False)
    parser.add_argument("--do_inference", action="store_true", help='Whether to inference using prompt from users.', default=False)

    parser.add_argument("--dry", type=bool, default=False, required=False, help="Do a dry-run to check the client")
    parser.add_argument("--client_id", type=int, default=1, choices=range(0, 10), required=False, help="Specifies the artificial data partition of CIFAR10 to be used. \
        Picks partition 0 by default")
    parser.add_argument(  "--toy", action="store_true", help="Set to true to quicky run the client using only 10 datasamples. \
        Useful for testing purposes. Default: False")
    parser.add_argument("--use_cuda", type=bool, default=True, required=False, help="Set to true to use GPU. Default: False")
    parser.add_argument("--model", type=str, default="efficientnet", choices=["efficientnet", "alexnet"],
        help="Use either Efficientnet or Alexnet models. If you want to achieve differential privacy, please use the Alexnet model")
    parser.add_argument("--client_count", type=int, help='number of clients', default=1)
    parser.add_argument("--client_blocks", type=int, help='number of client\'s blocks', default=1)
    parser.add_argument("--per_client_steps", type=int, help="the number of the traing steps of every client", default=1)
    parser.add_argument("--max_output_length", type=int, help='max_output_length.', default=256)
    parser.add_argument("--pred_finished", type=bool, help='pred_finished.', default=False)
    parser.add_argument("--start_inference", type=str, help='display answer when get prompt.', default="OK")
    args = parser.parse_args()

    return args

class FL_Server(Server):
    """Flower server for Split fed learning."""
    def __init__(self, optimizer,schedule,ServerModel, args, client_manager: ClientManager, strategy: Strategy = None, child_conn_server=None) -> None:
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.servermodel = ServerModel
        self.optim = optimizer
        self.schedule = schedule 
        self.model_args = args
        self.child_conn_server = child_conn_server
        global self_device
        self.device = torch.device(self_device)
        if self.model_args.do_train or self.model_args.do_BatchParallel_train:
            self.servermodel.train()
        else:
            self.servermodel.eval()

    def train_server(self, ServerModel, position_ids, attention_mask, hidden_state, past_key_values):
        if hidden_state.dtype != torch.float16:
            hidden_state = hidden_state.half()
        
        feature2 = ServerModel(position_ids = position_ids,
            attention_mask = attention_mask,
            hidden_state = hidden_state,            
            past_key_values = past_key_values) # dict
        
        return feature2
    
    def zero_grad(self, optimizer):
        optimizer.zero_grad()

    def backward(self, feature, gradient):
        feature.backward(gradient)
    
    def save_model(self, step):
        logger.info("Saving Lora")  
        # filtered_state_dict = self.model[0].state_dict()['transformer.prefix_encoder.embedding.weight']

        check_fold = os.path.join(self.model_args.output_dir, "model-B")

        # out_dir = os.path.join(check_fold, 'lora-adapter.bin')

        self.servermodel.save_pretrained(os.path.join(check_fold, 'checkpoint-{}'.format(str(step+1))))
        logger.info("Lora权重保存成功!") 
    
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
        log(INFO, "Initializing fed-split learning!")

        # get number of clients
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=num_rounds,
            parameters=None,
            client_manager=self._client_manager,
        ) 

        if not client_instructions:
            log(INFO, "Caution! No clients selected, cancel and check again!")
            return None
        log(
            DEBUG,
            "strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )
        log(INFO, "Total of {} clients participating.".format(self._client_manager.num_available()))

        # Run split federated learning for num_rounds
        log(INFO, "Split-FL starting!")
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
        log(INFO, "Fed-split learning finished in %s", elapsed)

        # save model
        # torch.save(self.servermodel, 'checkpoint/testcheckmlp/server.pt')

        # aggregate
        # log(INFO, "Start aggregate clients' models parameters!")
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
                att_mask = torch.from_numpy(featureA[1]).to(self.device)
                p_ids = torch.from_numpy(featureA[2]).to(self.device)
                # 前向传播的时候不需要
                pkv=torch.from_numpy(featureA[3]).to(self.device)   # 当batch_size为1 torch.Size([2, 32, 352, 128])
                # TODO: 当batchsize为2时，这里需要改变
                past_key, past_value = torch.split(pkv, 1, dim=0) 
                pkv1= DynamicCache.from_legacy_cache(None)
                pkv1.key_cache.append(past_key)
                pkv1.value_cache.append(past_value)
                pkv1.seen_tokens = torch.from_numpy(featureA[4])  
                # pkv1 = 
                # # reshape
                # pkv1 = pkv1.view(self.model_args.batch_size,self.model_args.pre_seq_len,2*27,32,128).permute([2, 1, 0, 3, 4]).split(2)
                # new = [v.clone().detach().requires_grad_(True) for v in pkv1]# list of tensor:(num_layers*[2, pre_seq_len, batch_size, num_head, head_hidden_size])
                # pkv1 = tuple(new)
                featureB = self.train_server(self.servermodel, 
                                             position_ids = p_ids,
                                             attention_mask = att_mask,
                                             hidden_state = hidden_stateA,            
                                             past_key_values = pkv1) # dict

                hidden_stateB = featureB.last_hidden_state.clone().detach()#.requires_grad_(True)
                att_mask = featureB.attention_mask.clone().detach()
                p_ids = featureB.position_ids.clone().detach()

                # pkv2 = featureB.past_key_values[0].clone().detach()# .requires_grad_(True)
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
                #server_optim.zero_grad() 10/4
                self.optim.zero_grad()
                featureB.last_hidden_state.backward(gradient_server_hidden)
                # print(gradient_last_pkv.dtype) # torch.float16
                # gradient_last_pkv = gradient_last_pkv.to(pkv1[-1].dtype)
                # print(gradient_last_pkv.dtype)
                # print(pkv1[0].grad.dtype)
                #pkv1[-1].grad = gradient_last_pkv
                # print(pkv1[0].grad)
                # server_optim.step()
                nn.utils.clip_grad_norm_(self.servermodel.parameters(), self.model_args.max_grad_norm)
                self.optim.step()
                # schedule.step()
                self.schedule.step()
                # reshape pkv1.grad (tuple of tensor[2,5,1,32,128] to tensor)
                #以下两行原本没有注释
                # p_list=[pkv1[i].grad for i in range(len(pkv1))]
                # pkv1_grad = torch.cat(p_list).permute([2, 1, 0, 3, 4]).reshape(self.model_args.batch_size,self.model_args.pre_seq_len, 2*27*4096)

                # print(pkv1_grad.size())
                # print(hidden_stateA.grad.size())
                # torch.Size([1, 5, 221184])
                # torch.Size([30, 1, 4096])
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
                # log(INFO, "steps end in %s", step_elapsed)
                # log(INFO, "steps %s: loss %s", current_round, fitres_partC.metrics['loss'])
                log(INFO, "steps %s: loss %s", current_round, loss)
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
            log(INFO, "client %d finished in %s", client_id, elapsed)
            client_id += 1

    def BatchParallel_train(self, client_instructions, num_rounds, timeout, start_time, histories):
        # default with BatchParallel trainging
        # with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
        for current_round in range(num_rounds):
            # Initialize tensors to store data from all clients
            hidden_statesA_list = []
            att_masks = []
            p_ids_list = []
            pkvs1_list = []
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
                # hidden_stateA: torch.Size([640, 1, 4096])
                att_mask = torch.from_numpy(featureA[1]).to(self.device)
                # att_mask: torch.Size([1, 1, 640, 768])
                p_ids = torch.from_numpy(featureA[2]).to(self.device)
                # p_ids: torch.Size([1, 2, 640])
                pkv = torch.from_numpy(featureA[3]).to(self.device) # 27 layers in one tensor  #TODAY
                past_key, past_value = torch.split(pkv, 1, dim=0) 
                pkv1= DynamicCache.from_legacy_cache(None)
                pkv1.key_cache.append(past_key)
                pkv1.value_cache.append(past_value)
                pkv1.seen_tokens = torch.from_numpy(featureA[4])  
                # print(f"pkv1.key_cache:{pkv1.key_cache}")
                # print(f"pkv1.key_cache[0]:{pkv1.key_cache[0]}") #pkv1.key_cache是list[tensor]
                # print(f"pkv1.key_cache[0].shape:{pkv1.key_cache[0].shape}")
                # print(f"pkv1.value_cache:{pkv1.value_cache}")
                # print(f"pkv1.seen_tokens:{pkv1.seen_tokens}")

                hidden_statesA_list.append(hidden_stateA)
                att_masks.append(att_mask)
                p_ids_list.append(p_ids)
                pkvs1_list.append(pkv1)
                del hidden_stateA
                del att_mask
                del p_ids
                del pkv1
                
            
            # Concatenate collected data into batch tensors
            hidden_stateA = torch.cat(hidden_statesA_list, dim=0)
            hidden_stateA.retain_grad()
            # hidden_stateA:torch.Size([640, 2, 4096]) dim 1是bacth_size
            att_mask = torch.cat(att_masks, dim=0)
            # att_mask:torch.Size([2, 1, 640, 768]) dim 0是bacth_size
            p_ids = torch.cat(p_ids_list, dim=0)
            # p_ids:torch.Size([2, 2, 640]) dim 0是bacth_size
            key_caches = [pkv1.key_cache[0] for pkv1 in pkvs1_list]
            value_caches = [pkv1.value_cache[0] for pkv1 in pkvs1_list]
            concatenated_key_cache = torch.cat(key_caches, dim=0)
            concatenated_value_cache = torch.cat(value_caches, dim=0)
            # print(f"concatenated_value_cache.shape:{concatenated_value_cache.shape}")

            pkv1= DynamicCache.from_legacy_cache(None)
            pkv1.key_cache.append(concatenated_key_cache)
            pkv1.value_cache.append(concatenated_value_cache)
            pkv1.seen_tokens = torch.from_numpy(featureA[4]) 
            del featureA
            

            '''=====================================================train server model B 前向传播模型B====================================================='''
            featureB = self.train_server(self.servermodel, 
                                            position_ids = p_ids,
                                            attention_mask = att_mask,
                                            hidden_state = hidden_stateA,            
                                            past_key_values = pkv1) # dict
            
            # 用完即清理
            del p_ids
            del att_mask
            del hidden_stateA
            del pkv1

            hidden_stateB = featureB.last_hidden_state.clone().detach()#.requires_grad_(True)
            # torch.Size([640, 2, 4096])
            att_mask = featureB.attention_mask.clone().detach()
            # torch.Size([2, 1, 640, 768])
            p_ids = featureB.position_ids.clone().detach()
            # torch.Size([2, 2, 640])
            pkv2 = featureB.past_key_values

            # print(pkv2.size(0))
            # torch.Size([2, 128, 2, 32, 128])
            # pkv2 = tuple(pkv2)

            gradient_server_list=[]

            '''=====================================================train model C 前向传播模型C并梯度反传====================================================='''
            client_id = 0 # 不一定对应真实的id，这里指的是client_instructions的第一个client
            for client_proxy, ins in client_instructions:
                # 准备隐藏层
                hidden_stateB_i = hidden_stateB[client_id:client_id+1,:,  :]
                # torch.Size([640, 1, 4096])
                att_mask_i = att_mask[client_id:client_id+1, :, :, :]
                # torch.Size([1, 1, 640, 768])
                p_ids_i = p_ids[client_id:client_id+1, :]
                # torch.Size([1, 2, 640])

                past_keys= pkv2.key_cache #list{tensor torch.Size([2, 32, 688, 128]) }
                past_values = pkv2.value_cache
                pkv3=[]
                for past_key in past_keys:
                    past_key = past_key[client_id:client_id+1, :, :, :]
                    pkv3.append(past_key.clone().detach()) 
                for past_value in past_values:
                    past_value = past_value[client_id:client_id+1, :, :, :]    
                    pkv3.append(past_value.clone().detach()) 
                # print(f"len(pkv3):{len(pkv3)}")
                pkv3 = torch.cat(pkv3, dim=0) 
                # print(f"pkv3.shape:{pkv3.shape}")
                seen_tokens = torch.tensor(pkv2.seen_tokens).to(self.device)

                featureB_od = collections.OrderedDict([('hidden_state', hidden_stateB_i),
                                                        ('attention_mask', att_mask_i),
                                                        ('position_ids', p_ids_i),
                                                        ('past_key_values', pkv3),
                                                        ('seen_tokens',seen_tokens)])
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
                # gradient_pkv_list.append(gradient_last_pkv)
                client_real_id = int(torch.from_numpy(gradient_server[1])) #client_real_id是flclient的真实id，对应parser.add_argument("--client_id", type=int, default=0)的值
                # client_id和这里的client_real_id一一对应

                # del gradient_server_hidden
                # del gradient_last_pkv
                # torch.cuda.empty_cache()
               
                # print("server side gradient:", gradient_server)
                # loss = fitres_partC.metrics['loss']
                #log(INFO, "steps %s client %d loss %s", current_round, client_id, fitres_partC.metrics['loss'])
                # logger.info("steps %s client %d loss %s", current_round, client_id, fitres_partC.metrics['loss'])
                logger.info("steps %s client %s loss %s", current_round, client_real_id, loss)
                
                client_id += 1

            gradient_server_hidden = torch.cat(gradient_server_list, dim=0).requires_grad_(True)
            #gradient_server_hidden.retain_grad()  # 7/23新增
            # torch.Size([640, 2, 4096])
            # gradient_last_pkv = torch.cat(gradient_pkv_list, dim=2).requires_grad_(True)
            #gradient_last_pkv.retain_grad()
            # torch.Size([2, 128, 2, 32, 128])

            # del gradient_server_list
            # del gradient_pkv_list
            # torch.cuda.empty_cache()

            '''=====================================================更新server model====================================================='''
            self.optim.zero_grad()
            featureB.last_hidden_state.backward(gradient_server_hidden)
            # print(gradient_last_pkv.dtype) # torch.float16
            # gradient_last_pkv = gradient_last_pkv.to(pkv1[-1].dtype)
            # print(gradient_last_pkv.dtype)
            # print(pkv1[0].grad.dtype)
            # pkv1[-1].grad = gradient_last_pkv
            # print(pkv1[0].grad)
            self.optim.step()
            self.schedule.step()

            # reshape pkv1.grad (tuple of tensor[2,5,1,32,128] to tensor)
            # p_list=[pkv1[i].grad for i in range(len(pkv1))]
            # pkv1_grad = torch.cat(p_list).permute([2, 1, 0, 3, 4]).reshape(self.model_args.client_count,self.model_args.pre_seq_len, 2*27*4096)
            # pkv1_grad.size(): torch.Size([2, 128, 221184])

            # print(hidden_stateA.grad.size())
            # torch.Size([640, 2, 4096])

            if (current_round+1) % self.model_args.save_step == 0:
                # self.save_model(config['current_step'], self.optim, self.schedule)
                self.save_model(current_round)

            client_id = 0
            for client_proxy, ins in client_instructions:
                '''=====================================================更新 client model partA ====================================================='''
                ins.config['type'] = 3
                ins.config['current_step'] = current_round
                hidden_stateA_i = hidden_stateA.grad[client_id:client_id + 1, :, :] #
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
                # log(INFO, "steps end in %s", step_elapsed)
                log(INFO, "steps %s: loss %s", current_round, loss)
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
            log(INFO, "step %d  %d clients finished in %s s",current_round, client_id, elapsed_formatted)
                
                
            # reset fit_ins
            ins.parameters = Parameters(tensors=[], tensor_type="")

    def old_evaluate(self, client_instructions, num_rounds, timeout, start_time, history):
        # default with serial trainging
        
        # with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
        client_id = 1
        for client_proxy, ins in client_instructions:
            # print('fitins before train',ins)            
            for current_round in range(num_rounds):
                # print(type(current_round)) # int
                # eval client part A
                step_time = timeit.default_timer()
                
                ins.config['type'] = 1
                fitres_partA = fit_client_model1(client_proxy, current_round, ins, timeout) # fitres               
                # print(type(feature1.parameters.tensors))  # list of ndarrays
                # print(fitres_partA.metrics) # dict of scalar
                featureA = parameters_to_ndarrays(fitres_partA.parameters)
                # print(feature1) # list of array
                hidden_stateA = torch.from_numpy(featureA[0]).to(self.device)
                hidden_stateA = hidden_stateA.clone().detach().requires_grad_(True)
                att_mask = torch.from_numpy(featureA[1]).to(self.device)
                p_ids = torch.from_numpy(featureA[2]).to(self.device)
                pkv1 = torch.from_numpy(featureA[3]).to(self.device) # 27 layers in one tensor
                # print(pkv1.dtype) # float16
                # reshape
                pkv1 = pkv1.view(self.model_args.batch_size,self.model_args.pre_seq_len,2*27,32,128).permute([2, 1, 0, 3, 4]).split(2)
                new = [v.clone().detach().requires_grad_(True) for v in pkv1]# list of tensor:(num_layers*[2, pre_seq_len, batch_size, num_head, head_hidden_size])
                pkv1 = tuple(new)
                            
                # eval server model
                featureB = self.train_server(self.servermodel, 
                                             position_ids = p_ids,
                                             attention_mask = att_mask,
                                             hidden_state = hidden_stateA,            
                                             past_key_values = pkv1) # dict

                hidden_stateB = featureB.last_hidden_state.clone().detach()#.requires_grad_(True)
                att_mask = featureB.attention_mask.clone().detach()
                p_ids = featureB.position_ids.clone().detach()

                pkv2 = featureB.past_key_values[0].clone().detach()# .requires_grad_(True)
                # print(pkv2.size(0))
                # pkv2 = tuple(pkv2)
                
                featureB_od = collections.OrderedDict([('hidden_state', hidden_stateB),
                                                       ('attention_mask', att_mask),
                                                       ('position_ids', p_ids),
                                                       ('past_key_values', pkv2)])
                feature_array = [val.cpu().numpy() for _, val in featureB_od.items()]
                ins.parameters = ndarrays_to_parameters(feature_array)

                # print('fitins for client part C', ins)
                
                # eval client partC
                ins.config['type'] = 2
                fitres_partC = fit_client_model2(client_proxy, ins, timeout) # featureB's gradient
                gradient_server = parameters_to_ndarrays(fitres_partC.parameters)
                # gradient_server_hidden = torch.from_numpy(gradient_server[0]).cuda()
                # gradient_last_pkv = torch.from_numpy(gradient_server[1]).cuda()
                # print("server side gradient:", gradient_server)
                # loss = fitres_partC.metrics['loss']

                step_end = timeit.default_timer()
                step_elapsed = step_end - step_time
                log(INFO, "steps end in %s", step_elapsed)
                # log(INFO, "steps %s: loss %s", current_round, fitres_partC.metrics['loss'])
                
                # reset fit_ins
                ins.parameters = Parameters(tensors=[], tensor_type="")

                # print('fit ins for next step:',ins) 
                # print(fitres_partC) # FitRes, metrics{}
                # it seams that fitresA=fitresC
                history.add_loss_distributed(current_round, fitres_partC.metrics['loss'])

            # Bookkeeping
            end_time = timeit.default_timer()
            elapsed = end_time - start_time
            log(INFO, "client %d finished in %s", client_id, elapsed)
            client_id += 1

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
                    featureB = self.train_server(self.servermodel,
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
                        logger.info(f"第{current_round+1}个问题回答结束!耗时{time_cost:.2f} 秒。Loss为{fitres_partC.metrics['loss']:.4f}")
                        history.add_loss_distributed(current_round, fitres_partC.metrics['loss'])  
                        self.model_args.start_inference = "NO"               
            # 回答完所有问题
            end_time = timeit.default_timer()
            elapsed = end_time - start_time
            log(INFO, "client %d finished in %s", client_id, elapsed)
            client_id += 1

    def predict(self, client_instructions, num_rounds, timeout, start_time, history):
        client_id = 1
        self.model_args.max_output_length = 128
        for client_proxy, ins in client_instructions:
            current_round=0
            while True:
                if self.child_conn_server is not None:
                    if self.child_conn_server.poll():
                        tmp = self.child_conn_server.recv()
                        self.model_args.start_inference = "OK"
                if self.model_args.start_inference == "OK":
                    print("开始前向传播")
                    need_test_data =True
                    start_time = time.time()
                    for _ in range(self.model_args.max_output_length):
                        ins.config['type'] = 1
                        #fitres_partA_start = time.time() #
                        fitres_partA = fit_client_model1(client_proxy, current_round, ins, timeout, need_test_data)
                        #fitres_partA_end = time.time() #
                        #fitres_partA_cost = fitres_partA_end - fitres_partA_start # 
                        #logger.info(f"接收到fitres_partA_cost:{fitres_partA_cost},此时时间为:{time.time()}") #
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
                        #featureB_start = time.time() #
                        featureB = self.train_server(self.servermodel,
                                                    position_ids = p_ids,
                                                    attention_mask = att_mask,
                                                    hidden_state = hidden_stateA,
                                                    past_key_values = pkv1) # dict
                        #featureB_end = time.time() #
                        #featureB_cost = featureB_end - featureB_start # 
                        #logger.info(f"featureB训练_cost:{featureB_cost}") #
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
                        #fitres_partC_start = time.time() #
                        fitres_partC = fit_client_model2(client_proxy, ins, timeout)  # We should return the prediction result
                        #fitres_partC_end = time.time() #
                        #fitres_partC_cost = fitres_partC_end - fitres_partC_start # 
                        #logger.info(f"接收到fitres_partC_cost:{fitres_partC_cost}") #

                        if fitres_partC.metrics['pred_finished']==1:

                            end_time = time.time()
                            time_cost = end_time - start_time
                            log(INFO, f"第{current_round+1}个问题回答结束!耗时{time_cost:.2f} 秒。")
                            current_round +=1
                            if self.model_args.do_inference:
                                self.model_args.start_inference = "NO"   # 当是测试集时,不变成NO,以便继续测试
                            break



def old_fit_client_model1(
    client: ClientProxy, server_round: int, ins: FitIns, timeout: Optional[float]=None, need_test_data: Optional[bool]= False, group_id: int=None
) -> FitRes:
    """Refine parameters on a single client."""
    ins.config['current_step'] = server_round
    if need_test_data:
        ins.config['need_test_data']=1
    else:
        ins.config['need_test_data']=0
    fit_res = client.fit(ins, timeout, group_id)
    return fit_res

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

def old_fit_client_model2(
    client: ClientProxy, ins: FitIns, timeout: Optional[float]=None, group_id: int=None
) -> FitRes:
    """Refine parameters on a single client."""
    fit_res = client.fit(ins, timeout, group_id)
    # print("fit_res:", fit_res)
    return fit_res

def fit_client_model2(
    client: ClientProxy, ins: FitIns, timeout: Optional[float]=None
) -> FitRes:
    """Refine parameters on a single client."""
    fit_res = client.fit(ins, timeout)
    # print("fit_res:", fit_res)
    return fit_res

def old_back_client_model1(
    client: ClientProxy, ins: FitIns, timeout: Optional[float]=None, group_id: int=None
) -> Tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    # ins.parameter = feature gradient
    
    fit_res = client.fit(ins, timeout, group_id) # 非抽象方法
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

    if FL_args.do_train:
        logger.info("正在进行服务器端训练")
        assert not (FL_args.do_BatchParallel_train or FL_args.do_eval or FL_args.do_predict or FL_args.do_inference), "当 do_train 为 True 时，其它标志必须为 False"
    if FL_args.do_BatchParallel_train:
        FL_args.client_count = 2
        logger.info(f"正在进行服务器端批训练,客户端数：{FL_args.client_count}")
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
        assert not (FL_args.do_train or FL_args.do_BatchParallel_train or FL_args.do_eval), "当 do_inference 为 True 时，do_train, do_BatchParallel_train, do_eval 必须为 False"
        
    global self_device
    self_device = 'cuda:0'
    device = torch.device('cuda:0')
    print(f"device:{device},self_device:{self_device}")

    torch.manual_seed(42)
    FL_args.device = device
    #FL_args.max_step = 15000
    #FL_args.pre_seq_len = 128
    FL_args.lr = 1e-5
    FL_args.per_client_steps = FL_args.max_step
    if type(FL_args.target_modules)==str:
        FL_args.target_modules = FL_args.target_modules.split(',')
    #FL_args.lora_modelB_path = "/home/zhangzishuai/SplitFederated-LLaMA/For_Open_Source/Fed-Llama-module/output/Friends/ReadCompre/model-B/checkpoint-1000"
    
    '''=====================================================配置Server Model====================================================='''
    start_set_model = time.time()

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

        # logger.info(f"FL_args.target_modules:{FL_args.target_modules}")
        t_m_s = []
        for index in FL_args.target_modules:
            for i in range(0,30):
                t_m_s.append(str(f'model.layers.{i}.self_attn' + '.' + index))
        # logger.info(f"t_m_s:{t_m_s}")
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
            model_B = LlamaForCausalLMB(config=configB).to('cpu')
            # for name, module in model_B.named_modules():
            #     logger.info(f"Module: {name}, Type: {type(module)}")

            # for name, param in model_B.named_parameters():
            #     logger.info(f"Parameter Name: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")   
            server_state_dict = torch.load(f'{FL_args.modelB_name_or_path}/pytorch_model.bin', map_location='cuda:0')
            server_state_dict = {f"model.layers.{k}": v for k, v in server_state_dict.items()}
            model_B.load_state_dict(server_state_dict)
            del server_state_dict
            model_B = model_B.to(device)

        else:
            model_B = LlamaForCausalLM.from_config(configB)
            n_params = sum({p.data_ptr(): p.numel() for p in model_B.parameters()}.values())
            logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

        # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
        # on a small vocab and want a smaller embedding size, remove this test.
        # embedding_size = model_B.get_input_embeddings().weight.shape[0]
        # if len(tokenizer) > embedding_size:
        #     model_B.resize_token_embeddings(len(tokenizer))
        if FL_args.load_in_bits==8:
            model_B = prepare_model_for_int8_training(model_B)
        elif FL_args.load_in_bits==4:
            model_B = prepare_model_for_kbit_training(model_B)


        model_B = get_peft_model(model_B, lora_config)
        # model_B.print_trainable_parameters()
        # optimizer初始化 
        optimB = AdamW(filter(lambda p: p.requires_grad, model_B.parameters()), lr=FL_args.lr, betas=FL_args.betas, eps=FL_args.eps, weight_decay=FL_args.weight_decay)
        # schedule 初始化
        scheduleB = LinearLR(optimB, start_factor=1.0, end_factor=0.0, total_iters=FL_args.max_step)

    else: # 不训练模型的时候加载lora模型
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
            model_B = LlamaForCausalLMB(config=configB).to('cpu') # 初始化在CPU
            #model_B = LlamaForCausalLMB(config=configB).to('cuda:1') # 初始化在GPU 
            # for name, module in model_B.named_modules():
            #     logger.info(f"Module: {name}, Type: {type(module)}")

            # for name, param in model_B.named_parameters():
            #     logger.info(f"Parameter Name: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")   
            server_state_dict = torch.load(f'{FL_args.modelB_name_or_path}/pytorch_model.bin', map_location='cuda:0')
            server_state_dict = {f"model.layers.{k}": v for k, v in server_state_dict.items()}
            model_B.load_state_dict(server_state_dict)
            del server_state_dict 
            model_B.to(device)

        if FL_args.lora_modelB_path:
            model_B = PeftModel.from_pretrained(model_B, FL_args.lora_modelB_path, device_map='auto',trust_remote_code=True)
            model_B = model_B.merge_and_unload()
            logger.warning(f"正加载{FL_args.lora_modelB_path}=的checkpoint")
        else:
            logger.warning("没有加载checkpoint!")

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
    log(INFO, f"server加载模型耗时{model_time_cost:.2f} 秒。")

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
    flserver = FL_Server(optimizer= optimB, schedule = scheduleB, ServerModel=model_B, args=FL_args, client_manager=clientmanager, strategy=flglm_strategy, child_conn_server=child_conn_server)

    # Start server 
    fl.server.start_server(
        server_address="10.143.12.74:8080",
        server=flserver,
        config=fl.server.ServerConfig(num_rounds=FL_args.per_client_steps),
        strategy=flglm_strategy,
        client_manager=clientmanager,
        )

if __name__ == "__main__":
    # 直接运行
    main() 
    
    # 网页展示
    # args = parameters1()
    # pipe_fd = int(args.pipe)
    # child_conn_server = connection.Connection(pipe_fd)
    # main(child_conn_server)
    # child_conn_server.close()  # 关闭子管道

 # python fed-glm-module/flserver.py   