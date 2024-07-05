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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/zhengjiaying/project/TFed-GLM/chatglm-6b')
sys.path.append('/home/zhengjiaying/project/TFed-GLM/fed-glm-module/client')
sys.path.append('/home/zhengjiaying/project/TFed-GLM/fed-glm-module/client_part3')
sys.path.append('/home/zhengjiaying/project/TFed-GLM/fed-glm-module/server')

import concurrent.futures
import timeit
from logging import DEBUG, INFO
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
import torch
from  sfglm_strategy import SplitFed

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)

from server.server_model import ChatGLMForConditionalGenerationServerSide



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


class FL_Server(Server):
    """Flower server for Split fed learning."""
    def __init__(self, ServerModel, args, client_manager: ClientManager, strategy: Strategy = None) -> None:
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.servermodel = ServerModel
        self.model_args = args
        # self.global_models = global_models

    def train_server(self, 
                     ServerModel, 
                     position_ids,
            attention_mask,
            hidden_state,            
            past_key_values):
        
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
    
    def fit(self, num_rounds: int, timeout: Optional[float]=None): 
        """Run split federated learning with clients."""
        history = History()

        # Initialize parameters
        log(INFO, "Initializing fed-split learning!")

        # get number of clients
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=num_rounds,
            parameters=None,
            client_manager=self._client_manager,
        ) # [(client, fit_ins)]


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
        # default with serial trainging
        server_optim = AdamW(self.servermodel.parameters(), lr=self.model_args.lr, betas=self.model_args.betas,eps=self.model_args.eps, weight_decay=self.model_args.weight_decay)
        
        schedule = LinearLR(server_optim, start_factor=1.0, end_factor=0.0, total_iters=self.model_args.max_step)# num_rounds == num of traing steps of each client
        # with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
        client_id = 1
        for client_proxy, ins in client_instructions:       
            for current_round in range(num_rounds):
                step_time = timeit.default_timer()
                
                ins.config['type'] = 1
                fitres_partA = fit_client_model1(client_proxy, current_round, ins, timeout) # fitres               

                featureA = parameters_to_ndarrays(fitres_partA.parameters)

                hidden_stateA = torch.from_numpy(featureA[0]).cuda()                                          
                hidden_stateA = hidden_stateA.clone().detach().requires_grad_(True)
                att_mask = torch.from_numpy(featureA[1]).cuda()
                p_ids = torch.from_numpy(featureA[2]).cuda()
                pkv1 = torch.from_numpy(featureA[3]).cuda() 

                pkv1 = pkv1.view(self.model_args.batch_size,self.model_args.pre_seq_len,2*27,32,128).permute([2, 1, 0, 3, 4]).split(2)
                new = [v.clone().detach().requires_grad_(True) for v in pkv1]
                pkv1 = tuple(new)
                            
                # train server model
                featureB = self.train_server(self.servermodel, 
                                             position_ids = p_ids,
                                             attention_mask = att_mask,
                                             hidden_state = hidden_stateA,            
                                             past_key_values = pkv1) # dict

                hidden_stateB = featureB.last_hidden_state.clone().detach()#.requires_grad_(True)
                att_mask = featureB.attention_mask.clone().detach()
                p_ids = featureB.position_ids.clone().detach()

                pkv2 = featureB.past_key_values[0].clone().detach()# .requires_grad_(True)
                
                featureB_od = collections.OrderedDict([('hidden_state', hidden_stateB),
                                                       ('attention_mask', att_mask),
                                                       ('position_ids', p_ids),
                                                       ('past_key_values', pkv2)])
                feature_array = [val.cpu().numpy() for _, val in featureB_od.items()]
                ins.parameters = ndarrays_to_parameters(feature_array)

                # print('fitins for client part C', ins)
                
                # train client partC
                ins.config['type'] = 2
                fitres_partC = fit_client_model2(client_proxy, ins, timeout) # featureB's gradient
                gradient_server = parameters_to_ndarrays(fitres_partC.parameters)
                gradient_server_hidden = torch.from_numpy(gradient_server[0]).cuda()
                gradient_last_pkv = torch.from_numpy(gradient_server[1]).cuda()
                # print("server side gradient:", gradient_server)
                # loss = fitres_partC.metrics['loss']

                # step server model
                server_optim.zero_grad()

                featureB.last_hidden_state.backward(gradient_server_hidden)

                pkv1[-1].grad = gradient_last_pkv
                server_optim.step()
                schedule.step()

                # reshape pkv1.grad (tuple of tensor[2,5,1,32,128] to tensor)
                p_list=[pkv1[i].grad for i in range(len(pkv1))]
                pkv1_grad = torch.cat(p_list).permute([2, 1, 0, 3, 4]).reshape(self.model_args.batch_size,self.model_args.pre_seq_len, 2*27*4096)

                # backfit client model partA 
                ins.config['type'] = 3
                ins.config['current_step'] = current_round
                gradient_od = collections.OrderedDict([('hidden_gradient', hidden_stateA.grad),
                                                       ('past_key_value_gradient',pkv1_grad)]) # it's a tuple of tensor need change to tensor!
                gradient =  [val.cpu().numpy() for _, val in gradient_od.items()]
                ins.parameters = ndarrays_to_parameters(gradient)

                # print('fitins for backward client model A')

                _ = back_client_model1(client_proxy, ins, timeout)
                step_end = timeit.default_timer()
                step_elapsed = step_end - step_time
                # log(INFO, "steps end in %s", step_elapsed)
                log(INFO, "steps %s: loss %s", current_round, fitres_partC.metrics['loss'])
                
                # reset fit_ins
                ins.parameters = Parameters(tensors=[], tensor_type="")

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
        for client_proxy, ins in client_instructions:
            # print('fitins before train',ins)            
            for current_round in range(num_rounds):
                # eval client part A
                step_time = timeit.default_timer()
                
                ins.config['type'] = 1
                fitres_partA = fit_client_model1(client_proxy, current_round, ins, timeout) # fitres               
                featureA = parameters_to_ndarrays(fitres_partA.parameters)
                hidden_stateA = torch.from_numpy(featureA[0]).cuda()                                          
                hidden_stateA = hidden_stateA.clone().detach().requires_grad_(True)
                att_mask = torch.from_numpy(featureA[1]).cuda()
                p_ids = torch.from_numpy(featureA[2]).cuda()
                pkv1 = torch.from_numpy(featureA[3]).cuda() # 27 layers in one tensor

                # reshape
                pkv1 = pkv1.view(self.model_args.batch_size,self.model_args.pre_seq_len,2*27,32,128).permute([2, 1, 0, 3, 4]).split(2)
                new = [v.clone().detach().requires_grad_(True) for v in pkv1]
                pkv1 = tuple(new)
                            
                # eval server model
                featureB = self.train_server(self.servermodel, 
                                             position_ids = p_ids,
                                             attention_mask = att_mask,
                                             hidden_state = hidden_stateA,            
                                             past_key_values = pkv1) 

                hidden_stateB = featureB.last_hidden_state.clone().detach()#.requires_grad_(True)
                att_mask = featureB.attention_mask.clone().detach()
                p_ids = featureB.position_ids.clone().detach()

                pkv2 = featureB.past_key_values[0].clone().detach()# .requires_grad_(True)

                
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

                step_end = timeit.default_timer()
                step_elapsed = step_end - step_time
                log(INFO, "steps end in %s", step_elapsed)
                # log(INFO, "steps %s: loss %s", current_round, fitres_partC.metrics['loss'])
                
                # reset fit_ins
                ins.parameters = Parameters(tensors=[], tensor_type="")


                history.add_loss_distributed(current_round, fitres_partC.metrics['loss'])

            # Bookkeeping
            end_time = timeit.default_timer()
            elapsed = end_time - start_time
            log(INFO, "client %d finished in %s", client_id, elapsed)
            client_id += 1

    def predict(self, client_instructions, num_rounds, timeout, start_time, history):
        client_id = 1
        self.model_args.max_output_length = 64
        if self.model_args.do_inference and self.model_args.do_predict:
            num_rounds=1
        for client_proxy, ins in client_instructions:
            # print('fitins before train',ins)
            for current_round in range(num_rounds):
                need_test_data =True
                for _ in range(self.model_args.max_output_length):

                    step_time = timeit.default_timer()
                    ins.config['type'] = 1
                    fitres_partA = fit_client_model1(client_proxy, current_round, ins, timeout, need_test_data)
                    featureA = parameters_to_ndarrays(fitres_partA.parameters)
                    hidden_stateA = torch.from_numpy(featureA[0]).cuda()
                    hidden_stateA = hidden_stateA.clone().detach().requires_grad_(True)
                    att_mask = torch.from_numpy(featureA[1]).cuda()
                    p_ids = torch.from_numpy(featureA[2]).cuda()
                    pkv1 = torch.from_numpy(featureA[3]).cuda() # 27 layers in one tensor
                    # reshape
                    pkv1 = pkv1.view(self.model_args.batch_size,self.model_args.pre_seq_len,2*27,32,128).permute([2, 1, 0, 3, 4]).split(2)
                    new = [v.clone().detach().requires_grad_(True) for v in pkv1]
                    pkv1 = tuple(new)

                    # predict server model
                    featureB = self.train_server(self.servermodel,
                                                position_ids = p_ids,
                                                attention_mask = att_mask,
                                                hidden_state = hidden_stateA,
                                                past_key_values = pkv1) # dict

                    hidden_stateB = featureB.last_hidden_state.clone().detach()#.requires_grad_(True)
                    att_mask = featureB.attention_mask.clone().detach()
                    p_ids = featureB.position_ids.clone().detach()

                    pkv2 = featureB.past_key_values[0].clone().detach()  # .requires_grad_(True)

                    featureB_od = collections.OrderedDict([('hidden_state', hidden_stateB),
                                                        ('attention_mask', att_mask),
                                                        ('position_ids', p_ids),
                                                        ('past_key_values', pkv2)])
                    feature_array = [val.cpu().numpy() for _, val in featureB_od.items()]
                    ins.parameters = ndarrays_to_parameters(feature_array)
                    # print('fitins for client part C', ins)

                    # predict client partC
                    ins.config['type'] = 2
                    need_test_data = False
                    fitres_partC = fit_client_model2(client_proxy, ins, timeout)  # We should return the prediction result
                    if fitres_partC.metrics['pred_finished']==1:
                        print(f"Question {current_round+1} sovled！")
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
    return fit_res

def back_client_model1(
    client: ClientProxy, ins: FitIns, timeout: Optional[float]=None
) -> Tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    # ins.parameter = feature gradient
    
    fit_res = client.fit(ins, timeout) 
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

def parameters1():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_blocks", type=int, help='number of client\'s blocks', default=1)
    parser.add_argument("--per_client_steps", type=int, help="the number of the traing steps of every client", default=1)
    
     # model args
    parser.add_argument("--quantization_bit", type=int, help="quantization bit",  default=4)
    parser.add_argument("--pre_seq_len", type=int, help="length of p-tuning v2 prefix sequence ", default=5)     
    parser.add_argument("--batch_size", type=int, help="traing batch size", default=1) 
    # training args
    parser.add_argument("--seed", type=int, help="random seed", default=42)
    parser.add_argument("--device",type=str, default = 'cuda')
    parser.add_argument("--max_grad_norm", type=float, help='max grad_clipping norm', default=1.0)
    parser.add_argument("--lr", type=float, help='learning rate', default=1e-6)
    parser.add_argument("--betas", type=tuple, help='(adamhf)optimizer betas', default=(0.9,0.999))
    parser.add_argument("--eps", type=float, help='(adamhf)optimizer eps', default=1e-7)
    parser.add_argument("--weight_decay", type=float, help='(adamhf)optimizer weight decay', default=0.0)
    parser.add_argument("--max_step", type=int, help='number of max training steps, should be same with serve side!', default=10)
    # traing state, should be same with client!
    parser.add_argument("--do_train", type=bool, help='Whether to run training.', default=False)
    parser.add_argument("--do_eval", type=bool, help='Whether to run eval on the dev set.', default=False)
    parser.add_argument("--do_predict", type=bool, help='Whether to run predictions on the test set.', default=False)
    parser.add_argument("--max_output_length", type=int, help='max_output_length.', default=256)
    parser.add_argument("--pred_finished", type=bool, help='pred_finished.', default=False)
    args = parser.parse_args()

    return args

def main():
    torch.manual_seed(42)
    # get train parameters
    args = parameters1()
    args.do_train = True
    args.do_eval = False
    args.do_predict = False
    args.max_step = 30000    
    args.pre_seq_len = 128
    args.lr = 2e-2
    args.per_client_steps = args.max_step

    # SET SERVER MODEL
    # modelB config and initialize
    server_config = AutoConfig.from_pretrained('./server/', trust_remote_code=True)
    server_config.pre_seq_len = args.pre_seq_len # 
    server_config.prefix_projection = False
    server_model = ChatGLMForConditionalGenerationServerSide(config=server_config).cuda()
    server_state_dict = torch.load('./server/server_model_param.bin')
    server_model.load_state_dict(server_state_dict)

    if args.quantization_bit is not None:
        server_model = server_model.quantize(args.quantization_bit)
    if args.pre_seq_len is not None:
        server_model = server_model.half().cuda()

    # get client manager
    clientmanager = SimpleClientManager()


    # Define strategy 
    flglm_strategy = SplitFed(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,  # number of clients
        min_evaluate_clients=1,
        min_available_clients=1,
        on_fit_config_fn = config_client
    )

    # get server
    flserver = FL_Server(ServerModel=server_model, args=args,  client_manager=clientmanager, strategy=flglm_strategy)

    # Start server 
    fl.server.start_server(
        server_address="10.143.12.73:8080",
        server=flserver,
        config=fl.server.ServerConfig(num_rounds=args.per_client_steps),
        strategy=flglm_strategy,
        client_manager=clientmanager,
        )

if __name__ == "__main__":
    main()

 # python fed-glm-module/flserver.py   
