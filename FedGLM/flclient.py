import json
import sys 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.optimization import AdamW
from torch.optim.lr_scheduler import LinearLR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('chatglm-6b path')
sys.path.append('./client')
sys.path.append('./client_part3')

from client_part3.client_model_partC import ChatGLMForConditionalGenerationClientSideC
from client.client_model_partA import ChatGLMForConditionalGenerationClientSide

import argparse
import jieba 
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

# import utils
import flwr as fl

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from get_datalist import get_data_list

# from client.client_model import ModelforTest, ChatGLMForConditionalGenerationClirntSide

# from torch.utils.data import DataLoader

class Fed_Client(NumPyClient):
    def __init__(self, datalist, optimizer, schedule, models, tokenizer, model_args) -> None:
        super(Fed_Client, self).__init__()

        self.datalist = datalist
        self.len_dataset = len(datalist[0])
        self.model = models  # [ModelA, ModelB]
        self.optim = optimizer
        self.schedule = schedule 
        self.label = None
        self.all_label=[]
        self.f1 = None
        self.model_args = model_args
        self.train_metricx = {'step':[], 'loss':[]}
        self.eval_output = {'pred_id':[]}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data = None
        self.total_pred=[]

        if self.model_args.do_train:
            self.model[0].train()
            self.model[1].train()
        else:
            self.model[0].eval()
            self.model[1].eval()
            self.tokenizer = tokenizer


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
    

    def fit(self, parameters: NDArrays=None, config: Dict[str, Scalar]=None):
        feature = []
        num_sample = 0
        metrics = {}
        if config['type'] == 1:
            # print('fit client part A')
            feature = self.fit_partA(current_step=config['current_step'], need_test_data=config['need_test_data'])
            #feature = self.fit_partA(config['current_step'])
            
            metrics = {}
        
        elif config['type']==2:
            # print('fit client part C')
            #feature, num_sample, metrics = self.fit_partC(parameters, config) # feature is gradient
            if self.model_args.do_predict:
                feature, num_sample, metrics = self.fit_partC(parameters, config)

            else:
                feature, num_sample, metrics = self.fit_partC(parameters, config)

        elif config['type']==3:
            # print('backward client part A')
            feature, num_sample, metrics = self.backward_modelA(parameters, config)

        return feature, num_sample, metrics
    
    def fit_partA(self, parameters: NDArrays=None, current_step=0, need_test_data=0):
        # (data, label) =self.datalist[current_step]

        if self.model_args.do_train:
            current_epoch = current_step // self.len_dataset
            current_batch =current_step % self.len_dataset
            data = self.datalist[current_epoch][current_batch]['input_ids'].cuda()
            label = self.datalist[current_epoch][current_batch]['labels'].cuda()
        elif self.model_args.do_eval:
            data = self.datalist[current_step]['input_ids'].cuda()
            label = self.datalist[current_step]['labels'].cuda()
        else:
            if need_test_data ==1:
                data = self.datalist[current_step]['input_ids'].cuda()
                # 6/26新增 data形状  torch.Size([1, 128])
                self.data = data
                label = None
            else:
                data = self.data.cuda()
                label = None

        # forward pass
        f1 = self.model[0](data)
        hidden1 = f1.last_hidden_state.clone().detach() #.requires_grad_(True)
        att_mask = f1.attention_mask.clone().detach()
        p_ids = f1.position_ids.clone().detach()
        p_list = []
        for i in range(len(f1.past_key_values)):
            s = f1.past_key_values[i]
            s = s.clone().detach() #.requires_grad_(True)
            p_list.append(s) # list
        # reshape to one tensor
        pkv = torch.cat(p_list).permute([2, 1, 0, 3, 4]).reshape(self.model_args.batch_size,self.model_args.pre_seq_len,2*27*4096)
        
        # self.get_parameters(ff1) # send feature of modle1 to server
        self.label = label
        self.f1 = f1
        

        od = collections.OrderedDict([('hidden_stateA', hidden1),
                                      ('attention_mask', att_mask),
                                      ('position_ids', p_ids),
                                      ('past_key_values', pkv)])
        feature = [val.cpu().numpy() for _, val in od.items()]


        return feature # Feature doesn't contain gradients！

    def fit_partC(self, parameters: NDArrays=None, config: Dict[str, Scalar]=None):        
        smashed_data = parameters
        past_key_values = torch.from_numpy(smashed_data[3]).cuda() 
        past_key_values = past_key_values.clone().detach().requires_grad_(True)
        hidden = torch.from_numpy(smashed_data[0]).cuda().requires_grad_(True)
        att_mask = torch.from_numpy(smashed_data[1]).cuda()
        p_ids = torch.from_numpy(smashed_data[2]).cuda()
        

        pkv2 = ([past_key_values])
        final_output = self.model[1](position_ids = p_ids,
                            attention_mask = att_mask,
                            hidden_state = hidden,
                            past_key_values = pkv2, 
                            labels = self.label) 
        
        metrics = {}
        gradient = []
        labels = []
        if self.model_args.do_train:
            print('DO_TRAIN')
            # log loss
            loss = final_output[0]
            # print(loss.item())
            log(INFO, 'step %s lr %s  loss %s:', config["current_step"], self.optim[0].param_groups[0]['lr'], loss.item())
            # backward        
            self.optim[1].zero_grad()
            loss.backward()  # model3 step
            # print(f1.grad)
        
            nn.utils.clip_grad_norm_(self.model[1].parameters(), self.model_args.max_grad_norm)
            self.optim[1].step()
            self.schedule[1].step()

            hidden_gradient = hidden.grad # tensor
            pkv_grad_last = pkv2[0].grad
            od = collections.OrderedDict([('hiddden_gradient', hidden_gradient),
                                      ('pkv_grad_last', pkv_grad_last)])
            gradient = [val.cpu().numpy() for _, val in od.items()]

        
            metrics['loss'] = loss.item()

            self.train_metricx['step'].append(config['current_step'])
            self.train_metricx['loss'].append(loss.item())

        elif self.model_args.do_predict:
            metrics['pred_finished'] = 0
            if len(self.total_pred)==0:
                print('DO_PREDICT')
                
            logits=final_output.logits[:,-1,:] #(B，vocab size)
            probs = F.softmax(logits, dim=-1)
            pred_ids = self.sample_text(probs)
            self.total_pred.append(int(pred_ids[0][0]))
            print(f"This is pred_ids {len(self.total_pred)}:{pred_ids}")
            self.data = self.concatenate_output(self.data, pred_ids)

            self.model_args.max_output_length=64
            if pred_ids[0][-1] == 130005 or len(self.total_pred) == self.model_args.max_output_length:
                tokenizer = AutoTokenizer.from_pretrained('chatglm-6b', trust_remote_code=True)
                response = tokenizer.decode(self.total_pred)
                print(f"response:{response}")
                self.total_pred=[]
                metrics['pred_finished'] = 1



        else:
            log(INFO, 'finish step %s', config["current_step"])
            probs = F.softmax(final_output[1], dim=-1)
            pred_ids = torch.argmax(probs, dim=-1)

            self.eval_output['pred_id'].append(pred_ids.cpu())
            self.all_label.append(self.label)
            if config['current_step'] == self.model_args.max_step-1 :
                # compute metrix
                pred_ids = torch.cat(self.eval_output['pred_id']).cpu().tolist()
                labels_all = torch.cat(self.all_label).cpu().tolist()

                score = compute_metrics(pred_ids, labels_all, self.tokenizer, self.model_args)
                print(score)
                print('Now saving evalute metrix!')
                print('Evaluation finish!')



        return gradient, 0, metrics  # GetParametersRes

    def sample_text(self, probs, temperature=0.95, top_p=0.7):
        probs = probs.pow(1 / temperature)

        # sort by probability
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        
        # compute accumulated probs
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # remove token under the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0  
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        probs[indices_to_remove] = 0
        probs = probs / probs.sum(dim=-1, keepdim=True)

        # sample
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token

    def concatenate_output(self, input_tensor, output_tensor, pad_token=3):
        batch_size, seq_len = input_tensor.size()
        non_padding_input = input_tensor[input_tensor != pad_token].view(batch_size, -1)
        # concate output with input
        new_input = torch.cat((non_padding_input, output_tensor), dim=1)

        # re-padding
        pad_length = seq_len - new_input.size(1)  
        padded_input = torch.nn.functional.pad(new_input, (pad_length, 0), value=pad_token)
        
        return padded_input

    def backward_modelA(self, parameters: NDArrays=None, config: Dict[str, Scalar]=None):
        # ff1_grad = self.set_parameters(args.ff1_grad) # recieve gradient of ff1 from server

        hidden_gradient = torch.from_numpy(parameters[0]).cuda()
        pkv_gradient = torch.from_numpy(parameters[1]).cuda() # tensor([batch_size,pre_seq_len, 2*27*4096])
        # reshape pkv_gradient
        # pkv_gradient = pkv_gradient.unsqueeze(0).expand(self.model_args.batch_size, -1, -1) # batch size==1
        pkv_gradient = pkv_gradient.view(self.model_args.batch_size,self.model_args.pre_seq_len, 2*27,32,128).permute([2, 1, 0, 3, 4]).split(2)
        newpkv_gradient = [torch.stack([v[0],v[1]]) for v in pkv_gradient]
        self.optim[0].zero_grad()

        for i in range(27):
            self.f1.past_key_values[i].backward(newpkv_gradient[i], retain_graph=True) 

        self.f1.last_hidden_state.backward(hidden_gradient)
        
        # Gradient Clipping
        nn.utils.clip_grad_norm_(self.model[0].parameters(), self.model_args.max_grad_norm)
        self.optim[0].step()
        self.schedule[0].step()

        # if config['current_step'] != 0:
        if (config['current_step']+1) % self.model_args.save_step == 0:
            self.save_model(config['current_step'], self.optim, self.schedule)
        
        if config['current_step'] == self.model_args.max_step-1 :
            print('Now saving training metrix!')
            json_save_path = os.path.join(self.model_args.output_dir,'loss.json')

            with open(json_save_path, 'w') as f:
                json.dump(self.train_metricx, f)

        return [], 0, {}

    def save_model(self, step, optimizer, schedule):
        # torch.save(self.model[0], 'checkpoint/testcheckmlp/client_part1.pt')
        print("Saving PrefixEncoder")  
        filtered_state_dict = self.model[0].state_dict()['transformer.prefix_encoder.embedding.weight']
        # schedule_state_A = schedule[0].state_dict()
        # schedule_state_C = schedule[1].state_dict()
        # optimizer_state_A = optimizer[0].state_dict()
        # optimizer_state_C = optimizer[1].state_dict()

        # filtered_state_dict = {}
        # for k, v in self.model[0].named_parameters():
        #     if v.requires_grad:
        #         print("requires_grad parameter:",k)
        #         filtered_state_dict[k] = state_dict[k]

        check_fold = os.path.join(self.model_args.output_dir, 'checkpoint-{}'.format(str(step+1)))
        if not os.path.exists(check_fold):
             os.makedirs(check_fold) 
        out_dir_p = os.path.join(check_fold, 'splitfed-ptuning.bin')

        if self.model_args.freeze_head == False:
            out_dir_h = os.path.join(check_fold, 'splitfed-ptuning-head.bin')
            head_state_dict = self.model[1].state_dict()['lm_head.weight']
            torch.save(head_state_dict, out_dir_h)

        torch.save(filtered_state_dict, out_dir_p)


    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        #self.set_parameters(parameters)

        #loss, accuracy = test_client(self.model, self.valloader, self.device)
        loss=0.00
        accuracy=0

        return float(loss), len(self.valloader), {"accuracy": accuracy}




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

def FLparser(): # use glm arguments with argument.py
    # Parse command line argument `partition`
    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--dry", type=bool, default=False, required=False, help="Do a dry-run to check the client")
    parser.add_argument("--client-id", type=int, default=0, choices=range(0, 10), required=False, help="Specifies the artificial data partition of CIFAR10 to be used. \
        Picks partition 0 by default")
    parser.add_argument(  "--toy", action="store_true", help="Set to true to quicky run the client using only 10 datasamples. \
        Useful for testing purposes. Default: False")
    parser.add_argument("--use_cuda", type=bool, default=False, required=False, help="Set to true to use GPU. Default: False")
    parser.add_argument("--model", type=str, default="efficientnet", choices=["efficientnet", "alexnet"],
        help="Use either Efficientnet or Alexnet models. \
             If you want to achieve differential privacy, please use the Alexnet model")
    
    parser.add_argument("--per_client_steps", type=int, help="the number of the traing steps of every client", default=100)
    parser.add_argument("--ptuning_checkpoint", type=str, help="ptuning_checkpoint", 
                        default=None)
    # model args
    parser.add_argument("--quantization_bit", type=int, help="quantization bit",  default=4)
    parser.add_argument("--pre_seq_len", type=int, help="length of p-tuning v2 prefix sequence ", default=5)     
    parser.add_argument("--batch_size", type=int, help="traing batch size", default=1) 
    parser.add_argument("--freeze_head",type=bool, help='Whether to freeze the lm_head layer.', default=False)
    # training args
    parser.add_argument("--seed", type=int, help="random seed", default=42)
    parser.add_argument("--device",type=str, default = 'cuda')
    parser.add_argument("--max_grad_norm", type=float, help='max grad_clipping norm', default=1.0)
    parser.add_argument("--lr", type=float, help='learning rate', default=1e-6)
    parser.add_argument("--betas", type=tuple, help='(adamhf)optimizer betas', default=(0.9,0.999))
    parser.add_argument("--eps", type=float, help='(adamhf)optimizer eps', default=1e-7)
    parser.add_argument("--weight_decay", type=float, help='(adamhf)optimizer weight decay', default=0.0)
    parser.add_argument("--output_dir", type=str, help = 'output folder path', default='/home/zhengjiaying/project/TFed-GLM/checkpoint/default')
    parser.add_argument("--save_step", type=int, help = 'step to save the prefix encoder', default=1)
    parser.add_argument("--max_step", type=int, help='number of max training steps, should be same with serve side!', default=10)
    parser.add_argument("--do_train", type=bool, help='Whether to run training.', default=False)
    parser.add_argument("--do_eval", type=bool, help='Whether to run eval on the dev set.', default=False)
    parser.add_argument("--do_predict", type=bool, help='Whether to run predictions on the test set.', default=False)
    # data arguments
    parser.add_argument("--data_fold", type=Optional[str], default=None)
    parser.add_argument("--cache_dir", type=Optional[str], default=None)
    parser.add_argument("--use_auth_token", type=bool, default=True)
    parser.add_argument("--max_source_length", type=int, default = 128)
    parser.add_argument("--max_target_length", type=int, default = 512)
    parser.add_argument("--source_prefix", type=Optional[str], default=None)
    parser.add_argument("--passage_column", type=Optional[str], default=None)
    parser.add_argument("--passage2_column", type=Optional[str], default=None)
    parser.add_argument("--premise_column", type=Optional[str], default=None)
    parser.add_argument("--question_column", type=Optional[str], default=None)
    parser.add_argument("--answer_column", type=Optional[str], default=None) 
    parser.add_argument("--history_column", type=Optional[str], default=None) 
    parser.add_argument("--preprocessing_num_workers", type=int, default=1)
    parser.add_argument("--overwrite_cache", type=Optional[str], default=None) 
    parser.add_argument("--ignore_pad_token_for_loss", type=bool, default=True)

    parser.add_argument("--dataloader_num_workers", type=int, default=1)
    parser.add_argument("--dataloader_pin_memory", type=bool, default=True)
    parser.add_argument("--dataloader_drop_last", type=bool, default=True)
    
    parser.add_argument("--max_train_samples", type=Optional[int], default=None)
    parser.add_argument("--max_eval_samples", type=Optional[int], default=None)  
    parser.add_argument("--max_predict_samples", type=Optional[int], default=None)
    parser.add_argument("--val_max_target_length", type=Optional[int], default=None)
    
    parser.add_argument("--train_file", type=str,  default='train.json')
    parser.add_argument("--validation_file", type=str,  default='val.json') 
    parser.add_argument("--test_file", type=str,  default='val.json')
    
    args = parser.parse_args()

    return args


def compute_metrics(preds, labels, tokenizer, args):
        # preds, labels = eval_preds  # batched token list 

        if isinstance(preds, tuple): 
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

         # FOR BOOLQ!!!
        for i in range(len(decoded_labels)):
            if decoded_preds[i]=='':
                decoded_preds[i]='Yes' if decoded_labels[i]=='No' else 'No'

        print('predictions:', preds)
        print('decode_predictions:', decoded_preds)
        print('labels:', labels)
        print('decode_labels:', decoded_labels)

        score_dict = {
            "rouge-1": [],
            "rouge-2": [],
            "rouge-l": [],
            "bleu-4": [] ,
            # "Accuracy": [],
            # "Confusion_matrix":[],
            # "F1-Score":[] 
        }
        for pred, label in zip(decoded_preds, decoded_labels):
            # ...input
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            rouge = Rouge()
            scores = rouge.get_scores(' '.join(hypothesis) , ' '.join(reference))
            result = scores[0]
            
            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))
            # score_dict["Accuracy"].append(round(acc * 100, 4))

        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))

        # score_dict["Accuracy"].append(acc)
        # score_dict["Confusion_matrix"].append(cm.tolist)
        # score_dict["F1-Score"]= round(f1score* 100, 4)
                
        return score_dict

def main() -> None:
    torch.manual_seed(42)
    # get parse
    modelA_args = FLparser()
    # modelA_args.do_train = True
    # modelA_args.do_eval = False
    # modelA_args.do_predict = False
    modelA_args.do_train = False
    modelA_args.do_eval = False
    modelA_args.do_predict = True

    modelA_args.lr = 2e-2
    modelA_args.data_fold = './data/QA/huatuo'
    modelA_args.output_dir = ' your output_dir path'
    modelA_args.max_step = 30000
    modelA_args.premise_column = 'premise'
    modelA_args.question_column = 'question'
    modelA_args.passage_column = 'question'
    modelA_args.answer_column = 'answer'
    modelA_args.pre_seq_len = 128
    modelA_args.save_step = 1500
    modelA_args.ptuning_checkpoint = 'your ptuning_checkpoint path'
    modelA_args.freeze_head = True
    modelA_args.val_max_target_length = 1000
    modelA_args.max_predict_samples = 1000 

    if not modelA_args.do_train:
        if modelA_args.ptuning_checkpoint == None:
            print('please check the ptuning_checkpoint! it should\'t be None when you are not in training mode!')

    set_seed(modelA_args.seed)

    # modelA config and initial
    configA = AutoConfig.from_pretrained('./client', trust_remote_code=True)
    configA.pre_seq_len = modelA_args.pre_seq_len
    configA.prefix_projection = False

    # model = AutoModel.from_pretrained('./fed-glm-module/client/', config=config,  trust_remote_code=True)
    modelA = ChatGLMForConditionalGenerationClientSide(config=configA).cuda()
    if modelA_args.do_train:
        param = modelA.state_dict()['transformer.prefix_encoder.embedding.weight']

        state_dictA = torch.load('./client/client_model_partA_param.bin')

        state_dictA['transformer.prefix_encoder.embedding.weight'] = param  

        modelA.load_state_dict(state_dictA)

    else:
        param = modelA.state_dict()['transformer.prefix_encoder.embedding.weight']
        state_dictA = torch.load('./client/client_model_partA_param.bin')
        state_dictA['transformer.prefix_encoder.embedding.weight'] = param
        modelA.load_state_dict(state_dictA)
        ptuning_state_dict = torch.load(os.path.join(modelA_args.ptuning_checkpoint, "splitfed-ptuning.bin"))
        new_prefix_state_dict = {"embedding.weight": ptuning_state_dict} #torch.Size([128, 229376])
        modelA.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)


    # modelC config and initialize
    configC = AutoConfig.from_pretrained('./client_part3/', trust_remote_code=True)
    configC.pre_seq_len = modelA_args.pre_seq_len 
    configC.prefix_projection = False
    modelC = ChatGLMForConditionalGenerationClientSideC(config=configC).cuda()
    state_dictC = torch.load('./client_part3/client_model_partC_param.bin')
    modelC.load_state_dict(state_dictC)
    
    # Quantization
    if modelA_args.quantization_bit is not None:
        print(f"Quantized to {modelA_args.quantization_bit} bit")
        modelA = modelA.quantize(modelA_args.quantization_bit)
        modelC = modelC.quantize(modelA_args.quantization_bit)
    if modelA_args.pre_seq_len is not None:
        # P-tuning v2
        modelA = modelA.half().cuda()
        modelA.transformer.prefix_encoder.float()
        modelC = modelC.half().cuda()


    models = [modelA, modelC]
    tokenizer = AutoTokenizer.from_pretrained('chatglm-6b', trust_remote_code=True)
    optimA = AdamW(modelA.parameters(), lr=modelA_args.lr, betas=modelA_args.betas,eps=modelA_args.eps, weight_decay=modelA_args.weight_decay)
    optimC = AdamW(modelC.parameters(), lr=modelA_args.lr, betas=modelA_args.betas,eps=modelA_args.eps, weight_decay=modelA_args.weight_decay)
    scheduleA = LinearLR(optimA, start_factor=1.0, end_factor=0.0, total_iters=modelA_args.max_step)
    scheduleC = LinearLR(optimC, start_factor=1.0, end_factor=0.0, total_iters=modelA_args.max_step)
    schedule = [scheduleA,scheduleC]

    # get datalist
    datalist = get_data_list(modelA_args, tokenizer)

    # get optimizers
    optimizer = [optimA, optimC]

        
    # Start Flower client
    fl_client = Fed_Client(datalist, optimizer, schedule, models, tokenizer, modelA_args).to_client()
    fl.client.start_client(server_address="10.143.12.73:8080", client=fl_client)

# python fed-glm-module/flclient.py
if __name__ == "__main__":
    main()
