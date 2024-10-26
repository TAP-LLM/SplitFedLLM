#region
import json
from multiprocessing import connection
import sys 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.optimization import AdamW
from torch.optim.lr_scheduler import LinearLR
import gc
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('./client')
sys.path.append('./client_part3')
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from client.modeling_llama_A import LlamaForCausalLM
from client_part3.modeling_llama_C import LlamaForCausalLMC
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

# from client.client_model import ModelforTest, ChatGLMForConditionalGenerationClirntSide

'''=====================================================设置日志====================================================='''
log_path = "/home/zhangzishuai/SplitFederated-LLaMA/log/FedLlama/train_client_2.log"
logging.basicConfig(filename=log_path,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                    level=logging.INFO,
                    filemode='w')

# create a logger object
logger = logging.getLogger(__name__)
# from torch.utils.data import DataLoader
#endregion

#region
# @dataclass
# class ModelArguments:
#     """
#     Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
#     """

#     modelA_name_or_path: Optional[str] = field(
#         default=None,
#         metadata={
#             "help": (
#                 "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
#             )
#         },
#     )
#     modelC_name_or_path: Optional[str] = field(
#         default=None,
#         metadata={
#             "help": (
#                 "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
#             )
#         },
#     )
#     model_type: Optional[str] = field(
#         default=None,
#         metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
#     )
#     config_overrides: Optional[str] = field(
#         default=None,
#         metadata={
#             "help": (
#                 "Override some existing default config settings when a model is trained from scratch. Example: "
#                 "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
#             )
#         },
#     )
#     modelA_config_name: Optional[str] = field(
#         default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
#     )
#     modelC_config_name: Optional[str] = field(
#         default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
#     )
#     tokenizer_name: Optional[str] = field(
#         default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
#     )
#     cache_dir: Optional[str] = field(
#         default=None,
#         metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
#     )
#     lora_r: Optional[int] = field(default=16)
#     lora_alpha: Optional[int] = field(default=32)
#     target_modules: Optional[str] = field(
#         default='q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj',
#         metadata={
#             "help": "List of module names or regex expression of the module names to replace with Lora."
#             "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
#         },
#     )
#     use_fast_tokenizer: bool = field(
#         default=True,
#         metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
#     )
#     load_in_bits: Optional[int] = field(default=8)
#     model_revision: str = field(
#         default="main",
#         metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
#     )
#     use_auth_token: bool = field(
#         default=False,
#         metadata={
#             "help": (
#                 "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
#                 "with private models)."
#             )
#         },
#     )
    
#     torch_dtype: Optional[str] = field(
#         default=None,
#         metadata={
#             "help": (
#                 "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
#                 "dtype will be automatically derived from the model's weights."
#             ),
#             "choices": ["auto", "bfloat16", "float16", "float32"],
#         },
#     )
#     lora_model_path: Optional[str] = field(
#         default=None,
#         metadata={
#             "help": (
#                 "The pretrained model checkpoint for lora weights."
#             ),
#         },
#     ) 

#     def __post_init__(self):
#         if self.config_overrides is not None and (self.modelA_config_name is not None or self.model_name_or_path is not None):
#             raise ValueError(
#                 "--config_overrides can't be used in combination with --modelA_config_name or --model_name_or_path"
#             )
#         if self.config_overrides is not None and (self.modelC_config_name is not None or self.model_name_or_path is not None):
#             raise ValueError(
#                 "--config_overrides can't be used in combination with --modelC_config_name or --model_name_or_path"
#             )
#         if type(self.target_modules)==str:
#             self.target_modules = self.target_modules.split(',')


# @dataclass
# class DataTrainingArguments:
#     """
#     Arguments pertaining to what data we are going to input our model for training and eval.
#     """
#     train_on_inputs: bool = field(
#         default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
#     )
#     dataset_name: Optional[str] = field(
#         default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
#     )
#     dataset_config_name: Optional[str] = field(
#         default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
#     )
#     train_files: Optional[List[str]]  = field(default=None, metadata={"help": "The input training data file (a text file)."})
#     validation_files: Optional[List[str]]  = field(
#         default=None,
#         metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
#     )
    # max_train_samples: Optional[int] = field(
    #     default=None,
    #     metadata={
    #         "help": (
    #             "For debugging purposes or quicker training, truncate the number of training examples to this "
    #             "value if set."
    #         )
    #     },
    # )
    # max_eval_samples: Optional[int] = field(
    #     default=None,
    #     metadata={
    #         "help": (
    #             "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
    #             "value if set."
    #         )
    #     },
    # )
    # streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    # block_size: Optional[int] = field(
    #     default=None,
    #     metadata={
    #         "help": (
    #             "Optional input sequence length after tokenization. "
    #             "The training dataset will be truncated in block of this size for training. "
    #             "Default to the model max input length for single sentence inputs (take into account special tokens)."
    #         )
    #     },
    # )
    # overwrite_cache: bool = field(
    #     default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    # )
    # validation_split_percentage: Optional[int] = field(
    #     default=5,
    #     metadata={
    #         "help": "The percentage of the train set used as validation set in case there's no validation split"
    #     },
    # )
    # preprocessing_num_workers: Optional[int] = field(
    #     default=None,
    #     metadata={"help": "The number of processes to use for the preprocessing."},
    # )
    # keep_linebreaks: bool = field(
    #     default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    # )

    # def __post_init__(self):
    #     if self.streaming:
    #         require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

    #     if self.dataset_name is None and self.train_files is None and self.validation_files is None:
    #         raise ValueError("Need either a dataset name or a training/validation file.")
    #     else:
    #         if self.train_files is not None:
    #             extension = self.train_files[0].split(".")[-1]
    #             assert extension in ["csv", "jsonl", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
    #         if self.validation_files is not None:
    #             extension = self.validation_files[0].split(".")[-1]
    #             assert extension in ["csv", "jsonl",  "json","txt"], "`validation_file` should be a csv, a json or a txt file."
#endregion          

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

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
    parser.add_argument("--modelA_name_or_path", type=str, default="/home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module/client",
                        help="The model checkpoint for weights initialization. Don't set if you want to train a model from scratch.")
    parser.add_argument("--modelC_name_or_path", type=str, default="/home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module/client_part3",
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
    parser.add_argument("--target_modules", type=str, default='q_proj,v_proj,k_proj,o_proj',
                        help="List of module names or regex expression of the module names to replace with Lora. For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'")
    parser.add_argument("--use_fast_tokenizer", default=False,
                        help="Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.")
    parser.add_argument("--load_in_bits", type=int, default=8)
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
    # parser.add_argument("--quantization_bit", type=int, help="quantization bit",  default=4)
    # parser.add_argument("--pre_seq_len", type=int, help="length of p-tuning v2 prefix sequence ", default=5)     
    # parser.add_argument("--freeze_head",type=bool, help='Whether to freeze the lm_head layer.', default=False)

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
    parser.add_argument("--output_dir", type=str, help = 'output folder path', default="/home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module/output/Friends/ReadCompre/version_1")
    parser.add_argument("--save_step", type=int, help = 'step to save the prefix encoder', default=20)
    parser.add_argument("--overwrite_output_dir", type=bool, help = 'whether to overwrite the output folder', default=False)
    parser.add_argument("--block_size", type=int, help = 'I dont know yet', default=None)

    '''=====================================================datasets args====================================================='''  
    parser.add_argument("--ptuning_checkpoint", type=str, help="ptuning_checkpoint", default=None)
    parser.add_argument("--data_fold", type=str, default=None)
    parser.add_argument("--max_source_length", type=int, default = 640)
    parser.add_argument("--max_target_length", type=int, default = 48)
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
    parser.add_argument("--max_step", type=int, help='number of max training steps, should be same with serve side!', default=10)
    parser.add_argument("--dataloader_num_workers", type=int, default=6)
    parser.add_argument("--dataloader_pin_memory", type=bool, default=True)
    parser.add_argument("--dataloader_drop_last", type=bool, default=True)    
    # parser.add_argument("--train_file", type=str,  default='reading-comprehension-trn.json')
    # parser.add_argument("--validation_file", type=str,  default='reading-comprehension-dev.json') 
    # parser.add_argument("--test_file", type=str,  default='reading-comprehension-tst.json')
    parser.add_argument("--train_file", type=str,  default='reading-comprehension-trn-part2.json')
    parser.add_argument("--validation_file", type=str,  default=None) 
    parser.add_argument("--test_file", type=str,  default=None)


    '''=====================================================Fl Arguments====================================================='''  

    parser.add_argument("--prompt", type=str, default=None)
    #parser.add_argument("--raw_prompt", type=str, help='Prompt received from bash script.', default=None)  
    parser.add_argument("--pipe", type=str, default=None)    
    parser.add_argument("--max_predict_samples", type=Optional[int], default=None)
    parser.add_argument("--val_max_target_length", type=Optional[int], default=None)
    parser.add_argument("--do_train", action="store_true", help='Whether to run training.', default=False)
    parser.add_argument("--do_eval", action="store_true", help='Whether to run eval on the dev set.', default=False)
    parser.add_argument("--do_predict", action="store_true", help='Whether to run predictions on the test set.', default=False)
    parser.add_argument("--do_inference", action="store_true", help='Whether to inference using prompt from users.', default=False)
    parser.add_argument("--resume_from_checkpoint", type=bool, help='Whether to resume from checkpoint.', default=False)
    parser.add_argument("--dry", type=bool, default=False, required=False, help="Do a dry-run to check the client")
    parser.add_argument("--client_id", type=int, default=1, choices=range(0, 10), required=False, help="Specifies the artificial data partition of CIFAR10 to be used. \
        Picks partition 0 by default")
    parser.add_argument(  "--toy", action="store_true", help="Set to true to quicky run the client using only 10 datasamples. \
        Useful for testing purposes. Default: False")
    parser.add_argument("--use_cuda", type=bool, default=True, required=False, help="Set to true to use GPU. Default: False")
    parser.add_argument("--model", type=str, default="efficientnet", choices=["efficientnet", "alexnet"],
        help="Use either Efficientnet or Alexnet models. If you want to achieve differential privacy, please use the Alexnet model")
    parser.add_argument("--per_client_steps", type=int, help="the number of the traing steps of every client", default=100)

    args = parser.parse_args()

    return args

class Fed_Client(NumPyClient):
    def __init__(self, datalist, optimizer, schedule, models, tokenizer, FL_args, child_conn) -> None:
        super(Fed_Client, self).__init__()
        self.child_conn = child_conn
        self.datalist = datalist # list化的dataloader,因为这里不是主循环入口
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
        self.optim = optimizer
        self.schedule = schedule 
        self.label = None
        self.all_label=[]
        self.f1 = None
        self.FL_args = FL_args
        self.train_metricx = {'step':[], 'loss':[]}
        self.eval_output = {'pred_id':[]}
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.id = torch.tensor(self.FL_args.client_id).to(self.device)
        self.data = None
        self.total_pred=[]
        if self.FL_args.do_predict and not self.FL_args.do_inference:
            self.score_list=[]
        if self.FL_args.do_train:
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
            if self.FL_args.do_predict:
                feature, num_sample, metrics = self.fit_partC(parameters, config, self.child_conn)

            else:
                feature, num_sample, metrics = self.fit_partC(parameters, config)

        elif config['type']==3:
            # print('backward client part A')
            feature, num_sample, metrics = self.backward_modelA(parameters, config)

        return feature, num_sample, metrics
    
    def fit_partA(self, parameters: NDArrays=None, current_step=0, need_test_data=0):
        # (data, label) =self.datalist[current_step]

        if self.FL_args.do_train:
            current_epoch = current_step // self.len_dataset
            current_batch =current_step % self.len_dataset
            data = self.datalist[current_epoch][current_batch]['input_ids'].to(self.device)
            attn_mask = self.datalist[current_epoch][current_batch]['attention_mask'].to(self.device)
            # data.type:torch.Size([1, 640])
            label = self.datalist[current_epoch][current_batch]['labels'].to(self.device)
            #label.type: torch.Size([1, 168]) tensor([[ -100,  -100,     1,   518, 25580, 29962, 16492,
        elif self.FL_args.do_eval:
            # 验证集有多少个，current_step就应该有多少个
            if need_test_data ==1:
                data = self.datalist[current_step]['input_ids'].to(self.device)
                self.data = data
            else:
                data = self.data.to(self.device)
            label = self.datalist[current_step]['labels'].to(self.device)
        else:
            if need_test_data ==1:  # 根据query前向传播
                if not self.FL_args.do_inference: #测试集
                    data = self.datalist[current_step]['input_ids'].to(self.device)
                    label = self.datalist[current_step]['labels'].to(self.device)  # 测试集有标签
                # 6/26新增 data形状  torch.Size([1, 128])   
                else: # 用户query
                    data = self.datalist.to(self.device)
                    label = None # 用户推理没有标签
                self.data = data
                
            else: # 拼接query前向传播
                data = self.data.to(self.device)
                if not self.FL_args.do_inference:  # 推理的时候没有label
                    label = self.datalist[current_step]['labels'].to(self.device)
                else:
                    label = None
                #logger.info(f"label:{label}")

        # forward pass
        #start_1 = time.time() # 
        if self.FL_args.do_train:
            f1 = self.model[0](input_ids=data, attention_mask=attn_mask)  # data (batchsize,seq_len)
        else:
            f1 = self.model[0](data)
        #end_1 = time.time() #
        #cost_1 = end_1 - start_1 # 
        #logger.info(f"前向传播modelA_cost:{cost_1}") #
        #start_2 = time.time() # 
        hidden_A = f1.last_hidden_state.clone().detach() #.requires_grad_(True)
        # hidden_A.shape torch.Size([1, 352, 4096]) (batch_size, ,embedding_size)
        att_mask = f1.attention_mask.clone().detach()
        # att_mask.shape torch.Size([1, 1, 4096, 4096])
        p_ids = f1.position_ids.clone().detach()
        # p_ids.shape 

        # pastkey_value(tuple(tensor{[2, 5, 1, 32, 128]})) to tensor
        # param = torch.cat(p_list).permute([2, 1, 0, 3, 4]).reshape(1,128,2*28*4096)
        # param = param.squeeze(0)
        # key_cache: [tensor([[[[-0.2559, ...ackward0>)]
        # seen_tokens: 168
        # value_cache: [tensor([[[[ 6.3417e-...ackward0>)]
        #以下是GLM的 
        pkv = []
        past_key= f1.past_key_values.key_cache[0]
        past_value = f1.past_key_values.value_cache[0]
        pkv.append(past_key.clone().detach()) # torch.Size([1, 32, 168, 128])
        pkv.append(past_value.clone().detach()) # torch.Size([1, 32, 168, 128])
        pkv = torch.cat(pkv) # torch.Size([2, 32, 168, 128])  
        seen_tokens = torch.tensor(f1.past_key_values.seen_tokens).to(self.device)

        self.label = label
        self.f1 = f1
        
        od = collections.OrderedDict([('hidden_stateA', hidden_A),
                                      ('attention_mask', att_mask),
                                      ('position_ids', p_ids),
                                      ('past_key_values', pkv),
                                      ('seen_tokens',seen_tokens)])
        feature = [val.cpu().numpy() for _, val in od.items()]
        #end_2 = time.time() #
        #cost_2 = end_2 - start_2 # 
        #logger.info(f"打包modelA传输的数据_cost:{cost_2},此时时间为:{time.time()}") #

        return feature # 数据不包含梯度！

    def fit_partC(self, parameters: NDArrays=None, config: Dict[str, Scalar]=None , child_conn=None):       
        # ff2 = self.set_parameters(ff2) # recieve feture of model2 from server
        # print('there')

        # past_key_values = torch.from_numpy(parameters[3]).to(self.device) #  torch.Size([2, 5, 1, 32, 128])
        # # print('past_key_values size in fit C:',past_key_values.size())
        # past_key_values = past_key_values.clone().detach().requires_grad_(True)
        # pkv2 = ([past_key_values])
        '''=====================================================计算loss并更新模型C====================================================='''
        #start_3 = time.time() # 
        pkv=torch.from_numpy(parameters[3]).to(self.device) #  layers in one tensor
        pkv_tensors = torch.split(pkv, 1, dim=0)
        past_key = pkv_tensors[0:31]
        past_value = pkv_tensors[31:62]
        # past_key, past_value = torch.split(pkv, 31, dim=0)
        # past_key = [torch.split(past_key, 1, dim=0)]
        # past_value = [torch.split(past_value, 1, dim=0)]
        pkv2= DynamicCache.from_legacy_cache(None)
        for past_key in past_key:
            pkv2.key_cache.append(past_key.requires_grad_(True))
        for past_value in past_value:
            pkv2.value_cache.append(past_value.requires_grad_(True))
        pkv2.seen_tokens = torch.from_numpy(parameters[4])  
        hidden = torch.from_numpy(parameters[0]).to(self.device).requires_grad_(True)
        att_mask = torch.from_numpy(parameters[1]).to(self.device)
        p_ids = torch.from_numpy(parameters[2]).to(self.device)
        #end_3 = time.time() # 
        #cost_3 = end_3 - start_3 # 
        #logger.info(f"modelC解析接收到的数据准备前向传播_cost:{cost_3}") #
        #start_4 = time.time() # 
        final_output = self.model[1](position_ids = p_ids,
                            attention_mask = att_mask,
                            hidden_states = hidden,
                            past_key_values = pkv2, 
                            labels = self.label)
        #end_4 = time.time() # 
        #cost_4 = end_4 - start_4 # 
        #logger.info(f"modelC前向传播_cost:{cost_4}") # 
        # final_output.logits.size:torch.Size([1, 128, 130528])
        
        metrics = {}
        gradient = []
        labels = []
        #start_5 = time.time() # 
        if self.FL_args.do_train:
            # log loss
            loss = final_output[0]
            # print(loss.item())
            log(INFO, 'step %s lr %s  loss %s:', config["current_step"], self.optim[0].param_groups[0]['lr'], loss.item())

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
            #pkv_grad_last = pkv2[0].grad
            # od = collections.OrderedDict([('hiddden_gradient', hidden_gradient),
            #                           ('pkv_grad_last', pkv_grad_last),
            #                           ('client_id', self.id)])
            od = collections.OrderedDict([('hiddden_gradient', hidden_gradient),
                            ('client_id', self.id)])
            gradient = [val.cpu().numpy() for _, val in od.items()]

        
            metrics['loss'] = loss.item()

            self.train_metricx['step'].append(config['current_step'])
            self.train_metricx['loss'].append(loss.item())

        # 6/26新增
        elif self.FL_args.do_predict:
            metrics['pred_finished'] = 0
            #log(INFO, 'DO_PREDICT')
            if len(self.total_pred)==0:
                print('DO_PREDICT')

            # 6/26新增
            # final_output: tuple(loss, logits)
            # final_output[1]是一个元组，长度为1，final_output[1][0]是一个元组，长度为2，每一个都是一个tensor, 形状都是torch.Size([256, 1, 32, 128])
            #print(f"final_output.logits:{final_output.logits}")
            # tensor(,batch_size,, vocab_size)
            logits=final_output.logits[:,-1,:] #(B，vocab size)
            probs = F.softmax(logits, dim=-1)
            # 此时有多种生成方法，这里使用采样生成方法
            # print("self.data:",self.data)
            pred_ids = self.sample_text(probs)
            self.total_pred.append(int(pred_ids[0][0]))
            # 解码预测结果
            tokenizer = AutoTokenizer.from_pretrained("/home/zhangzishuai/SplitFederated-LLaMA/Models/Llama2-7B-chat-service", trust_remote_code=True)
            response = tokenizer.decode(int(pred_ids[0][0]))

            logger.info(f"这是第{len(self.total_pred)}个pred_ids:{pred_ids},解码结果:{response}")

            #发送给网页
            if self.child_conn is not None:
                self.child_conn.send(response)            

            # 找到padding的第一个位置，把data的第一个padding符号替换掉
            self.data = self.concatenate_output(self.data, pred_ids)

            self.FL_args.max_output_length=128
            #end_5 = time.time() # 
            #cost_5 = end_5 - start_5 # 
            #logger.info(f"modelC得到pred_decode_cost:{cost_5}") # 
            # 如果输出结束符号，则告诉服务端停止
            if pred_ids[0][-1] == 2 or len(self.total_pred) == self.FL_args.max_output_length:
                #结束符2
                #self.FL_args.pred_finished = True
                #使用decoder输出所有的预测结果
                tokenizer = AutoTokenizer.from_pretrained("/home/zhangzishuai/SplitFederated-LLaMA/Models/Llama2-7B-chat-service", trust_remote_code=True)
                response = tokenizer.decode(self.total_pred, skip_special_tokens=True)
                logger.info(f"response:{response}")
                print(f"response:{response}")
                if not self.FL_args.do_inference:
                    score = compute_metrics(self.total_pred, self.label, self.tokenizer, self.FL_args)
                    logger.info(f"当前回答的score:{score}")
                    self.score_list.append(score)
                    correct_predictions = sum(self.score_list)
                    accuracy = correct_predictions / len(self.score_list)
                    logger.info(f"已回答{len(self.score_list)}个问题,当前回答的score_list:{self.score_list},准确率为:{accuracy:.2f}")
                if child_conn is not None:
                    child_conn.close()
                self.total_pred=[]
                metrics['pred_finished'] = 1



        else:
            log(INFO, 'finish step %s', config["current_step"])
            probs = F.softmax(final_output[1], dim=-1)
            pred_ids = torch.argmax(probs, dim=-1)

            self.eval_output['pred_id'].append(pred_ids.cpu())
            self.all_label.append(self.label)
            if config['current_step'] == self.FL_args.max_step-1 :
                # compute metrix
                pred_ids = torch.cat(self.eval_output['pred_id']).cpu().tolist()
                labels_all = torch.cat(self.all_label).cpu().tolist()

                score = compute_metrics(pred_ids, labels_all, self.tokenizer, self.FL_args)
                print(score)
                print('Now saving evalute metrix!')
                print('Evaluation finish!')


        return gradient, 0, metrics  # GetParametersRes

    #采样生成代码
    def sample_text(self, probs, temperature=0.95, top_p=0.7):
        # 应用temperature
        probs = probs.pow(1 / temperature)
        # if ( probs  != 0).any():
        #     print("inputs_embeds 不全是0")
        # else:
        #     print("inputs_embeds 全是0")

        # 按概率排序
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        
        # 计算累积概率
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 移除低于阈值的token
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0  # 保留最高概率的token
        
        # 屏蔽掉被移除的indices
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        probs[indices_to_remove] = 0
    
        # 重新归一化概率分布
        probs = probs / probs.sum(dim=-1, keepdim=True)

        # 采样
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token

    def concatenate_output(self, input_tensor, output_tensor, pad_token=3):
        # 获取输入的形状
        batch_size, seq_len = input_tensor.size()

        # 去掉padding部分
        non_padding_input = input_tensor[input_tensor != pad_token].view(batch_size, -1)

        # 拼接输出
        new_input = torch.cat((non_padding_input, output_tensor), dim=1)

        # 重新填充到原始长度
        pad_length = seq_len - new_input.size(1)  #128
        #pad_length = 2*seq_len - new_input.size(1) #256：问题和回答总token不超过256
        padded_input = torch.nn.functional.pad(new_input, (pad_length, 0), value=pad_token)
        
        return padded_input

    def backward_modelA(self, parameters: NDArrays=None, config: Dict[str, Scalar]=None):
        # ff1_grad = self.set_parameters(args.ff1_grad) # recieve gradient of ff1 from server

        hidden_gradient = torch.from_numpy(parameters[0]).to(self.device)
        # print(hidden_gradient)

        # pkv_gradient = torch.from_numpy(parameters[1]).to(self.device) # tensor([batch_size,pre_seq_len, 2*27*4096])

        # reshape pkv_gradient
        # pkv_gradient = pkv_gradient.unsqueeze(0).expand(self.FL_args.batch_size, -1, -1) # batch size==1

        # pkv_gradient = pkv_gradient.view(self.FL_args.batch_size,self.FL_args.pre_seq_len, 2*27,32,128).permute([2, 1, 0, 3, 4]).split(2)
        # newpkv_gradient = [torch.stack([v[0],v[1]]) for v in pkv_gradient]

        # for i in range(len(newpkv_gradient)):


        # print(type(lastpkv_gradient))
        self.optim[0].zero_grad()
        # self.f1.last_hidden_state.backward(hidden_gradient, retain_graph=True)

        # for i in range(27):
        # # print(i)
        # # print(f1.past_key_values[i].size())
        # # print(grad[i].size())
        #     self.f1.past_key_values[i].backward(newpkv_gradient[i], retain_graph=True) 

        self.f1.last_hidden_state.backward(hidden_gradient)
        
        # 梯度剪裁
        nn.utils.clip_grad_norm_(self.model[0].parameters(), self.FL_args.max_grad_norm)
        self.optim[0].step()
        self.schedule[0].step()

        if (config['current_step']+1) % self.FL_args.save_step == 0:
            # self.save_model(config['current_step'], self.optim, self.schedule)
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


def old_compute_metrics(preds, labels, tokenizer, args):
        # preds, labels = eval_preds  # batched token list 

        if isinstance(preds, tuple): 
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            logger.info(f"labels:{labels}")
            logger.info(f"labels.shape:{labels.shape}")
            
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

         # FOR BOOLQ!!!
        # for i in range(len(decoded_labels)):
        #     if decoded_preds[i]=='':
        #         decoded_preds[i]='Yes' if decoded_labels[i]=='No' else 'No'

        print('predictions:', preds)
        print('decode_predictions:', decoded_preds)
        print('labels:', labels)
        print('decode_labels:', decoded_labels)
        
        # auc = 0
        # for i in range(len(decoded_labels)):
        #     if decoded_labels[i] == decoded_preds[i]:
        #         auc + 1
        
        # acc = auc/len(decoded_labels)

        # preds_1 = []
        # labels_1 = []
        # # CB
        # for item in decoded_preds:
        #     if item=='Yes':
        #         preds_1.append(0)
        #     elif item=='No':
        #         preds_1.append(1)
        #     elif item=='Maybe':
        #         preds_1.append(2)

        # # for item in decoded_labels:
        # #     labels_1.append(1 if item=='Yes' else 0)
        # for item in decoded_labels:
        #     if item=='Yes':
        #         labels_1.append(0)
        #     elif item=='No':
        #         labels_1.append(1)
        #     elif item=='Maybe':
        #         labels_1.append(2)

        # pred_1 = np.array(preds_1)
        # label_1 = np.array(labels_1)

        # print(pred_1)
        # print(label_1)

        acc = accuracy_score(label_1,pred_1)
        cm = confusion_matrix(decoded_labels, decoded_preds, labels=['Yes','No','Maybe']) # CB
        f1score = f1_score(label_1, pred_1, average='macro')

        # print('metrics:')
        # print('Accuracy:', acc)
        # print('Confusion_matrix:', cm)
        # print('F1-Score',f1score)

        score_dict = {
            # "rouge-1": [],
            # "rouge-2": [],
            # "rouge-l": [],
            # "bleu-4": [] ,
            "Accuracy": [],
            "Confusion_matrix":[],
            "F1-Score":[] 
        }
        # for pred, label in zip(decoded_preds, decoded_labels):
        #     # ...input
        #     hypothesis = list(jieba.cut(pred))
        #     reference = list(jieba.cut(label))
        #     rouge = Rouge()
        #     scores = rouge.get_scores(' '.join(hypothesis) , ' '.join(reference))
        #     result = scores[0]
            
        #     for k, v in result.items():
        #         score_dict[k].append(round(v["f"] * 100, 4))
        #     bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
        #     score_dict["bleu-4"].append(round(bleu_score * 100, 4))
        #     # score_dict["Accuracy"].append(round(acc * 100, 4))

        # for k, v in score_dict.items():
        #     score_dict[k] = float(np.mean(v))

        score_dict["Accuracy"].append(acc)
        score_dict["Confusion_matrix"].append(cm.tolist)
        score_dict["F1-Score"]= round(f1score* 100, 4)
                
        return score_dict

def compute_metrics(preds, labels, tokenizer, args):
        # preds, labels = eval_preds  # 仅一个回答的 token list 

        decoded_preds = tokenizer.decode(preds, skip_special_tokens=True)
        if args.ignore_pad_token_for_loss:
            # where(a,b,c)  若a为真，则输出该位置上的原始值，如果条件为假，则输出第二个参数中的值c
            labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)
            #logger.info(f"labels:{labels}")
        decoded_labels = tokenizer.decode(labels.squeeze(), skip_special_tokens=True)

        #logger.info('predictions:', preds)
        logger.info(f'decode_predictions:{decoded_preds}')
        #logger.info('labels:', labels)
        logger.info(f'decode_labels:{decoded_labels}')
        
        if decoded_preds == decoded_labels:
            logger.info(f"回答正确!decode_labels=decode_predictions={decoded_labels}")
            score = 1
        else:
            logger.info(f"回答错误!decode_labels={decoded_labels} ;  decode_predictions={decoded_preds}")
            score = 0
        


        # pred_1 = np.array(preds_1)
        # label_1 = np.array(labels_1)

        # print(pred_1)
        # print(label_1)

        # acc = accuracy_score(label_1,pred_1)
        # cm = confusion_matrix(decoded_labels, decoded_preds, labels=['Yes','No','Maybe']) # CB
        # f1score = f1_score(label_1, pred_1, average='macro')

        # print('metrics:')
        # print('Accuracy:', acc)
        # print('Confusion_matrix:', cm)
        # print('F1-Score',f1score)

        # score_dict = {
        #     # "rouge-1": [],
        #     # "rouge-2": [],
        #     # "rouge-l": [],
        #     # "bleu-4": [] ,
        #     "Accuracy": [],
        #     "Confusion_matrix":[],
        #     "F1-Score":[] 
        # }
        # for pred, label in zip(decoded_preds, decoded_labels):
        #     # ...input
        #     hypothesis = list(jieba.cut(pred))
        #     reference = list(jieba.cut(label))
        #     rouge = Rouge()
        #     scores = rouge.get_scores(' '.join(hypothesis) , ' '.join(reference))
        #     result = scores[0]
            
        #     for k, v in result.items():
        #         score_dict[k].append(round(v["f"] * 100, 4))
        #     bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
        #     score_dict["bleu-4"].append(round(bleu_score * 100, 4))
        #     # score_dict["Accuracy"].append(round(acc * 100, 4))

        # for k, v in score_dict.items():
        #     score_dict[k] = float(np.mean(v))

        # score_dict["Accuracy"].append(acc)
        # score_dict["Confusion_matrix"].append(cm.tolist)
        # score_dict["F1-Score"]= round(f1score* 100, 4)
                
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

    # device
    device = torch.device("cuda:1" if torch.cuda.is_available() and FL_args.use_cuda else "cpu")

    torch.manual_seed(42)
    FL_args.lr = 1e-5
    FL_args.data_fold = '/home/zhangzishuai/SplitFederated-LLaMA/Dataset/Friends/ReadComprehension'
    #FL_args.max_step = 15000
    # # For huatuo
    # FL_args.premise_column = 'premise'
    # FL_args.question_column = 'question'
    # FL_args.passage_column = 'question'
    # FL_args.answer_column = 'answer'
    # For Friends
    FL_args.question_column = 'query'
    FL_args.passage_column = 'utterances'
    FL_args.answer_column = 'answer'
    #FL_args.save_step = 20
    # FL_args.freeze_head = True
    FL_args.val_max_target_length = 1000
    FL_args.max_predict_samples = 1000 # 样本最大数
    FL_args.prompt = prompt
    #FL_args.lora_modelA_path = '/home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module/output/Friends/ReadCompre/model-A/checkpoint-1000'
    #FL_args.lora_modelC_path = '/home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module/output/Friends/ReadCompre/model-C/checkpoint-1000'
    '''=====================================================配置模型参数====================================================='''
    start_set_model=time.time()
    if FL_args.do_train:
        # Detecting last checkpoint.
        # last_checkpoint = None
        # if os.path.isdir(FL_args.output_dir) and FL_args.do_train and not FL_args.overwrite_output_dir:
        #     last_checkpoint = get_last_checkpoint(FL_args.output_dir)
        #     if last_checkpoint is None and len(os.listdir(FL_args.output_dir)) > 0:
        #         raise ValueError(
        #             f"Output directory ({FL_args.output_dir}) already exists and is not empty. "
        #             "Use --overwrite_output_dir to overcome."
        #         )
        #     elif last_checkpoint is not None and FL_args.resume_from_checkpoint:
        #         logger.info(
        #             f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
        #             "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        #         )

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
            if FL_args.config_overrides is not None:
                logger.info(f"Overriding config: {FL_args.config_overrides}")
                config.update_from_string(FL_args.config_overrides)
                logger.info(f"New config: {config}")
            

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
        tokenizer.pad_token = tokenizer.unk_token

        logger.info(f"FL_args.target_modules:{FL_args.target_modules}")
        client_layer_id = [0]
        t_m_c = []
        for index in FL_args.target_modules:
            logger.info(f"index:{index}")
            for i in client_layer_id:
                t_m_c.append(str(f'model.layers.{i}.self_attn' + '.' + index))
        #logger.info(f"t_m_c:{t_m_c}")
        FL_args.target_modules = t_m_c
        logger.info(f"FL_args.target_modules:{FL_args.target_modules}")

        lora_configA = LoraConfig(
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
        logger.info(f"lora_config:{lora_configA}")
        #region 修改loraC
        # FL_args.target_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj']
        # client_layer_id = [31]
        # t_m_c = []
        # for index in FL_args.target_modules:
        #     for i in client_layer_id:
        #         t_m_c.append(str(f'model.layers.{i}.self_attn' + '.' + index))
        # FL_args.target_modules = t_m_c
        # lora_configC = LoraConfig(
        #     r=FL_args.lora_r,
        #     lora_alpha=FL_args.lora_alpha,
        #     # target_modules=["query_key_value"],
        #     # target_modules =  ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        #     target_modules =  FL_args.target_modules,
        #     fan_in_fan_out = False,
        #     lora_dropout=0.05,
        #     inference_mode=False,
        #     bias="none",
        #     task_type="CAUSAL_LM",
        # )
        #endregion
        lora_configC = lora_configA
        logger.info(f"lora_config:{lora_configC}")
        
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16
        # )

        # model_A config and initial
        if FL_args.modelA_name_or_path:

            torch_dtype = (
                FL_args.torch_dtype
                if FL_args.torch_dtype in ["auto", None]
                else getattr(torch, FL_args.torch_dtype)
            )
            logger.info(f"modelA torch_dtype:{torch_dtype}")

            model_A = LlamaForCausalLM(config=configA).to("cpu")
            clientA_state_dict = torch.load(f'{FL_args.modelA_name_or_path}/pytorch_model_A.bin', map_location='cuda:1')
            model_A.load_state_dict(clientA_state_dict)
            # for name, module in model_A.named_modules():
            #     logger.info(f"Module: {name}, Type: {type(module)}")

            # for name, param in model_A.named_parameters():
            #     logger.info(f"Parameter Name: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")   
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

            model_C = LlamaForCausalLMC(config=configC).to("cpu")
            clientC_state_dict = torch.load(f'{FL_args.modelC_name_or_path}/pytorch_model_C.bin', map_location='cuda:1')
            model_C.load_state_dict(clientC_state_dict)
            #region 将权重字典中的层名调整为适合从31开始的索引
            # adjusted_state_dict = {}
            # for name, param in clientC_state_dict.items():
            #     # 假设所有层名都是以 'model.layers.' 开头
            #     if 'model.layers.0.' in name:
            #         # 计算新的层索引
            #         new_name = name.replace('model.layers.0.', 'model.layers.31.')
            #         adjusted_state_dict[new_name] = param
            #     else:
            #         adjusted_state_dict[name] = param
            # model_C.load_state_dict(adjusted_state_dict)

            # for name, module in model_C.named_modules():
            #     logger.info(f"Module: {name}, Type: {type(module)}")

            # for name, param in model_C.named_parameters():
            #     logger.info(f"Parameter Name: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")
            #endregion   
            del clientC_state_dict
            model_C = model_C.to(device)
            
        else:
            model_C = LlamaForCausalLM.from_config(configC)
            n_params = sum({p.data_ptr(): p.numel() for p in model_C.parameters()}.values())
            logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

        models = [model_A, model_C]
        for model in models:
            if FL_args.load_in_bits==8:
                model = prepare_model_for_int8_training(model)
            elif FL_args.load_in_bits==4:
                model = prepare_model_for_kbit_training(model)

        model_A = get_peft_model(model_A, lora_configA)
        # model_A.print_trainable_parameters()
        model_C = get_peft_model(model_C, lora_configC)
        #print(model_C)
        # model_C.print_trainable_parameters()
        # optimizer初始化
        optimA = AdamW(filter(lambda p: p.requires_grad, model_A.parameters()), lr=FL_args.lr, betas=FL_args.betas, eps=FL_args.eps, weight_decay=FL_args.weight_decay)
        optimC = AdamW(filter(lambda p: p.requires_grad, model_C.parameters()), lr=FL_args.lr, betas=FL_args.betas, eps=FL_args.eps, weight_decay=FL_args.weight_decay)
        # schedule 初始化
        scheduleA = LinearLR(optimA, start_factor=1.0, end_factor=0.0, total_iters=FL_args.max_step)
        scheduleC = LinearLR(optimC, start_factor=1.0, end_factor=0.0, total_iters=FL_args.max_step)
        schedule = [scheduleA,scheduleC]
        optimizer = [optimA, optimC]

    else: # 不训练模型的时候加载lora模型
        set_seed(FL_args.seed)
        config_kwargs = {
            "cache_dir": FL_args.cache_dir,
            "revision": FL_args.model_revision,
            "use_auth_token": True if FL_args.use_auth_token else None,
        }
        if FL_args.modelA_config_name and FL_args.modelC_config_name:   # 从这里进入
            configA = AutoConfig.from_pretrained(FL_args.modelA_config_name, **config_kwargs)
            configC = AutoConfig.from_pretrained(FL_args.modelC_config_name, **config_kwargs)
        # 模型A
        if FL_args.modelA_name_or_path:
            torch_dtype = (
                FL_args.torch_dtype
                if FL_args.torch_dtype in ["auto", None]
                else getattr(torch, FL_args.torch_dtype)
            )
            model_A = LlamaForCausalLM(config=configA).to('cpu')#.to(device)
            clientA_state_dict = torch.load(f'{FL_args.modelA_name_or_path}/pytorch_model_A.bin', map_location='cuda:1')
            model_A.load_state_dict(clientA_state_dict)
            del clientA_state_dict
            model_A = model_A.to(device)

        if FL_args.modelC_name_or_path:
            torch_dtype = (
                FL_args.torch_dtype
                if FL_args.torch_dtype in ["auto", None]
                else getattr(torch, FL_args.torch_dtype)
            )
            model_C = LlamaForCausalLMC(config=configC).to('cpu')#.to(device) 
            clientC_state_dict = torch.load(f'{FL_args.modelC_name_or_path}/pytorch_model_C.bin', map_location='cuda:1')
            model_C.load_state_dict(clientC_state_dict)
            del clientC_state_dict
            model_C = model_C.to(device)
        
        if FL_args.lora_modelA_path and FL_args.lora_modelC_path:
            model_A = PeftModel.from_pretrained(model_A, FL_args.lora_modelA_path, device_map='auto',trust_remote_code=True)
            model_A = model_A.merge_and_unload()
            model_C = PeftModel.from_pretrained(model_C, FL_args.lora_modelC_path, device_map='auto',trust_remote_code=True)
            model_C = model_C.merge_and_unload()
            logger.warning(f"正加载{FL_args.lora_modelA_path}和{FL_args.lora_modelC_path}的checkpoint")
        else:
            logger.warning("没有加载任何checkpoint!")

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
        # optimizer初始化
        schedule = None
        # schedule 初始化
        optimizer = None

    models = [model_A, model_C]

    end_set_model = time.time()
    model_time_cost = end_set_model - start_set_model
    log(INFO, f"client加载模型耗时{model_time_cost:.2f} 秒。")

    # # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # # on a small vocab and want a smaller embedding size, remove this test.
    # embedding_size = model.get_input_embeddings().weight.shape[0]
    # if len(tokenizer) > embedding_size:
    #     model.resize_token_embeddings(len(tokenizer))
    
    # tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    if FL_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 2048:
            block_size = 2048
    else:
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    datalist = get_dataset(FL_args, tokenizer)
    logger.info(f"client_args:{FL_args}")
    # Start Flower client
    fl_client = Fed_Client(datalist, optimizer, schedule, models, tokenizer, FL_args, child_conn).to_client()
    #print(f"client 0的cid为:",fl_client.cid)
    fl.client.start_client(server_address="10.143.12.74:8080", client=fl_client)

# python fed-glm-module/flclient.py
if __name__ == "__main__":
    args = FLparser()

    if args.do_train:
        logger.info("正在进行客户端训练")
        assert not (args.do_eval or args.do_predict or args.do_inference), "当 do_train 为 True 时，其它标志必须为 False"
        main()
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
        assert not (args.do_train or args.do_eval), "当 do_inference 为 True 时, do_train, do_BatchParallel_train, do_eval 必须为 False"
        raw_prompt='''Instruction: Please identify who '[MASK]' represents in the input according to the dialogue.
 Dialogue: '"@ent00": "I ca n't believe what I 'm hearing here ."
"@ent01": "( sings ) I ca n't believe what I 'm hearing here ..."
"@ent02": "What ? I - I said you had a -"
"@ent01": "( sings ) What I said you had ..."
"@ent02": "( to @ent01 ) Would you stop ?"
"@ent01": "Oh , was I doing it again ?"
"@ent03": "Yes !"
"@ent02": "I said that you had a nice butt , it 's just not a great butt ."
"@ent00": "Oh , you would n't know a great butt if it came up and bit ya ."
"@ent04": "There 's an image ."
"@ent05": "( walks up with a pot of coffee ) Would anybody like more coffee ?"
"@ent06": "Did you make it , or are you just serving it ?"
"@ent05": "I 'm just serving it ."
"@ent03": "Yeah . Yeah , I 'll have a cup of coffee ."
"@ent06": "Kids , new dream ... I 'm in Las Vegas . ( @ent05 sits down to hear @ent06 's dream . )"
"@ent07": "( To @ent05 ) Ahh , miss ? More coffee ?"
"@ent05": "Ugh . ( To another @ent07 that 's leaving . ) Excuse me , could you give this to that guy over there ? ( Hands him the coffee pot . ) Go ahead . ( He does so . ) Thank you . ( To the gang . ) Sorry . Okay , Las Vegas ."
"@ent06": "Okay , so , I 'm in Las Vegas ... I 'm @ent08 -"'. 
Input:'@ent01 repeats what [MASK] and @ent00 says in song , to their annoyance .'. 
Output:'[MASK]' represents:</s>'''
        main(raw_prompt)
        




    
    # 以下是为了本地测试推理
    #prompt = '<s>Human:你好！我眼睛不舒服。\n</s><s>Assistant:'
    #Hello! Who are you? What can you do?
    #prompt = "<s>Human: Hello, I'm having discomfort in my eyes.\n</s><s>Assistant: "

#     raw_prompt='''[INST]Context: '"@ent00": "Hey , @ent01 !"
# "@ent01": "Hey @ent00 , welcome back ! How was Florida ?"
# "@ent00": "You had sex , did n't you ?"
# "@ent01": "How do you do that ?"
# "@ent00": "Oh , I hate you , I 'm pushing my @ent02 through @ent03 and you 're having sex ! So ? Who ?"
# "@ent01": "You know @ent04 ?"
# "@ent00": "@ent04 ? Oh yeah , I know @ent04 ."
# "@ent01": "You mean you know @ent04 like I know @ent04 ?"
# "@ent00": "Are you kidding ? I take credit for @ent04 . Y'know before me , there was no snap in his turtle for two years ."'. Query:'@ent01 slept with [MASK] the Wine Guy .'. You need to identify who '[MASK]' represents in the query. Answer:[/INST]'''
#     #prompt = f'<s>Human:{raw_prompt}\n</s><s>Assistant:'

#     main(raw_prompt)


    # 以下是为了网页展示推理
    # args = FLparser()
    # prompt = args.prompt  #接受命令行参数
    # pipe_fd = int(args.pipe)
    # child_conn = connection.Connection(pipe_fd)
    # main(str(prompt), child_conn)

    # child_conn.close()  # 关闭子管道
