import sys 
import os
import torch
import torch.nn as nn
from transformers.optimization import AdamW

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('chatglm-6b file path')
sys.path.append('./client')
sys.path.append('./client_part3')

from client_part3.client_model_partC import ChatGLMForConditionalGenerationClientSideC
from client.client_model_partA import ChatGLMForConditionalGenerationClientSide

import argparse
from datasets import load_dataset
from collections import OrderedDict
import collections
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar
from flwr.client.client import Client
from flwr.client import NumPyClient

# import utils
import flwr as fl
from typing import Optional

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
from fedsplit_arguments import  ModelArguments, DataTrainingArguments    


def FLparser(): # use glm arguments with argument.py
    # Parse command line argument `partition`
    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--dry",
        type=bool,
        default=False,
        required=False,
        help="Do a dry-run to check the client",
    )
    parser.add_argument(
        "--client-id",
        type=int,
        default=0,
        choices=range(0, 10),
        required=False,
        help="Specifies the artificial data partition of CIFAR10 to be used. \
        Picks partition 0 by default",
    )
    parser.add_argument(
        "--toy",
        action="store_true",
        help="Set to true to quicky run the client using only 10 datasamples. \
        Useful for testing purposes. Default: False",
    )
    parser.add_argument(
        "--use_cuda",
        type=bool,
        default=False,
        required=False,
        help="Set to true to use GPU. Default: False",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="efficientnet",
        choices=["efficientnet", "alexnet"],
        help="Use either Efficientnet or Alexnet models. \
             If you want to achieve differential privacy, please use the Alexnet model",
    )
    parser.add_argument("--save_name",type=str, default = 'pytorch_model_parabatch')
    parser.add_argument("--client_blocks", type=int, help='number of client\'s blocks', default=1)
    parser.add_argument("--per_client_steps", type=int, help="the number of the traing steps of every client", default=100)
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptuning_checkpoint", type=str, help="ptuning_checkpoint",default=None)
    # model args
    parser.add_argument("--quantization_bit", type=int, help="quantization bit",  default=4)
    parser.add_argument("--pre_seq_len", type=int, help="length of p-tuning v2 prefix sequence ", default=5)     
    parser.add_argument("--batch_size", type=int, help="traing batch size", default=1) 
    # training args
    parser.add_argument("--seed", type=int, help="random seed", default=42)
    parser.add_argument("--device",type=str, default = 'cuda')
    parser.add_argument("--max_grad_norm", type=float, help='max grad_clipping norm', default=1.0)
    parser.add_argument("--lr", type=float, help='learning rate', default=2e-2)
    parser.add_argument("--betas", type=tuple, help='(adamhf)optimizer betas', default=(0.9,0.999))
    parser.add_argument("--eps", type=float, help='(adamhf)optimizer eps', default=1e-8)
    parser.add_argument("--weight_decay", type=float, help='(adamhf)optimizer weight decay', default=0.0)
    parser.add_argument("--output_dir", type=str, help = 'output folder path', default='/home/zhengjiaying/project/TFedGLM/checkpoint/default')
    parser.add_argument("--save_step", type=int, help = 'step to save the prefix encoder', default=1)
    parser.add_argument("--max_step", type=int, help='number of max training steps, should be same with serve side!', default=10)
    parser.add_argument("--do_train", type=bool, help='Whether to run training.', default=False)
    parser.add_argument("--do_eval", type=bool, help='Whether to run eval on the dev set.', default=False)
    parser.add_argument("--do_predict", type=bool, help='Whether to run predictions on the test set.', default=False)
    # data arguments
    parser.add_argument("--data_fold", type=Optional[str], default=None)
    parser.add_argument("--cache_dir", type=Optional[str], default=None)
    parser.add_argument("--use_auth_token", type=bool, default=True)
    parser.add_argument("--max_source_length", type=int, default = 20)
    parser.add_argument("--max_target_length", type=int, default = 10)
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
    
    parser.add_argument("--train_file", type=str,  default='train.jsonl')
    parser.add_argument("--validation_file", type=str,  default='val.jsonl') 
    parser.add_argument("--test_file", type=str,  default='val.jsonl')
    
    
        
    args = parser.parse_args()

    return args

# get datalist

def get_data_list(modelA_args, tokenizer):
    # Load dataset
    data_dir = modelA_args.data_fold
    data_files = {}
    prefix = modelA_args.source_prefix if modelA_args.source_prefix is not None else ""
    if modelA_args.train_file is not None:
        data_files["train"] = modelA_args.train_file
        extension = modelA_args.train_file.split(".")[-1]
    if modelA_args.validation_file is not None:
        # data_files["validation"] = modelA_args.validation_file
        data_files["validation"] = modelA_args.train_file
        extension = modelA_args.validation_file.split(".")[-1]
    if modelA_args.test_file is not None:
        data_files["test"] = modelA_args.test_file
        extension = modelA_args.test_file.split(".")[-1]
    # print(model_args.use_auth_token) # False
    raw_datasets = load_dataset(
        'json',
        data_dir=data_dir,
        data_files=data_files,
        cache_dir=modelA_args.cache_dir, 
        use_auth_token=True if modelA_args.use_auth_token else None,
        # use_auth_token=None,
    )
    if modelA_args.do_train:
        column_names = raw_datasets["train"].column_names # boolq question passage label idx
    elif modelA_args.do_eval:
        # column_names = raw_datasets["validation"].column_names
        column_names = raw_datasets["train"].column_names
    elif modelA_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        print("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
    
    passage_column = modelA_args.passage_column
    passage_column2 = modelA_args.passage2_column
    premise_column = modelA_args.premise_column
    question_column = modelA_args.question_column
    answer_column = modelA_args.answer_column
    history_column = modelA_args.history_column   
    
    # Temporarily set max_target_length for training.
    max_target_length = modelA_args.max_target_length

    def preprocess_function_eval(examples):
        inputs, targets = [], []
        # print(len(examples[passage_column]))
        for i in range(len(examples[passage_column])):
            print(i)
            if examples[passage_column][i] : # :and examples[answer_column][i]
                query = examples[passage_column][i]  # origin cnn xsum
                answer = examples[answer_column][i]

                if history_column is None or len(examples[history_column][i]) == 0:
                    prompt = query
                else:
                    prompt = ""
                    history = examples[history_column][i]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

                inputs.append(prompt)
                # targets.append(examples[answer_column][i])
                targets.append(answer)
                # targets.append('Yes' if examples[answer_column][i] else 'No')


        model_inputs = tokenizer(inputs, max_length=modelA_args.max_source_length, truncation=True, padding=True)
        labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)

        if modelA_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def preprocess_function_train(examples):
        max_seq_length = modelA_args.max_source_length + modelA_args.max_target_length

        model_inputs = {
            "input_ids": [],
            "labels": [],
        }
        for i in range(len(examples[passage_column])):
            if examples[passage_column][i] : # : and examples[answer_column][i]# in soperglue some samples' label is bool 
                query, answer = examples[passage_column][i], examples[answer_column][i]

                if history_column is None:
                    prompt = query
                else:
                    prompt = ""
                    history = examples[history_column][i]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

                prompt = prefix + prompt
                a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)

                if len(a_ids) > 1024:
                    continue
                b_ids = tokenizer.encode(text=answer, add_special_tokens=False)

                if len(a_ids) > modelA_args.max_source_length - 1:
                    a_ids = a_ids[: modelA_args.max_source_length - 1]

                if len(b_ids) > modelA_args.max_target_length - 2:
                    b_ids = b_ids[: modelA_args.max_target_length - 2]

                input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

                context_length = input_ids.index(tokenizer.bos_token_id)
                mask_position = context_length - 1
                labels = [-100] * context_length + input_ids[mask_position+1:]
                
                pad_len = max_seq_length - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len

                labels = labels + [tokenizer.pad_token_id] * pad_len
                if modelA_args.ignore_pad_token_for_loss:
                    labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)

        return model_inputs
    
    dataset = None
    # dataset
    if modelA_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]

        if modelA_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), modelA_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        # with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(
                preprocess_function_train,
                batched=True,
                num_proc=modelA_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not modelA_args.overwrite_cache,
                # desc="Running tokenizer on train dataset",
            )
        dataset = train_dataset

    if modelA_args.do_eval:
        max_target_length = modelA_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        # eval_dataset = raw_datasets["train"]
        if modelA_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), modelA_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        eval_dataset = eval_dataset.map(
                preprocess_function_eval,
                batched=True,
                num_proc=modelA_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not modelA_args.overwrite_cache,
                # desc="Running tokenizer on validation dataset",
            )
        dataset = eval_dataset

        print('length of eval dataset',len(eval_dataset))

    if modelA_args.do_predict:
        max_target_length = modelA_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]

        if modelA_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), modelA_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
            
        predict_dataset = predict_dataset.map(
                preprocess_function_eval,
                batched=True,
                num_proc=modelA_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not modelA_args.overwrite_cache,
                # desc="Running tokenizer on prediction dataset",
            )
        dataset = predict_dataset

        print('length of predicted dataset',len(predict_dataset))
    
    print('length of dataset',len(dataset))

    # Data collator
    label_pad_token_id = -100 if modelA_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=None,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=False
    )
    # dataloader
    generator = torch.Generator()
    generator.manual_seed(modelA_args.seed)
    sampler = RandomSampler(dataset, generator=generator)

    if modelA_args.do_train:
        dataloader = DataLoader(
                train_dataset,
                batch_size=modelA_args.batch_size,
                sampler=sampler,
                collate_fn=data_collator,
                drop_last=modelA_args.dataloader_drop_last,
                num_workers=modelA_args.dataloader_num_workers,
                pin_memory=modelA_args.dataloader_pin_memory,
                worker_init_fn=seed_worker,
            )
    elif modelA_args.do_eval:
        dataloader = DataLoader(
                eval_dataset,
                batch_size=modelA_args.batch_size,
                sampler=sampler,
                collate_fn=data_collator,
                drop_last=modelA_args.dataloader_drop_last,
                num_workers=modelA_args.dataloader_num_workers,
                pin_memory=modelA_args.dataloader_pin_memory,
                worker_init_fn=seed_worker,
            )
    elif modelA_args.do_predict:
        dataloader = DataLoader(
                predict_dataset,
                batch_size=modelA_args.batch_size,
                sampler=sampler,
                collate_fn=data_collator,
                drop_last=modelA_args.dataloader_drop_last,
                num_workers=modelA_args.dataloader_num_workers,
                pin_memory=modelA_args.dataloader_pin_memory,
                worker_init_fn=seed_worker,
            )
    # list dataloader
    data_list = []
    if modelA_args.do_train:
        modelA_args.max_step        
        num_epochs = (modelA_args.max_step // len(dataset)) + 1
        for epoch in range(num_epochs):
            epoch_data = []
            for batch in dataloader:
                epoch_data.append(batch)
            data_list.append(epoch_data)
    else:
        for batch in dataloader:
            data_list.append(batch)

    return data_list

def seed_worker(_):
    """
    Helper function to set worker seed during Dataloader initialization.
    """
    worker_seed = torch.initial_seed() % 2**32
    set_seed(worker_seed)


def main():
    modelA_args = FLparser()
    modelA_args.do_train = True
    modelA_args.data_fold = './data/QA/huatuo'
    modelA_args.max_step = 30000
    modelA_args.premise_column = 'premise'
    modelA_args.question_column = 'question'
    modelA_args.passage_column = 'hypothesis'
    modelA_args.answer_column = 'answer'

    tokenizer = AutoTokenizer.from_pretrained('chatglm-6b', trust_remote_code=True)
    datalist = get_data_list( modelA_args, tokenizer)
    print(len(datalist))
    print(len(datalist[0]))
    print(datalist[0][0])
    print(datalist[1][0]) 
    print(datalist[9][0]) # dict:{'input_ids':[int], 'labels':[int]}

if __name__ == "__main__":
    main()
