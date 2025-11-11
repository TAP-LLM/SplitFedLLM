import sys 
import os
import torch
import torch.nn as nn
from transformers.optimization import AdamW
import transformers
import argparse
from datasets import load_dataset
from collections import OrderedDict
import collections
from typing import Dict, Tuple

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


def FLparser(): # use glm arguments with argument.py
    # Parse command line argument `partition`
    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    parser = argparse.ArgumentParser(description="Llama2-finetuning")
    # model args    
    # parser.add_argument("--batch_size", type=int, help="traing batch size", default=1) 
    # training args
    parser.add_argument("--seed", type=int, help="random seed", default=42)
    parser.add_argument("--output_dir", type=str, help = 'output folder path', default='/home/zhengjiaying/project/TFedGLM/checkpoint/default')

    parser.add_argument("--max_step", type=int, help='number of max training steps, should be same with serve side!', default=10)
    parser.add_argument("--do_train", type=bool, help='Whether to run training.', default=False)
    parser.add_argument("--do_eval", type=bool, help='Whether to run eval on the dev set.', default=False)
    parser.add_argument("--do_predict", type=bool, help='Whether to run predictions on the test set.', default=False)
    # data arguments
    parser.add_argument("--data_fold", type=Optional[str], default=None)
    parser.add_argument("--data_cache_dir", type=Optional[str], default=None)
    parser.add_argument("--data_use_auth_token", type=bool, default=True)
    parser.add_argument("--max_source_length", type=int, default = 20)
    parser.add_argument("--max_target_length", type=int, default = 10)
    parser.add_argument("--source_prefix", type=Optional[str], default=None)
    parser.add_argument("--passage_column", type=Optional[str], default=None)
    parser.add_argument("--passage2_column", type=Optional[str], default=None)
    parser.add_argument("--premise_column", type=Optional[str], default=None)
    parser.add_argument("--question_column", type=Optional[str], default=None)
    parser.add_argument("--answer_column", type=Optional[str], default=None) 
    parser.add_argument("--history_column", type=Optional[str], default=None) 
    parser.add_argument("--data_preprocessing_num_workers", type=int, default=1)
    parser.add_argument("--data_overwrite_cache", type=Optional[str], default=None) 
    parser.add_argument("--ignore_pad_token_for_loss", type=bool, default=True)

    parser.add_argument("--dataloader_num_workers_1", type=int, default=1)
    parser.add_argument("--dataloader_pin_memory_1", type=bool, default=True)
    parser.add_argument("--dataloader_drop_last_1", type=bool, default=True)
    
    parser.add_argument("--data_max_train_samples", type=Optional[int], default=None)
    parser.add_argument("--data_max_eval_samples", type=Optional[int], default=None)  
    
    parser.add_argument("--train_file", type=str,  default='train.jsonl')
    parser.add_argument("--validation_file", type=str,  default='val.jsonl') 
    parser.add_argument("--test_file", type=str,  default='test.jsonl')
       
    args = parser.parse_args()

    return args

# get datalist



def get_dataset_no_pad(args, tokenizer):
    # 以下部分需要改进为数据处理函数以及dataloader
    # Load dataset
    data_dir = args.data_fold
    data_files = {}
    prefix = args.source_prefix if args.source_prefix is not None else ""
    if args.train_file is not None:
        data_files["train"] = args.train_file
        extension = args.train_file.split(".")[-1]
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
        # data_files["validation"] = args.train_file
        extension = args.validation_file.split(".")[-1]
    if args.test_file is not None:
        data_files["test"] = args.test_file
        extension = args.test_file.split(".")[-1]
    # print(model_args.data_use_auth_token) # False
    raw_datasets = load_dataset(
        'json',
        data_dir=data_dir,
        data_files=data_files,
        cache_dir=args.cache_dir, 
        use_auth_token=True if args.use_auth_token else None,
        # data_use_auth_token=None,
    )
    if args.do_train:
        column_names = raw_datasets["train"].column_names # boolq question passage label idx
    elif args.do_eval:
        column_names = raw_datasets["validation"].column_names
        # column_names = raw_datasets["train"].column_names
    elif args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        print("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
    
    passage_column = args.passage_column
    passage_column2 = args.passage2_column
    premise_column = args.premise_column
    question_column = args.question_column
    answer_column = args.answer_column
    history_column = args.history_column   
    
    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length  #目前是512

    def print_dataset_example(example):
        print("input_ids",example["input_ids"])
        print("input_ids_length",len(example["input_ids"]))
        # print("inputs", tokenizer.decode(example["input_ids"]), skip_special_tokens=True)
        print("label_ids", example["labels"])
        print("label_ids_length", len(example["labels"]))
        # print("labels", tokenizer.decode(example["labels"]), skip_special_tokens=True)

    def preprocess_function_eval(examples):
        max_seq_length = args.max_source_length + args.max_target_length
        inputs, targets = [], []
        model_inputs = {
            "input_ids": [],
            "labels": [],
            "id": [],            # for CoQA 官方测试代码
            # "turn_id": [],  # for CoQA 官方测试代码
        }
        # print(len(examples[passage_column]))
        for i in range(len(examples[passage_column])):
            # print(i)
            if examples[passage_column][i] : # :and examples[answer_column][i]
                #region 其他数据集
                # query = examples[passage_column][i]  # origin cnn xsum
                # boolq
                # query = examples[passage_column][i] + '. Question: {}? Answer:'.format(examples[question_column][i])
                # wic
                # query = '\"{}\" / \"{}\" Similar sense of {}? '.format(examples[passage_column][i], 
                #                                                        examples[passage_column2][i],
                #                                                        examples[question_column][i])
                # answer = 'Yes' if examples[answer_column][i] else 'No'

                # copa
                # query = 'Choice A: \"{}\" or Choice B: \"{}\" Premise: {} The {} is '.format(examples[passage_column][i],
                #                                          examples[passage_column2][i],
                #                                          examples[premise_column][i],
                #                                          examples[question_column][i])
                # answer = 'A' if examples[answer_column][i] else 'B'

                # wsc only true
                # query = '[{}]The pronoun \'*{}*\' refers to'.format(examples[passage_column][i], 
                #                                                   examples[question_column][i]['span2_text'])
                # answer = examples[question_column][i]['span1_text']

                # full wsc
                # query = '[INST][{}] Is the pronoun \'{}\' refers to \'{}\' ?[/INST]'.format(examples[passage_column][i], 
                #                                                   examples[question_column][i]['span2_text'],
                #                                                   examples[question_column][i]['span1_text'])
                                                                                                            
                # answer = 'Yes' if examples[answer_column][i] else 'No'

                # rte
                # query = "[INST]Hypothesis: \"{}\", premise:\"{}\", If the hypothesis is entailed by the premise? Yes or No? Answer:[/INST]".format(examples[passage_column][i],examples[premise_column][i])
                # answer = 'Yes' if examples[answer_column][i]=='entailment' else 'No'

                # huatuo
                # query = '[INST]Question: {}? Answer:[/INST]'.format(examples[passage_column][i])
                # answer = examples[answer_column][i]

                # cb
                # query = "[INST]Hypothesis: \"{}\", premise:\"{}\", If the hypothesis is entailed by the premise? Yes, No or Maybe? Answer:[/INST]".format(examples[question_column][i],examples[premise_column][i])
                # answer = ''
                # if examples[answer_column][i]=='entailment':
                #     answer = 'Yes'
                # elif examples[answer_column][i]=='contradiction':
                #     answer = 'No'
                # elif examples[answer_column][i]== 'neutral':
                #     answer = 'Maybe'
                
                # record
                # query = '[{}][{}]'.format(examples[passage_column][i]['text'], examples[question_column][i][0]['query'])
                # query = query.replace('@placeholder', '[MASK]')
                # # print(query)
                # answer = ''
                # for j in range(len(examples[question_column][i][0]['answers'])):
                # # answer = '[{}]'.format(examples[question_column][i][0]['answers'][0]['text'])
                #     answer += ' [{}]'.format(examples[question_column][i][0]['answers'][j]['text'])

                # processed_data = []
                # # Friends
                # for utterance in examples[passage_column][i]:
                #     speaker = utterance['speakers']
                #     tokens = utterance['tokens']
                #     processed_data.append(f'{speaker}: {tokens}')
                # processed_data = "\n".join(processed_data)
                # query = "Instruction: Please identify who \'[MASK]\' represents in the input according to the dialogue.\n Dialogue: \'{}\'. \nInput:\'{}\'. \nOutput:\'[MASK]\' represents:</s>".format(processed_data, examples[question_column][i])
                # query = query.replace('@placeholder', '[MASK]')
                # answer = examples[answer_column][i]                
                # # print(f"answer:{answer}")
                # if history_column is None or len(examples[history_column][i]) == 0:
                #     prompt = query
                # else:
                #     prompt = ""
                #     history = examples[history_column][i]
                #     for turn_idx, (old_query, response) in enumerate(history):
                #         prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                #     prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
                #endregion

                #region 目前的数据集配置
                """=====================================================CoQA====================================================="""
        #         UtteranceTurns = len(examples[question_column][i])
        #         QAPairs = "The story is \'{}\'.".format(examples[passage_column][i])
        #         for j in range(UtteranceTurns):
        #             if j==0:
        #                 QAPairs += "My question is \'{}\'.[/INST]{}</s>".format(examples[question_column][i][0]["input_text"],
        #                                                                     examples[answer_column][i][0]["input_text"])
        #             else:
        #                 QAPairs += "<s>[INST]{}[/INST]{}</s>".format(examples[question_column][i][j]["input_text"],
        #                                                             examples[answer_column][i][j]["input_text"])
        #         query = "<s>[INST] <<SYS>>You are a helpful assistant.You need to answer my question base on the provided story.<</SYS>>" + QAPairs
        #         if history_column is None:
        #             prompt = query
        #         # 多轮对话数据集单独写  
        #         prompt = prefix + prompt
        #         a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
        #         if len(a_ids) > args.max_source_length - 1:
        #             #a_ids = a_ids[: args.max_source_length - 1] # len():127  ,这里是截取前面一部分
        #             a_ids = a_ids[-(args.max_source_length - 1):] # len():127  这里是截取后面一部分
        #         input_ids = a_ids


        #         # padding  所有数据集通用             
        #         # pad_len = max_seq_length - len(input_ids)
        #         # atten_mask= [0]* pad_len + [1]*len(input_ids)
        #         # input_ids = [tokenizer.pad_token_id] * pad_len + input_ids  #padding 在左边
        #         label_ids = input_ids
        #         #labels = labels + [tokenizer.eos_token_id] +[-100]* (len(input_ids) - len(labels)-1)  # + [-100]-1
        #         # if args.ignore_pad_token_for_loss:
        #         #     label_ids = [(l if l != tokenizer.pad_token_id else -100) for l in label_ids]  #pad token会被标记为-100

        #         model_inputs["input_ids"].append(input_ids)
        #         model_inputs["labels"].append(label_ids)
        #         # model_inputs["attention_mask"].append(atten_mask)

        # return model_inputs
                """=====================================================MultiRC====================================================="""
        #         system_prompt="<s>[INST] <<SYS>>You are a helpful assistant.You need to judge whether the question is Yes or No base on the provided passage.<</SYS>>"
        #         passage_i = 'The passage is \"{}\".'.format(examples[passage_column][i]['text'])

        #         for j in range(len(examples[passage_column][i]['questions'])):
        #             for k in range(len(examples[passage_column][i]['questions'][j]['answers'])):
        #                 query  = "Question:\"{}\". The possible answer is \"{}\". Is it correct?[/INST]".format(
        #                                 examples[passage_column][i]['questions'][j]['question'],
        #                                 examples[passage_column][i]['questions'][j]['answers'][k]['text']) 
        #                 label_k = examples['passage'][i]['questions'][j]['answers'][k]['label'] 
        #                 answer = 'Yes' if label_k else 'No'
        #                 # query = query + answer

        #                 prompt = system_prompt + passage_i + query
        #                 prompt = prefix + prompt
        #                 a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
        #                 # print(f"第{i}个样本的第{j}个问题的prompt长度为{len(a_ids)}")
        #                 b_ids = tokenizer.encode(text=answer, add_special_tokens=False)


        #                 if len(a_ids) > args.max_source_length - 1:
        #                     #a_ids = a_ids[: args.max_source_length - 1] # len():127  ,这里是截取前面一部分
        #                     a_ids = a_ids[-(args.max_source_length - 1):] # len():127  这里是截取后面一部分
        #                 input_ids = a_ids

                       
        #                 if len(b_ids) > args.max_target_length - 2:
        #                     b_ids = b_ids[: args.max_target_length - 2]

        #                 # padding  所有数据集通用             
        #                 pad_len = max_seq_length - len(input_ids)    #虽然在batchsize为1的时候原本不需要pad，但是为了不传递pkv，因此还是pad
        #                 atten_mask= [0]* pad_len + [1]*len(input_ids)
        #                 input_ids = [tokenizer.pad_token_id] * pad_len + input_ids  #padding 在左边
        #                 label_ids = b_ids
        #                 if args.ignore_pad_token_for_loss:
        #                     label_ids = [(l if l != tokenizer.pad_token_id else -100) for l in label_ids]  #pad token会被标记为-100

        #                 pid = tokenizer.encode(text=str(examples["idx"][i]), add_special_tokens=False)
        #                 qid = tokenizer.encode(text=str(examples[passage_column][i]['questions'][j]['idx']), add_special_tokens=False)
        #                 aid = tokenizer.encode(text=str(examples[passage_column][i]['questions'][j]['answers'][k]['idx']), add_special_tokens=False)

        #                 model_inputs["input_ids"].append(input_ids)
        #                 model_inputs["labels"].append(label_ids)
        #                 model_inputs["attention_mask"].append(atten_mask)
        #                 model_inputs["pid"].append(pid) 
        #                 model_inputs["qid"].append(qid)
        #                 model_inputs["aid"].append(aid) 
        # return model_inputs
                """=====================================================Xsum====================================================="""
                # query = "<s>[INST] <<SYS>>You are a helpful assistant.You need to summarize an abstract base on the provided passage.<</SYS>>The passage is \'{}\'.[/INST]".format(examples[passage_column][i])
                # answer = examples[answer_column][i]

                #endregion

                """=====================================================Record====================================================="""
                query = "<s>[INST] <<SYS>>You are a helpful assistant.You need to predict what @placeholder represents in the query base on the provided passage.<</SYS>>The passage is \'{}\'.The query is \'{}\'.[/INST]".format(examples[passage_column][i],examples[passage_column][i])
                answer = examples[answer_column][i]


                if history_column is None:
                    prompt = query

                prompt = prefix + prompt
                a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
                # print(len(a_ids))
                if len(a_ids) > args.max_source_length - 1:
                    #a_ids = a_ids[: args.max_source_length - 1] # len():127  ,这里是截取前面一部分
                    a_ids = a_ids[-(args.max_source_length - 1):] # len():127  这里是截取后面一部分
                
                b_ids = tokenizer.encode(text=answer, add_special_tokens=False)
                
                #以下是CoQA
                # b_ids = []
                # sep_token_id= 19603
                # # 对每个答案分别编码，并插入分隔符,便于后面根据分隔符号区分不同的答案
                # for answer in expected_answer:
                #     answer_ids = tokenizer.encode(text=answer, add_special_tokens=False)
                #     b_ids.extend(answer_ids)
                #     b_ids.append(sep_token_id)  # 插入分隔符
                # # 去掉最后一个 [SEP]，因为它不需要
                # b_ids = b_ids[:-1]

                # print(f"b_ids:{b_ids}")
                if len(a_ids) > args.max_source_length - 1:
                    a_ids = a_ids[: args.max_source_length - 1]

                if len(b_ids) > args.max_target_length - 2:
                    b_ids = b_ids[: args.max_target_length - 2]

                a_ids = a_ids[1:] if a_ids[0] == 1 else a_ids
                label_ids = b_ids[1:] if b_ids[0]== 1 else b_ids 
                input_ids = [tokenizer.bos_token_id] + a_ids
                label_ids = label_ids + [tokenizer.eos_token_id]
                # input_ids = input_ids + [2] * (len(label_ids)-len(input_ids))
                # print(f"label_ids:{label_ids}")
                # padding
                # pad_len = args.max_source_length + args.max_target_length
                # input_ids = input_ids + [tokenizer.unk_token_id] * (pad_len - len(input_ids))
                # labels = label_ids + [tokenizer.unk_token_id] * (pad_len - len(label_ids))
                # if args.ignore_pad_token_for_loss:
                #     labels = [(l if l != tokenizer.unk_token_id else -100) for l in labels]

                #记录当前数据的id
                id = tokenizer.encode(text=examples["idx"][i], add_special_tokens=False)

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(label_ids)
                model_inputs["id"].append(id)   # for CoQA\xum 测试代码
                # model_inputs["turn_id"].append(examples["turn_id"][i]) # for CoQA 官方测试代码
        return model_inputs

        #region
        #         # record
        #         # a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
        #         # # print(len(a_ids))
        #         # if len(a_ids) > 1024:
        #         #     continue

        #         inputs.append(prompt)
        #         # targets.append(examples[answer_column][i])
        #         targets.append(answer)
        #         # targets.append('Yes' if examples[answer_column][i] else 'No')

        #         # full multirc
        #         # passage_i = '\"{}\"'.format(examples[passage_column][i]['text'])
        #         # for j in range(len(examples[passage_column][i]['questions'])):
        #         #     for k in range(len(examples[passage_column][i]['questions'][j]['answers'])):
        #         #         query  = passage_i + " Question:{} Is it answer \"{}\"".format(
        #         #                          examples[passage_column][i]['questions'][j]['question'],
        #         #                          examples[passage_column][i]['questions'][j]['answers'][k]['text']) 
        #         #         answer = examples['passage'][i]['questions'][j]['answers'][k]['label']
                        
        #         #         if history_column is None or len(examples[history_column][i]) == 0:
        #         #             prompt = query
        #         #         else:
        #         #             prompt = ""
        #         #             history = examples[history_column][i]
        #         #             for turn_idx, (old_query, response) in enumerate(history):
        #         #                 prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
        #         #             prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

        #         #         inputs.append(prompt)
        #         #         targets.append('Yes' if answer else 'No')
  
        #         # record 
        #         # cloze_question = examples[passage_column][i] + examples[question_column][i]

        # model_inputs = tokenizer(inputs, max_length=args.max_source_length, truncation=True, padding=False)
        # # print(model_inputs['input_ids'][0])
        # labels = tokenizer(text_target=targets, max_length=args.max_target_length, truncation=True, padding=False)

        # # padding
        # # if without padding, ValueError: Expected input batch_size (255) to match target batch_size (7).
        # labels['input_ids'] = [label + [tokenizer.pad_token_id] * (args.max_source_length-len(label)) for label in labels["input_ids"]]
        
        # if args.ignore_pad_token_for_loss: 
        #     labels["input_ids"] = [
        #         [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        #     ]
        # model_inputs["labels"] = labels["input_ids"]
        # print(len(labels["input_ids"][0]))

        # return model_inputs
        # endregion

    def preprocess_function_train(examples):
        max_seq_length = args.max_source_length + args.max_target_length
        model_inputs = {
            "input_ids": [],
            "labels": [],
            "attention_mask": []
        }
        for i in range(len(examples[passage_column])):
            if examples[passage_column][i] : # : and examples[answer_column][i]# in soperglue some samples' label is bool 
               
                # region
                # query, answer = examples[passage_column][i], examples[answer_column][i]
                # boolq
                # query = examples[passage_column][i] + '. Question: {}? Answer:'.format(examples[question_column][i])

                # wic
                # query = '\"{}\" / \"{}\" Similar sense of {}? '.format(examples[passage_column][i], 
                #                                                        examples[passage_column2][i],
                #                                                        examples[question_column][i])

                # answer = 'Yes' if examples[answer_column][i] else 'No'

                # copa trainbatchsize1
                # query = 'Choice A: \"{}\" or Choice B: \"{}\" Premise: {} The {} is '.format(examples[passage_column][i],
                #                                          examples[passage_column2][i],
                #                                          examples[premise_column][i],
                #                                          examples[question_column][i])
                # answer = 'A' if examples[answer_column][i] else 'B'

                # wsc only true
                # query = '[{}]The pronoun \'*{}*\' refers to'.format(examples[passage_column][i], 
                #                                                   examples[question_column][i]['span2_text'])
                # answer = examples[question_column][i]['span1_text']

                # full wsc
                # query = '[INST][{}] The pronoun \'*{}*\' refers to {} ?[/INST]'.format(examples[passage_column][i], 
                #                                                   examples[question_column][i]['span2_text'],
                #                                                   examples[question_column][i]['span1_text'])
                                                                                                            
                # answer = 'Yes' if examples[answer_column][i] else 'No'

                # # rte
                # query = "[INST]Hypothesis: \"{}\", premise:\"{}\", If the hypothesis is entailed by the premise? Yes or No? Answer:[/INST]".format(examples[passage_column][i],examples[premise_column][i])
                # answer = 'Yes' if examples[answer_column][i]=='entailment' else 'No'

                # huatuo
                #query = '[INST]Question: {}? Answer:[/INST]'.format(examples[passage_column][i])
                # '[INST]Question: 我这几天感觉宫颈部位有点不舒服，不知道是怎么了，于是就赶紧去做了妇科检查，后来知道得的是宫颈炎症，听说这个病不太好治，也不知道该怎么治疗才好些。宫颈炎这个病应该怎样治疗？? Answer:[/INST]'
                #answer = examples[answer_column][i]
                # '宫颈炎是一种常见的妇科疾病，治疗建议不要长期服用消炎药，会造成耐药性和菌群失调，造成更严重的炎症感染。中成药推荐椿乳凝胶，本品是目前市场上首个用于宫颈炎的国药准字号凝胶剂。采用凝胶剂将中药用于慢性宫颈炎的治疗，阴道给药，可以使药物直达病灶部位，长时间粘附而起作用，能延长给药时间，增加药物的吸收量，从而达到提高局部浓度和生物有效性的目的。'
                
                # cb
                # query = "\"{}\"? | [MASK], \"{}\"".format(examples[question_column][i],examples[premise_column][i])
                # query = "[INST]Hypothesis: \"{}\", premise:\"{}\", If the hypothesis is entailed by the premise? Yes, No or Maybe? Answer:[/INST]".format(examples[question_column][i],examples[premise_column][i])
                
                # answer = ''
                # if examples[answer_column][i]=='entailment':
                #     answer = 'Yes'
                # elif examples[answer_column][i]=='contradiction':
                #     answer = 'No'
                # elif examples[answer_column][i]== 'neutral':
                #     answer = 'Maybe'

                # record
                # query = '[{}][{}]'.format(examples[passage_column][i]['text'], examples[question_column][i][0]['query'])
                # query = query.replace('@placeholder', '[MASK]')
                # multi-query/multi-answer

                # answer = '[{}]'.format(examples[question_column][i][0]['answers'][0]['text'])

                # full multirc
                # passage_i = '\"{}\"'.format(examples[passage_column][i]['text'])
                # for j in range(len(examples[passage_column][i]['questions'])):
                #     for k in range(len(examples[passage_column][i]['questions'][j]['answers'])):
                #         query  = passage_i + " Question:{} Is it answer \"{}\" ?".format(
                #                          examples[passage_column][i]['questions'][j]['question'],
                #                          examples[passage_column][i]['questions'][j]['answers'][k]['text']) 
                #         label_k = examples['passage'][i]['questions'][j]['answers'][k]['label'] 
                #         answer = 'Yes' if label_k else 'No'

                #         if history_column is None:
                #             prompt = query
                #         else:
                #             prompt = ""
                #             history = examples[history_column][i]
                #             for turn_idx, (old_query, response) in enumerate(history):
                #                 prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                #                 prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

                #         prompt = prefix + prompt
                #         a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
                #         b_ids = tokenizer.encode(text=answer, add_special_tokens=False)

                #         if len(a_ids) > args.max_source_length - 1:
                #             a_ids = a_ids[: args.max_source_length - 1]

                #         if len(b_ids) > args.max_target_length - 2:
                #             b_ids = b_ids[: args.max_target_length - 2]

                #         input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

                #         context_length = input_ids.index(tokenizer.bos_token_id)
                #         mask_position = context_length - 1
                #         labels = [-100] * context_length + input_ids[mask_position+1:]
                
                #         pad_len = max_seq_length - len(input_ids)
                #         input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                #         labels = labels + [tokenizer.pad_token_id] * pad_len
                #         if args.ignore_pad_token_for_loss:
                #             labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]

                #         model_inputs["input_ids"].append(input_ids)
                #         model_inputs["labels"].append(labels)

                # # Friends
                # processed_data = []
                # for utterance in examples[passage_column][i]:
                #     speaker = utterance['speakers']
                #     tokens = utterance['tokens']
                #     processed_data.append(f'{speaker}: {tokens}')
                # processed_data = "\n".join(processed_data)
                # query = "Instruction: Please identify who \'[MASK]\' represents in the input according to the dialogue.\n Dialogue: \'{}\'. \nInput:\'{}\'. \nOutput:\'[MASK]\' represents:"exampl.format(processed_data, es[question_column][i])
                # query = query.replace('@placeholder', '[MASK]')
                # answer = examples[answer_column][i]

                # if history_column is None:
                #     prompt = query
                # else:
                #     prompt = ""
                #     history = examples[history_column][i]
                #     for turn_idx, (old_query, response) in enumerate(history):
                #         prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                #     prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
                #endregion

                # region 本次实验配置
                """=====================================================CoQA====================================================="""
                # UtteranceTurns = len(examples[question_column][i])
                # QAPairs = "The story is \'{}\'.".format(examples[passage_column][i])
                # for j in range(UtteranceTurns):
                #     if j==0:
                #         QAPairs += "My question is \'{}\'.[/INST]{}</s>".format(examples[question_column][i][0]["input_text"],
                #                                                             examples[answer_column][i][0]["input_text"])
                #     else:
                #         QAPairs += "<s>[INST]{}[/INST]{}</s>".format(examples[question_column][i][j]["input_text"],
                #                                                     examples[answer_column][i][j]["input_text"])
                # query = "<s>[INST] <<SYS>>You are a helpful assistant.You need to answer my question base on the provided story.<</SYS>>" + QAPairs
                # if history_column is None:
                #     prompt = query

        #         # 多轮对话数据集单独写  
        #         prompt = prefix + prompt
        #         a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)

        #         if len(a_ids) > args.max_source_length - 1:
        #             #a_ids = a_ids[: args.max_source_length - 1] # len():127  ,这里是截取前面一部分
        #             a_ids = a_ids[-(args.max_source_length - 1):] # len():127  这里是截取后面一部分
        #         input_ids = a_ids


        #         # padding  所有数据集通用             
        #         # pad_len = max_seq_length - len(input_ids)
        #         # atten_mask= [0]* pad_len + [1]*len(input_ids)
        #         # input_ids = [tokenizer.pad_token_id] * pad_len + input_ids  #padding 在左边
        #         label_ids = input_ids
        #         #labels = labels + [tokenizer.eos_token_id] +[-100]* (len(input_ids) - len(labels)-1)  # + [-100]-1
        #         # if args.ignore_pad_token_for_loss:
        #         #     label_ids = [(l if l != tokenizer.pad_token_id else -100) for l in label_ids]  #pad token会被标记为-100

        #         model_inputs["input_ids"].append(input_ids)
        #         model_inputs["labels"].append(label_ids)
        #         # model_inputs["attention_mask"].append(atten_mask)

        # return model_inputs
                """=====================================================MultiRC====================================================="""
        #         system_prompt="<s>[INST] <<SYS>>You are a helpful assistant.You need to judge whether the question is Yes or No base on the provided passage.<</SYS>>"
        #         passage_i = 'The passage is \"{}\".'.format(examples[passage_column][i]['text'])

        #         for j in range(len(examples[passage_column][i]['questions'])):
        #             for k in range(len(examples[passage_column][i]['questions'][j]['answers'])):
        #                 query  = "Question:\"{}\". The possible answer is \"{}\". Is it correct?[/INST]".format(
        #                                 examples[passage_column][i]['questions'][j]['question'],
        #                                 examples[passage_column][i]['questions'][j]['answers'][k]['text']) 
        #                 label_k = examples['passage'][i]['questions'][j]['answers'][k]['label'] 
        #                 answer = 'Yes</s>' if label_k else 'No</s>'
        #                 query = query + answer



        #                 prompt = system_prompt + passage_i + query

        #                 prompt = prefix + prompt
        #                 a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
        #                 # print(f"第{i}个样本的第{j}个问题的prompt长度为{len(a_ids)}")

        #                 if len(a_ids) > args.max_source_length - 1:
        #                     #a_ids = a_ids[: args.max_source_length - 1] # len():127  ,这里是截取前面一部分
        #                     a_ids = a_ids[-(args.max_source_length - 1):] # len():127  这里是截取后面一部分
        #                 input_ids = a_ids


        #                 # padding  所有数据集通用             
        #                 pad_len = max_seq_length - len(input_ids)    #虽然在batchsize为1的时候原本不需要pad，但是为了不传递pkv，因此还是pad
        #                 atten_mask= [0]* pad_len + [1]*len(input_ids)
        #                 input_ids = [tokenizer.pad_token_id] * pad_len + input_ids  #padding 在左边
        #                 label_ids = input_ids
        #                 if args.ignore_pad_token_for_loss:
        #                     label_ids = [(l if l != tokenizer.pad_token_id else -100) for l in label_ids]  #pad token会被标记为-100

        #                 model_inputs["input_ids"].append(input_ids)
        #                 model_inputs["labels"].append(label_ids)
        #                 model_inputs["attention_mask"].append(atten_mask)

        # return model_inputs
        #endregion
                """=====================================================Xsum====================================================="""
        #         query = "<s>[INST] <<SYS>>You are a helpful assistant.You need to summarize an abstract base on the provided passage.<</SYS>>The passage is \'{}\'.[/INST]{}</s>".format(examples[passage_column][i], examples[answer_column][i])
        #         if history_column is None:
        #             prompt = query

        #         # 多轮对话数据集单独写  
        #         prompt = prefix + prompt
        #         a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
        #         # print(f"第{i}个样本的prompt长度为{len(a_ids)}")

        #         if len(a_ids) > args.max_source_length - 1:
        #             a_ids = a_ids[: args.max_source_length - 1] # len():127  ,这里是截取前面一部分
        #         input_ids = a_ids

        #         # padding  所有数据集通用             
        #         pad_len = max_seq_length - len(input_ids)    #虽然在batchsize为1的时候原本不需要pad，但是为了不传递pkv，因此还是pad
        #         atten_mask= [0]* pad_len + [1]*len(input_ids)
        #         input_ids = [tokenizer.pad_token_id] * pad_len + input_ids  #padding 在左边
        #         label_ids = input_ids
        #         if args.ignore_pad_token_for_loss:
        #             label_ids = [(l if l != tokenizer.pad_token_id else -100) for l in label_ids]  #pad token会被标记为-100

        #         model_inputs["input_ids"].append(input_ids)
        #         model_inputs["labels"].append(label_ids)
        #         model_inputs["attention_mask"].append(atten_mask)

        # return model_inputs

                """=====================================================Record====================================================="""
                if len(model_inputs["input_ids"])>25000:
                    continue

                for j in range(len(examples[question_column][i])):
                    query = "<s>[INST] <<SYS>>You are a helpful assistant.You need to predict what @placeholder represents in the query based on the provided passage.<</SYS>>The passage is \'{}\'.The query is \'{}\'.What does @placeholder represent?[/INST]".format(examples[passage_column][i]["text"],examples[question_column][i][j]["query"])
                    answer = "{}".format(examples[question_column][i][j]["answers"][0]["text"])

                    if history_column is None:
                        prompt = query

                    prompt = prefix + prompt
                    a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
                    b_ids = tokenizer.encode(text=answer, add_special_tokens=False)

                    # 跳过长度超过700的
                    if len(a_ids)>700:
                        continue

                    if len(a_ids) > args.max_source_length - 1:
                        a_ids = a_ids[: args.max_source_length - 1] # len():127
                    if len(b_ids) > args.max_target_length - 2:
                        b_ids = b_ids[: args.max_target_length - 2]
                    a_ids = a_ids[1:] if a_ids[0] == 1 else a_ids
                    labels_ids = b_ids[1:] if b_ids[0]== 1 else b_ids 
                    input_ids = [tokenizer.bos_token_id] + a_ids + labels_ids + [tokenizer.eos_token_id]

                    # padding 所有数据集通用
                    pad_len = max_seq_length - len(input_ids)    #虽然在batchsize为1的时候原本不需要pad，但是为了不传递pkv，因此还是pad
                    atten_mask= [0]* pad_len + [1]*len(input_ids)
                    input_ids = [tokenizer.pad_token_id] * pad_len + input_ids  #padding 在左边
                    label_ids = input_ids
                    if args.ignore_pad_token_for_loss:
                        label_ids = [(l if l != tokenizer.pad_token_id else -100) for l in label_ids]  #pad token会被标记为-100

                    model_inputs["input_ids"].append(input_ids)
                    model_inputs["labels"].append(label_ids)
                    model_inputs["attention_mask"].append(atten_mask)

        return model_inputs
    
    dataset = None
    # dataset
    if args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        # with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(
                preprocess_function_train,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                # desc="Running tokenizer on train dataset",
            )
        dataset = train_dataset
        # print_dataset_example(train_dataset[0])

    if args.do_eval:
        max_target_length = args.max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        # eval_dataset = raw_datasets["train"]
        if args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        # eval_dataset = eval_dataset.select(range(1000))
        # print(len(eval_dataset))
        # with training_args.main_process_first(desc="validation dataset map pre-processing"):
        eval_dataset = eval_dataset.map(
                preprocess_function_eval,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                # desc="Running tokenizer on validation dataset",
            )
        dataset = eval_dataset
        print_dataset_example(eval_dataset[0])
        print('length of eval dataset',len(eval_dataset))

    if args.do_predict:
        if args.do_inference:
            max_target_length = args.val_max_target_length
            # 将prompt转换数据集的格式
            prompt = args.prompt
            inputs= []
            inputs.append(prompt)
            model_inputs = tokenizer(inputs, max_length = args.max_source_length, truncation=True, padding=False)
            #以下是74原本的代码
            # model_inputs = torch.tensor(model_inputs["input_ids"])
            # # 获取输入的形状
            # seq_len = 130

            # # 重新填充到原始长度
            # pad_length = seq_len - model_inputs.size(1)  #128
            # #pad_length = 2*seq_len - new_input.size(1) #256：问题和回答总token不超过256
            # padded_input = torch.nn.functional.pad(model_inputs, (pad_length, 0), value=3)

            # return padded_input
            input_ids = torch.tensor(model_inputs["input_ids"])
            seq_len = args.max_output_length+2 #前向传播推理时 输入的最大tokens限制
            # 重新填充到原始长度
            pad_length = seq_len - input_ids.size(1) 
            model_inputs["attention_mask"]= torch.cat((torch.zeros(pad_length, dtype=torch.int64), torch.ones(input_ids.size(1), dtype=torch.int64)),dim=0)
            model_inputs["attention_mask"]=model_inputs["attention_mask"].unsqueeze(0) #最前面增加一维
            padded_input = torch.nn.functional.pad(input_ids, (pad_length,0), value=3) # padding side为左边
            model_inputs["input_ids"]= padded_input

            return model_inputs

        else: 
            max_target_length = args.max_target_length
            if "test" not in raw_datasets:
                raise ValueError("--do_predict requires a test dataset")
            predict_dataset = raw_datasets["test"]
            if args.max_predict_samples is not None:
                max_predict_samples = min(len(predict_dataset), args.max_predict_samples)
                predict_dataset = predict_dataset.select(range(max_predict_samples))
            # with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                    preprocess_function_eval,
                    batched=True,
                    num_proc=args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not args.overwrite_cache,
                    # desc="Running tokenizer on prediction dataset",
                )
            dataset = predict_dataset
            # print_dataset_example(predict_dataset[0])
            print('length of predicted dataset',len(predict_dataset))
    
    print('length of dataset',len(dataset))
    #region
    # # Data collator
    # label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    # data_collator = DataCollatorForSeq2Seq(
    #     tokenizer,
    #     model=None,
    #     label_pad_token_id=label_pad_token_id,
    #     pad_to_multiple_of=None,
    #     padding=False
    # )

    # dataloader

    # generator = torch.Generator()
    # generator.manual_seed(args.seed)
    # sampler = RandomSampler(dataset, generator=generator)

    # dataloader = DataLoader(
    #         dataset,
    #         batch_size=args.batch_size,
    #         sampler=sampler,
    #         collate_fn=data_collator,
    #         drop_last=args.dataloader_drop_last_1,
    #         num_workers=args.dataloader_num_workers_1,
    #         pin_memory=args.dataloader_pin_memory_1,
    #         worker_init_fn=seed_worker,
    #     )
    #endregion
    # data_collator
    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    # data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    max_length = args.max_source_length + args.max_target_length
    data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, model=None,pad_to_multiple_of=None, return_tensors="pt",label_pad_token_id=label_pad_token_id , padding=False)
    # dataloader
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    sampler = RandomSampler(dataset, generator=generator)
    if args.do_train:
        dataloader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                sampler=sampler,
                collate_fn=data_collator,
                drop_last=args.dataloader_drop_last,
                num_workers=args.dataloader_num_workers,
                pin_memory=args.dataloader_pin_memory,
                worker_init_fn=seed_worker,
            )
    elif args.do_eval:
        dataloader = DataLoader(
                eval_dataset,
                batch_size=args.batch_size,
                sampler=sampler,
                collate_fn=data_collator,
                drop_last=args.dataloader_drop_last,
                num_workers=args.dataloader_num_workers,
                pin_memory=args.dataloader_pin_memory,
                worker_init_fn=seed_worker,
            )
    elif args.do_predict:
        dataloader = DataLoader(
                predict_dataset,
                batch_size=args.batch_size,
                sampler=sampler,
                collate_fn=data_collator,
                drop_last=args.dataloader_drop_last,
                num_workers=args.dataloader_num_workers,
                pin_memory=args.dataloader_pin_memory,
                worker_init_fn=seed_worker,
            )

    # list dataloader
    data_list = []
    if args.do_train:
        # data_list[0]为一个epoch, 每一个epoch有很多个batch, 每一个batch的keys为dict_keys(['input_ids', 'labels', 'attention_mask'])
        num_epochs = (args.max_step // (len(dataset)//args.batch_size)) + 1     
        # num_epochs = (args.max_step // len(dataset)) + 1
        for epoch in range(num_epochs):
            epoch_data = []
            for batch in dataloader:
                epoch_data.append(batch)
            data_list.append(epoch_data)
    else:
        for batch in dataloader:
            data_list.append(batch)

    return data_list
    #return dataset

def seed_worker(_):
    """
    Helper function to set worker seed during Dataloader initialization.
    """
    worker_seed = torch.initial_seed() % 2**32
    set_seed(worker_seed)


def main():
    args = FLparser()
    args.do_train = True
    args.data_fold = '/home/XXXX/TFed-GLM/data/QA/huatuo'
    args.max_step = 30000
    args.premise_column = 'premise'
    args.question_column = 'question'
    args.passage_column = 'hypothesis'
    args.answer_column = 'answer'

    tokenizer = AutoTokenizer.from_pretrained("/home/XXXX/SplitFederated-LLaMA/Models/Llama2-7B-chat-service", trust_remote_code=True)
    dataset = get_dataset_no_pad( args, tokenizer)
    print(len(dataset))


if __name__ == "__main__":
    main()