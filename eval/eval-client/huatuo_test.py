from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel,PeftConfig,AutoPeftModelForCausalLM
import transformers
import torch
import csv
import os
import pandas as pd
from modeling_llama_client import LlamaForCausalLM
import socket
import requests
import pickle
from tqdm import tqdm
from datasets import Dataset
import json
import time

finetune_model_path = "/home/wanglingxiang/llama2_lora/sy2342131/models/llama2_huatuo"
base_model = "/home/wanglingxiang/llama2_lora/model/Llama-2-7b-hf" #拆分后的模型文件，可以调用“/home/wanglingxiang/llama2/models/Llama-2-7b-hf”测试时效果一致。
model = LlamaForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
    )
# tokenizer = AutoTokenizer.from_pretrained(base_model)
# model.save_pretrained("/home/wanglingxiang/llama2_lora/lora/test-client/model_client_llama")
# tokenizer.save_pretrained("/home/wanglingxiang/llama2_lora/lora/test-client/model_client_llama")
model = PeftModel.from_pretrained(model, finetune_model_path, device_map={"": 0})

# model = LlamaForCausalLM.from_pretrained(finetune_model_path,device_map='auto')
# model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(base_model)

# prompt = "Instruction: Please introduce yourself.\nInput: hi\nOutput:"
# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
# print(inputs)

# shape_size = len(inputs.input_ids[0])
# array_bytes = pickle.dumps(shape_size)

# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# receiver_ip = '10.143.12.71'
# receiver_port = 7170
# s.connect((receiver_ip, receiver_port))
# s.sendall(array_bytes)

# print("顺利传输数据集构建数据到服务器端")
# stop = input()
# shape = (1,len(inputs.input_ids[0]))
# Generate
# generate_ids = model.generate(input_ids=inputs.input_ids,do_sample=False,max_new_tokens=4)
# result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
# print(result)


json_file_path = '/home/wanglingxiang/llama2_lora/sy2342131/datasets/split_file1.json'

class MyDataset(torch.utils.data.Dataset):
    def __init__(self,json_file_path):
        self.dataset = Dataset.from_json(json_file_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        question = self.dataset[i]['question']
        answer = self.dataset[i]['answer']
        return question,answer

dataset = MyDataset(json_file_path)

# 准备输入文本
def input_data(question):
    input_text = '<s>[INST]'+question+'[/INST]'
    input_ids = tokenizer(input_text, return_tensors="pt",add_special_tokens=False).to(model.device) 
    return input_ids
def generate_output(input_ids):
    generate_input = {
        "input_ids":input_ids,
        "max_new_tokens":16,
        "do_sample":True,
        "top_k":50,
        "top_p":0.95,
        "temperature":0.3,
        "repetition_penalty":1.3,
        "eos_token_id":tokenizer.eos_token_id,
        "bos_token_id":tokenizer.bos_token_id,
        "pad_token_id":tokenizer.pad_token_id
        }
    return generate_input


output_file = []
failed_items = []
for i in tqdm(range(len(dataset)), desc="Processing data"):
        
        question,answer = dataset[i]
        input_contents = input_data(question)

        shape_size = len(input_contents.input_ids[0])
        array_bytes = pickle.dumps(shape_size)
        receiver_ip = '10.143.12.71'
        error_receiver_port = 7171
        error_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        error_socket.settimeout(5)  # 设置超时时间为5秒
        try:
            error_socket.connect((receiver_ip, error_receiver_port))
            data = error_socket.recv(1024)
            if data == b'error':
                print("服务器在处理输入时出现问题，重新发送输入")
                failed_item = {}
                failed_item["label"] = answer
                failed_item["generated_answer"] = generated_answer
                failed_items.append(failed_item)  # 将失败的项添加到列表中
                continue
        except socket.timeout:
            print("等待服务器响应超时")
        finally:
            error_socket.close()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        receiver_port = 7170
        
        s.connect((receiver_ip, receiver_port))
        s.sendall(array_bytes)
        print("顺利传输数据集构建数据到服务器端")
        s.close()

        time.sleep(1)  # 等待服务器处理数据


        print(shape_size)
        # stop = input()

        # generate_input = generate_output(input_contents.input_ids)
        # generate_ids  = model.generate(**generate_input)
        generate_ids = model.generate(input_ids=input_contents.input_ids,do_sample=False,max_new_tokens=256)
        text = tokenizer.decode(generate_ids[0])
        start_index = text.find('[/INST]') + len('[/INST]')
        generated_answer = text[start_index:]
        generated_answer = generated_answer.replace('</s>', '')
        print(generated_answer)
        output_results = {}
        output_results["label"] = answer
        output_results["generated_answer"] = generated_answer
        output_file.append(output_results)
        # if i == 2 : break
        print(f"已经完成了第{i+1}个推理") 
        

        
        

with open('/home/wanglingxiang/llama2_lora/sy2342131/datasets/zh_res_ft.json','a',encoding='utf-8') as json_file:
    json.dump(output_file, json_file, indent=4, ensure_ascii=False)
    print('done') 

with open('/home/wanglingxiang/llama2_lora/sy2342131/datasets/zh_error_res_ft.json','a',encoding='utf-8') as json_file:
    json.dump(failed_items, json_file, indent=4, ensure_ascii=False)
    print('done') 
