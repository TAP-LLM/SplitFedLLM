from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel,PeftConfig,AutoPeftModelForCausalLM
import transformers
import torch
import csv
import os
import pandas as pd
from modeling_llama_service import LlamaForCausalLM
import socket
import io
from flask import Flask, request
import pickle
import time
from threading import Thread
from queue import Queue
import requests_async as requests

receiver_ip = 'your_ip'
receiver_port = 7170

# 创建队列
data_queue = Queue()

def start_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((receiver_ip, receiver_port))
    s.listen(1)

    while True:
        conn, addr = s.accept()
        data = conn.recv(4096)
        received_array = pickle.loads(data)
        data_queue.put(received_array)
        conn.close()
    

t = Thread(target=start_server)
t.start()
error_sender_port = 7171
# 创建一个错误发送的socket
error_sender_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 绑定IP和端口
error_sender_socket.bind((receiver_ip, error_sender_port))
# 开始监听
error_sender_socket.listen(1)


print("服务器端已准备就绪")
time.sleep(3)

finetune_model_path = "your_checkpoint_path"
base_model = "your_base_model_path"
model = LlamaForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
    )
model = PeftModel.from_pretrained(model, finetune_model_path, device_map={"": 0})

tokenizer = AutoTokenizer.from_pretrained(base_model)

while(True):
    try:
        inputs_len = None
        while(inputs_len == None):
            # print("等待接收")
            if not data_queue.empty():
                    inputs_len = data_queue.get()
                    print("接收成功")
                    print(inputs_len)
                    # stop = input()
                   
            else:
                continue

        shape = (1,inputs_len)
        test_input_ids = torch.randint(1,32000,shape)
        print(test_input_ids)
        # stop = input()
        # Generate
        generate_ids = model.generate(input_ids=test_input_ids,do_sample=False,max_new_tokens=256)
        result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(result)

    except Exception as e:  # 捕获所有的异常
        print(f"在处理输入时出现问题: {e}")
        print(f"异常类型: {type(e)}")
        print(f"异常详细信息: {e.args}")
        conn, addr = error_sender_socket.accept()
        # 发送错误信息
        conn.sendall(b'error')
        print("已向客户端发送错误信息")
        # 关闭连接
        conn.close()
        error_sender_socket.close()  # 确保关闭socket
        error_sender_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 重新创建socket
        error_sender_socket.bind((receiver_ip, error_sender_port))  # 重新绑定
        error_sender_socket.listen(1)  # 重新监听
        continue
        