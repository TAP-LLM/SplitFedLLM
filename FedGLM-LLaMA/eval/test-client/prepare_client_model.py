from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel,PeftConfig,AutoPeftModelForCausalLM
import transformers
import torch
import csv
import os
import pandas as pd
from modeling_llama_client import LlamaForCausalLM,LlamaConfig
import socket
import requests
import pickle
from tqdm import tqdm
from datasets import Dataset
import json
import time
import gc

finetune_model_path = "your_finetune_model_path"
base_model = "your_base_model_path" 
model = LlamaForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map="auto",
    )
model = PeftModel.from_pretrained(model, finetune_model_path, device_map={"": 0})
tokenizer = AutoTokenizer.from_pretrained(base_model)
config = LlamaConfig.from_pretrained(base_model)

model.save_pretrained("your_model_path")
tokenizer.save_pretrained("your_model_path")
config.save_pretrained("your_model_path")