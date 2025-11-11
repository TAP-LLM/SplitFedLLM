import torch
import os
import sys
import logging
from peft import ( 
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
import argparse
import numpy as np
import re
from modeling_llama_official import LlamaForCausalLM
'''自由分割模型'''

log_path = "Output/lora_splitmodel.log"
logging.basicConfig(filename=log_path,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                    level=logging.INFO,
                    filemode='w')
# create a logger object
logger = logging.getLogger(__name__)

def FLparser(): # use glm arguments with argument.py
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--target_modules", type=str, default='embed_tokens,q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj',
                        help="List of module names or regex expression of the module names to replace with Lora. For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--SplitA", action="store_true", help="Enable SplitA")
    parser.add_argument("--SplitB", action="store_true", help="Enable SplitB")
    parser.add_argument("--SplitC", action="store_true", help="Enable SplitC")
    # LoraA/B/C 参数（布尔值）
    parser.add_argument("--LoraA", action="store_true", help="Enable LoraA")
    parser.add_argument("--LoraB", action="store_true", help="Enable LoraB")
    parser.add_argument("--LoraC", action="store_true", help="Enable LoraC")
    args = parser.parse_args()

    return args


# 加载预训练的中心化Llama模型
model = LlamaForCausalLM.from_pretrained('Centralized_Models/model_path')
FL_args = FLparser()

# 分割模型A、B、C，必选
SplitA, SplitB, SplitC = FL_args.SplitA, FL_args.SplitB, FL_args.SplitC
# 分割Lora模型A、B、C，保证每一次从相同的lora权重初始化,可选
LoraA, LoraB, LoraC = FL_args.LoraA, FL_args.LoraB, FL_args.LoraC

'''===========================================================拆分Lora权重==========================================================='''
if LoraA and LoraB and LoraC:
    print("正在分割lora权重")
    if type(FL_args.target_modules)==str:
        FL_args.target_modules = FL_args.target_modules.split(',')
    client_layer_id = np.arange(0,32)

    t_m_c = []
    self_attn_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    other_layer_modules = ['gate_proj', 'up_proj', 'down_proj']
    global_modules = ['embed_tokens']  # 不属于某一层（如模型顶层）
    for index in FL_args.target_modules:
        logger.info(f"index: {index}")
        if index in global_modules:
            # 顶层模块，如 embed_tokens
            t_m_c.append(f'model.{index}')
        elif index in self_attn_modules:
            # 属于 self_attn 的模块
            for i in client_layer_id:
                t_m_c.append(f'model.layers.{i}.self_attn.{index}')
        elif index in other_layer_modules:
            # 属于每层的其他模块，比如 feed_forward
            for i in client_layer_id:
                t_m_c.append(f'model.layers.{i}.mlp.{index}')
        else:
            logger.warning(f"未知模块类型: {index}，未添加到 target_modules")

    FL_args.target_modules = t_m_c
    logger.info(f"FL_args.target_modules:{FL_args.target_modules}")

    lora_config = LoraConfig(
        r=FL_args.lora_r,
        lora_alpha=FL_args.lora_alpha,
        target_modules =  FL_args.target_modules,
        fan_in_fan_out = False,
        lora_dropout=0.05,
        inference_mode=False,
        bias="none",
        task_type="CAUSAL_LM",
    )
    logger.info(f"lora_config:{lora_config}")

    model = get_peft_model(model, lora_config)
    # 打印模型的参数
    # for name, module in model.named_modules():
    #     logger.info(f"Module: {name}, Type: {type(module)}")
    # for name, param in model.named_parameters():
    #     logger.info(f"Parameter Name: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")   

    # 保存Lora权重
    # 构造路径
    output_dir = f"Lora_weights/lora_r{FL_args.lora_r}_alpha{FL_args.lora_alpha}_weights/"

    # 创建文件夹（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    model.save_pretrained(f"Lora_weights/lora_r{FL_args.lora_r}_alpha{FL_args.lora_alpha}_weights/", save_only_lora=True,safe_serialization=False,map_location="cuda:1")

    lora_weights_full = torch.load(f'Lora_weights/lora_r{FL_args.lora_r}_alpha{FL_args.lora_alpha}_weights/adapter_model.bin', map_location="cuda:1")
    for key in lora_weights_full.keys():
        logger.info(f"key:{key}")

    def adjust_lora_layer_number(original_key, offset=-1):
        """
        调整层号的通用函数
        :param original_key: 原始参数键名，例如"base_model.model.model.layers.1.self_attn.q_proj.lora_A.weight"
        :param offset: 需要调整的偏移量（默认为-1)
        :return: 调整后的键名
        """
        def replace_layer(match):
            original_num = int(match.group(1))
            new_num = original_num + offset
            return f".layers.{new_num}"
        
        # 使用正则表达式匹配 .layers.数字 的模式
        adjusted_key = re.sub(
            r'\.layers\.(\d+)',  # 精确匹配.layers.数字模式
            replace_layer,
            original_key
        )
        return adjusted_key


    def split_lora_weights(lora_dir, split_config):
        """
        :param lora_dir: Lora权重目录(包含adapter_model.bin)
        :param split_config: 拆分配置，如：
            {
                'A': {'layers': [0], 'save_path': 'lora_A'},
                'B': {'layers': [1,2,3], 'save_path': 'lora_B'},
                'C': {'layers': [4], 'save_path': 'lora_C'}
            }
        """
        full_weights = torch.load(os.path.join(lora_dir, "adapter_model.bin"))
        
        for part, config in split_config.items():
            if part == "A" and LoraA:
                part_weights = {}
                layer_pattern = re.compile(r'base_model\.model\.model\.layers\.(\d+)')
                
                for key in full_weights:
                    match = layer_pattern.search(key)
                    if match:
                        layer_num = int(match.group(1))
                        if layer_num in config['layers']:
                            part_weights[key] = full_weights[key]
                    else:  # 处理非层级参数（如lora_embedding）
                        part_weights[key] = full_weights[key]
                
                # 保存拆分后的权重
                os.makedirs(config['save_path'], exist_ok=True)
                torch.save(part_weights, os.path.join(config['save_path'], "adapter_model.bin"))
                # 复制配置文件
                os.system(f"cp {os.path.join(lora_dir,'adapter_config.json')} {config['save_path']}")

            if part == "B" and LoraB:
                part_weights = {}
                layer_pattern = re.compile(r'base_model\.model\.model\.layers\.(\d+)')
                
                for key in full_weights:
                    match = layer_pattern.search(key)
                    if match:
                        layer_num = int(match.group(1))
                        if layer_num in config['layers']:
                            new_key= adjust_lora_layer_number(key)
                            part_weights[new_key] = full_weights[key]
                    else:  # 处理非层级参数（如lora_embedding）
                        # part_weights[key] = full_weights[key]
                        pass
                # 保存拆分后的权重
                os.makedirs(config['save_path'], exist_ok=True)
                torch.save(part_weights, os.path.join(config['save_path'], "adapter_model.bin"))
                # 复制配置文件
                os.system(f"cp {os.path.join(lora_dir,'adapter_config.json')} {config['save_path']}")

            if part == "C" and LoraC:
                part_weights = {}
                layer_pattern = re.compile(r'base_model\.model\.model\.layers\.(\d+)')
                
                for key in full_weights:
                    match = layer_pattern.search(key)
                    if match:
                        layer_num = int(match.group(1))
                        if layer_num in config['layers']:
                            new_key= adjust_lora_layer_number(key,offset=-27)
                            part_weights[new_key] = full_weights[key]
                    else:  # 处理非层级参数（如lora_embedding）
                        #part_weights[key] = full_weights[key]
                        pass                
                # 保存拆分后的权重
                os.makedirs(config['save_path'], exist_ok=True)
                torch.save(part_weights, os.path.join(config['save_path'], "adapter_model.bin"))
                # 复制配置文件
                os.system(f"cp {os.path.join(lora_dir,'adapter_config.json')} {config['save_path']}")
    

    # 分割Lora权重（假设总共有5层）
    split_config = {
        'A': {'layers': list(range(0, 5)), 'save_path':f'Lora_weights/lora_r{FL_args.lora_r}_alpha{FL_args.lora_alpha}_weights/lora_A'},
        'B': {'layers': list(range(1, 31)), 'save_path':f'Lora_weights/lora_r{FL_args.lora_r}_alpha{FL_args.lora_alpha}_weights/lora_B'},
        'C': {'layers': list(range(27, 32)), 'save_path':f'Lora_weights/lora_r{FL_args.lora_r}_alpha{FL_args.lora_alpha}_weights/lora_C'}
    }
    split_lora_weights(f"Lora_weights/lora_r{FL_args.lora_r}_alpha{FL_args.lora_alpha}_weights", split_config)
    print("lora weights saved")


'''===========================================================拆分模型权重==========================================================='''
def rename_keys_A(current_dict):
    new_dict = {}
    for key in current_dict.keys():
        # 将键分割为列表，按照点分隔符
        key_parts = key.split('.')
        
        # 将 "1" 变为 "model.layers.0" 的格式，"1.0" 变为 "model.layers.0"
        if key_parts[0] == '0':
            # 第0层对应model.embed_tokens
            new_key = 'model.embed_tokens.weight'
        elif key_parts[0] == '1':
            # 第1层起变为model.layers.N
            new_key = 'model.layers'
        else:
            print(key)
            print("有问题")
        
        # 组合剩余部分
        if len(key_parts) > 2:
            new_key += '.' + '.'.join(key_parts[1:])

        # 将旧的值赋给新的键
        new_dict[new_key] = current_dict[key]
    
    return new_dict

def rename_keys_B(current_dict):
    new_dict = {}
    for key in current_dict.keys():
        # 将键分割为列表，按照点分隔符
        key_parts = key.split('.')
        

        new_key = 'model.layers' + '.' + '.'.join(key_parts)

        # 将旧的值赋给新的键
        new_dict[new_key] = current_dict[key]
    
    return new_dict

def rename_keys_C(current_dict):
    new_dict = {}
    for key in current_dict.keys():
        # 将键分割为列表，按照点分隔符
        key_parts = key.split('.')
        
        # 将 "1" 变为 "model.layers.0" 的格式，"1.0" 变为 "model.layers.0"
        if key_parts[0] == '0':
            # 第0层对应model.embed_tokens
            new_key = 'model.layers' + '.' + '.'.join(key_parts[1:])
        elif key_parts[0] == '1':
            # 第1层起变为model.layers.N
            new_key = 'model.norm.weight'
        elif key_parts[0] == '2':
            new_key = 'lm_head.weight'
        else:
            print(key)
            print("有问题")

        

        # 将旧的值赋给新的键
        new_dict[new_key] = current_dict[key]
    
    return new_dict

if SplitA and SplitB and SplitC:
    print("正在分割模型权重")
    # 拆分模型为modelA
    modelA = torch.nn.Sequential(model.model.embed_tokens, model.model.layers[:5])  # 第0，1，2，3，4层都保存，加载的时候不一定全部加载
    # 保存模型到不同的文件
    torch.save(modelA.state_dict(), 'clientA/temp_pytorch_model_A.bin')

    # logger.info(f"下面是modelA的参数")
    # logger.info("\n")  
    # for name, module in modelA.named_modules():
    #     logger.info(f"Module: {name}, Type: {type(module)}")
    # for name, param in modelA.named_parameters():
    #     logger.info(f"Parameter Name: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}") 

    # 拆分模型为modelB
    modelB = model.model.layers[1:31]  # 第1到30层,加载的时候不一定全部加载
    # 保存模型到不同的文件
    torch.save(modelB.state_dict(), 'server/temp_pytorch_model_B.bin')

    # 拆分模型为modelC
    modelC = torch.nn.Sequential(model.model.layers[27:], model.model.norm, model.lm_head)  # 第27,28,29,30,31层都保存，加载的时候不一定全部加载
    # 保存模型到不同的文件
    torch.save(modelC.state_dict(), 'clientC/temp_pytorch_model_C.bin')

    #由于Sequential修改模型的参数名称，所以需要重新命名
    state_dict_A = torch.load('clientA/temp_pytorch_model_A.bin')
    state_dict_A = rename_keys_A(state_dict_A)
    torch.save(state_dict_A, 'clientA/pytorch_model_A.bin')

    state_dict_B = torch.load('server/temp_pytorch_model_B.bin')
    state_dict_B = rename_keys_B(state_dict_B)
    torch.save(state_dict_B, 'server/pytorch_model_B.bin')

    state_dict_C = torch.load('clientC/temp_pytorch_model_C.bin')
    state_dict_C = rename_keys_C(state_dict_C)
    torch.save(state_dict_C, 'clientC/pytorch_model_C.bin')

    print("FedSEA-LLaMa Model weights saved, you can manually delete the temp weights.")


