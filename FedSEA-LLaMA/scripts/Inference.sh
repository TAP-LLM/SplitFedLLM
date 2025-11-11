#!/bin/bash

# 获取conda的基础路径
CONDA_BASE=$(conda info --base)
__conda_setup="$('$CONDA_BASE/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# 如果conda初始化成功
if [ $? -eq 0 ]; then
    # 执行conda的初始化脚本
    eval "$__conda_setup"
else
    # 如果conda初始化失败，尝试直接加载conda.sh
    if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        . "$CONDA_BASE/etc/profile.d/conda.sh"
    else
        # 如果conda.sh不存在，手动添加conda的bin目录到PATH环境变量
        export PATH="$CONDA_BASE/bin:$PATH"
    fi
fi

# 清除conda初始化变量
unset __conda_setup

# 离开当前环境
conda deactivate

# 激活名为 ppo_drone 的 conda 环境
conda activate 97rerank

# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
#只在特定GPU上运行
Device="cuda:1"
CheckpointStep=10
Lora_modelB_path="output/Record/version_0_r16_al32_Lr0.00003/model-B/checkpoint-${CheckpointStep}"
Lora_modelA_path="output/Record/version_0_r16_al32_Lr0.00003/model-A/checkpoint-${CheckpointStep}"
Lora_modelC_path="output/Record/version_0_r16_al32_Lr0.00003/model-C/checkpoint-${CheckpointStep}"
Max_source_length=700
ServerIP="10.143.12.73:7080"
ModelA_layers=2
ModelC_layers=3

# 定义日志文件路径
LOG_FILE_1="$Lora_modelB_path/output_server.log"
Test_file="test.jsonl"
Max_output_length=200
# 检查日志文件是否存在，如果存在则删除
if [ -f "$LOG_FILE_1" ]; then
    rm "$LOG_FILE_1"
fi

LOG_FILE_2="$Lora_modelA_path/output_client.log"

# 检查日志文件是否存在，如果存在则删除
if [ -f "$LOG_FILE_2" ]; then
    rm "$LOG_FILE_2"
fi

# 使用 nohup 在后台运行 Python 脚本
nohup python src/flserver.py \
    --do_inference \
    --device $Device \
    --server_ip $ServerIP \
    --modelA_layers $ModelA_layers \
    --modelC_layers $ModelC_layers \
    --lora_modelB_path $Lora_modelB_path > $LOG_FILE_1 2>&1 &

    # --do_inference > output_1.log 2>&1 &
sleep 120

nohup python src/flclient_id_0.py \
    --do_inference \
    --test_file $Test_file \
    --device $Device \
    --server_ip $ServerIP \
    --modelA_layers $ModelA_layers \
    --modelC_layers $ModelC_layers \
    --max_source_length $Max_source_length \
    --lora_modelA_path $Lora_modelA_path \
    --max_output_length $Max_output_length \
    --lora_modelC_path $Lora_modelC_path  > $LOG_FILE_2 2>&1 &

    # --do_inference > output_2.log 2>&1 &

# 复制运行脚本到输出文件夹
SCRIPT_NAME=$(basename "$0")
cp "$0" "$Lora_modelA_path/${SCRIPT_NAME%.*}_${TIMESTAMP}_${CheckpointStep}_Inference.sh"