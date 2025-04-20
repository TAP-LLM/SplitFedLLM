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

# 保证操作可复现
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
Max_source_length=1350
Max_target_length=80
Train_file="sample_train.json"
Passage_column="input_data"
Answer_column="label"
SAVE_STEP=1000
MAX_STEP=50000
Seed=42
Lora_r=32
Lora_alpha=32
LearningRate=0.00003  #默认1e-5 , llama官方1e-4,太大，这里改为 1e-5 
Weight_decay=0.01  #默认1e-2
Torch_dtype="float16" #训练的时候必须要用float32不然学习率太大了
#只在特定GPU上运行 ,单个服务器上运行多个程序需要更改GPU
Device="cuda:1"
#单个服务器上运行多个程序需要更改IP
ServerIP="10.143.12.74:8080" 
Batch_size=2
Dataset_name="Xsum"
# GPU 0占用4050 6080  GPU 1占用8080 5040
OUTPUT_DIR="/home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module/output/Xsum/version_12_r${Lora_r}_alpha${Lora_alpha}_lr${LearningRate}"
model_A_PATH="${OUTPUT_DIR}/model-A"
model_B_PATH="${OUTPUT_DIR}/model-B"
model_C_PATH="${OUTPUT_DIR}/model-C"
# 检查并创建目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$model_A_PATH"
mkdir -p "$model_B_PATH"
mkdir -p "$model_C_PATH"

# 定义日志文件路径
LOG_FILE_1="$OUTPUT_DIR/trian_server_output.log"
# 检查日志文件是否存在，如果存在则删除
if [ -f "$LOG_FILE_1" ]; then
    rm "$LOG_FILE_1"
fi
LOG_FILE_2="$OUTPUT_DIR/trian_client_output.log"
# 检查日志文件是否存在，如果存在则删除
if [ -f "$LOG_FILE_2" ]; then
    rm "$LOG_FILE_2"
fi

# 连接网络
nohup python /home/zhangzishuai/web_login/buaa_login.py &
# 启动邮件服务
Script_name=$(basename "$0")
Script_abs_path=$(cd "$(dirname "$0")"; pwd -P)/$(basename "$0")
nohup python /home/zhangzishuai/Tools/mail_reminder.py \
    --mail_start \
    --script_name $Script_name \
    --script_abs_path $Script_abs_path &

# 复制运行脚本到输出文件夹
SCRIPT_NAME=$(basename "$0")
cp "$0" "$OUTPUT_DIR/${SCRIPT_NAME%.*}_r${Lora_r}_alpha${Lora_alpha}_$TIMESTAMP.sh"

# 使用 nohup 在后台运行 Python 脚本
nohup python /home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module/flserver_llama7.py \
    --do_train \
    --device $Device \
    --server_ip $ServerIP \
    --lora_r $Lora_r \
    --lora_alpha $Lora_alpha \
    --seed $Seed \
    --batch_size $Batch_size \
    --lr $LearningRate \
    --torch_dtype $Torch_dtype \
    --weight_decay $Weight_decay \
    --save_step $SAVE_STEP \
    --max_step $MAX_STEP \
    --output_dir "$OUTPUT_DIR" > "$LOG_FILE_1" 2>&1 &

sleep 125

python /home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module/flclient_llama7.py \
    --do_train \
    --device $Device \
    --server_ip $ServerIP \
    --lora_r $Lora_r \
    --lora_alpha $Lora_alpha \
    --seed $Seed \
    --lr $LearningRate \
    --batch_size $Batch_size \
    --torch_dtype $Torch_dtype \
    --weight_decay $Weight_decay \
    --dataset_name $Dataset_name \
    --max_source_length $Max_source_length \
    --max_target_length $Max_target_length \
    --save_step $SAVE_STEP \
    --max_step $MAX_STEP \
    --train_file $Train_file \
    --passage_column $Passage_column \
    --answer_column $Answer_column \
    --output_dir "$OUTPUT_DIR" > "$LOG_FILE_2" 2>&1 
wait

# 连接网络
nohup python /home/zhangzishuai/web_login/buaa_login.py &
# 启动邮件服务
Script_name=$(basename "$0")
Script_abs_path=$(cd "$(dirname "$0")"; pwd -P)/$(basename "$0")
nohup python /home/zhangzishuai/Tools/mail_reminder.py \
    --mail_finished \
    --script_name $Script_name \
    --script_abs_path $Script_abs_path &
