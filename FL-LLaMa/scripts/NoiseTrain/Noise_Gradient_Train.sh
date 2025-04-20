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

#
#for privacy_budget in 116000 58000 145000 193333 290000 580000 1160000 400000; do
for Grad_noise_option in 1e-7 2e-7 3e-7 5e-7 8e-7 1e-6 15e-7 0; do
    echo "Grad_noise: $Grad_noise_option"
    # 获取当前时间戳
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    Data_fold="/home/zhangzishuai/SplitFederated-LLaMA/Dataset/MultiRC"
    Max_source_length=910
    Max_target_length=10
    Train_file="train.jsonl"
    Passage_column="passage"
    # Answer_column="label"
    SAVE_STEP=1000
    MAX_STEP=8000
    BatchSize=2
    Seed=42
    Lora_r=8
    Lora_alpha=16
    LearningRate=0.00001   #默认1e-5 , llama官方1e-4,太大，这里改为 1e-5 
    Weight_decay=0.01   #默认1e-2
    Torch_dtype="float16" #训练的时候必须要用float32不然学习率太大了
    #只在特定GPU上运行 ,单个服务器上运行多个程序需要更改GPU
    Device="cuda:0"
    #单个服务器上运行多个程序需要更改IP
    ServerIP="10.143.12.74:8080" 
    #400000
    Grad_noise=$Grad_noise_option
    DP_grad_norm_clip=0.04
    # GPU 4上已经占用8080 6080 4070 5040
    OUTPUT_DIR="/home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module/output/DP_debug/version_noise${Grad_noise_option}"
    model_A_PATH="${OUTPUT_DIR}/model-A"
    model_B_PATH="${OUTPUT_DIR}/model-B"
    model_C_PATH="${OUTPUT_DIR}/model-C"
    # 检查并创建目录
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$model_A_PATH"
    mkdir -p "$model_B_PATH"
    mkdir -p "$model_C_PATH"

    # 定义日志文件路径
    LOG_FILE_1="$OUTPUT_DIR/output_server.log"
    # 检查日志文件是否存在，如果存在则删除
    if [ -f "$LOG_FILE_1" ]; then
        rm "$LOG_FILE_1"
    fi
    LOG_FILE_2="$OUTPUT_DIR/output_client.log"
    # 检查日志文件是否存在，如果存在则删除
    if [ -f "$LOG_FILE_2" ]; then
        rm "$LOG_FILE_2"
    fi

    # 复制运行脚本到输出文件夹
    SCRIPT_NAME=$(basename "$0")
    cp "$0" "$OUTPUT_DIR/${SCRIPT_NAME%.*}_r${Lora_r}_alpha${Lora_alpha}_$TIMESTAMP.sh"

    # 连接网络
    nohup python /home/zhangzishuai/web_login/buaa_login.py &
    # 启动邮件服务
    Script_name=$(basename "$0")
    Script_abs_path=$(cd "$(dirname "$0")"; pwd -P)/$(basename "$0")
    nohup python /home/zhangzishuai/Tools/mail_reminder.py \
        --mail_start \
        --script_name $Script_name \
        --script_abs_path $Script_abs_path &

    # 使用 nohup 在后台运行 Python 脚本
    nohup python /home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module/flserver_llama8_DP.py \
        --do_train \
        --add_DP_gradient \
        --device $Device \
        --server_ip $ServerIP \
        --lora_r $Lora_r \
        --lora_alpha $Lora_alpha \
        --seed $Seed \
        --batch_size $BatchSize \
        --grad_noise $Grad_noise \
        --DP_grad_norm_clip $DP_grad_norm_clip \
        --lr $LearningRate \
        --torch_dtype $Torch_dtype \
        --weight_decay $Weight_decay \
        --save_step $SAVE_STEP \
        --max_step $MAX_STEP \
        --output_dir "$OUTPUT_DIR" > "$LOG_FILE_1" 2>&1 &

    sleep 110

    python /home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module/flclient_llama8_DP.py \
        --do_train \
        --add_DP_gradient \
        --device $Device \
        --server_ip $ServerIP \
        --lora_r $Lora_r \
        --lora_alpha $Lora_alpha \
        --seed $Seed \
        --batch_size $BatchSize \
        --grad_noise $Grad_noise \
        --DP_grad_norm_clip $DP_grad_norm_clip \
        --lr $LearningRate \
        --torch_dtype $Torch_dtype \
        --weight_decay $Weight_decay \
        --max_source_length $Max_source_length \
        --max_target_length $Max_target_length \
        --save_step $SAVE_STEP \
        --max_step $MAX_STEP \
        --train_file $Train_file \
        --passage_column $Passage_column \
        --output_dir "$OUTPUT_DIR" > "$LOG_FILE_2" 2>&1 &
    wait

done

# 连接网络
nohup python /home/zhangzishuai/web_login/buaa_login.py &
# 启动邮件服务
Script_name=$(basename "$0")
Script_abs_path=$(cd "$(dirname "$0")"; pwd -P)/$(basename "$0")
nohup python /home/zhangzishuai/Tools/mail_reminder.py \
    --mail_finished \
    --script_name $Script_name \
    --script_abs_path $Script_abs_path &
