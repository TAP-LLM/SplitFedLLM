#!/bin/bash

#备注
#不同的卡需要更换device、hyparams的路径

# ----------------- 基础配置 -----------------
# 获取conda的基础路径
CONDA_BASE=$(conda info --base)
#获取脚本绝对路径
Script_name=$(basename "$0")
Script_abs_path=$(cd "$(dirname "$0")"; pwd -P)/$(basename "$0")
    
# ----------------- Conda初始化 -----------------
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

# ----------------- 环境配置 -----------------
# 离开当前环境
conda deactivate
# 激活名为 ppo_drone 的 conda 环境
conda activate 97rerank
# 保证操作可复现
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ----------------- 超参数循环 -----------------
# 连接网络
nohup python /home/zhangzishuai/web_login/buaa_login.py &
# 启动邮件服务
nohup python /home/zhangzishuai/Tools/mail_reminder.py \
    --mail_start \
    --script_name $Script_name \
    --script_abs_path $Script_abs_path &

HYPERPARAMS_FILE="/home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module/scripts_for_Record/hyperparams_2.txt"
# 逐行读取超参数文件
while IFS= read -r line || [[ -n "$line" ]]; do
    # 跳过注释行和空行
    [[ "$line" =~ ^# ]] || [[ -z "$line" ]] && continue
    
    # 解析超参数
    LearningRate=$(echo $line | awk '{print $1}')
    Lora_r=$(echo $line | awk '{print $2}')
    Lora_alpha=$(echo $line | awk '{print $3}')
    Version=$(echo $line | awk '{print $4}')
    
    echo "正在训练:Lr=$LearningRate | Lora_r=$Lora_r | Lora_alpha=$Lora_alpha | version=$Version"

    # ----------------- 动态参数配置 -----------------
    # 获取当前时间戳
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    Weight_decay=0.01
    Seed=42
    OUTPUT_DIR="/home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module/output/Record/version_${Version}_r${Lora_r}_al${Lora_alpha}_Lr${LearningRate}"
    Device='cuda:1'
    Torch_dtype="float16"
    Max_source_length=700
    Max_target_length=20
    Batch_size=2
    Passage_column='passage'
    Question_column='qas'
    Train_file='sample_train.jsonl'
    MAX_STEP=50000
    SAVE_STEP=1000
    Dataset_name='Record'
    ServerIP="10.143.12.74:8080"
    # GPU 0占用4050 6080  GPU 1占用8080 5040

    mkdir -p "$OUTPUT_DIR"
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

    # 复制运行脚本到输出文件夹
    cp "$0" "$OUTPUT_DIR/${Script_name%.*}_$TIMESTAMP.sh"

    # ----------------- 执行训练 -----------------
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
        --max_source_length $Max_source_length \
        --max_target_length $Max_target_length \
        --save_step $SAVE_STEP \
        --max_step $MAX_STEP \
        --output_dir "$OUTPUT_DIR" > "$LOG_FILE_1" 2>&1 &

    sleep 125

    python /home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module/flclient_llama7.py \
        --do_train \
        --from_pickle \
        --device $Device \
        --server_ip $ServerIP \
        --lora_r $Lora_r \
        --lora_alpha $Lora_alpha \
        --seed $Seed \
        --dataset_name $Dataset_name \
        --batch_size $Batch_size \
        --lr $LearningRate \
        --torch_dtype $Torch_dtype \
        --weight_decay $Weight_decay \
        --max_source_length $Max_source_length \
        --max_target_length $Max_target_length \
        --save_step $SAVE_STEP \
        --max_step $MAX_STEP \
        --train_file $Train_file \
        --passage_column $Passage_column \
        --question_column $Question_column \
        --output_dir "$OUTPUT_DIR" > "$LOG_FILE_2" 2>&1 
    wait

    echo "完成训练:Lr=$LearningRate | Lora_r=$Lora_r | Lora_alpha=$Lora_alpha | version=$Version"
    echo "----------------------------------------"

done < "$HYPERPARAMS_FILE"

# ----------------- 训练结束通知 -----------------
nohup python /home/zhangzishuai/web_login/buaa_login.py &
nohup python /home/zhangzishuai/Tools/mail_reminder.py \
    --mail_finished \
    --script_name $Script_name \
    --script_abs_path $Script_abs_path &

