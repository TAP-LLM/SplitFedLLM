#!/bin/bash
set -u

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
# 激活conda 环境
conda activate 97rerank
# 保证操作可复现
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ----------------- 超参数循环 -----------------
# cd到工作目录
cd /home/zhangzishuai/SplitFederated-LLaMA/FL-LLaMA

HYPERPARAMS_FILE="./scripts/hyperparams.txt"
# 逐行读取超参数文件
while IFS= read -r line || [[ -n "$line" ]]; do
    # 跳过注释行和空行
    [[ "$line" =~ ^# ]] || [[ -z "$line" ]] && continue
    
    # 解析超参数
    LearningRate=$(echo $line | awk '{print $1}')
    Lora_r=$(echo $line | awk '{print $2}')
    Lora_alpha=$(echo $line | awk '{print $3}')
    Version=$(echo $line | awk '{print $4}')
    # 动态选择分割点
    ModelA_layers=$(echo $line | awk '{print $5}')
    ModelC_layers=$(echo $line | awk '{print $6}')

    echo "正在训练:Lr=$LearningRate | Lora_r=$Lora_r | Lora_alpha=$Lora_alpha | version=$Version | ModelA_layers=$ModelA_layers | ModelC_layers=$ModelC_layers"
    # ----------------- 动态参数配置 -----------------
    # 获取当前时间戳
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    Weight_decay=0.01
    Seed=42
    OUTPUT_DIR="output/Record/version_${Version}_r${Lora_r}_al${Lora_alpha}_Lr${LearningRate}"
    Device='cuda:1'
    Torch_dtype="float16"
    Max_source_length=910
    Max_target_length=10
    Batch_size=2

    # from_pickle为False时需要填写参数，为True时从pickle加载数据集
    # data2pic为Ture时,数据集保存为pickle
    Passage_column='passage'
    Question_column='qas'
    Train_file='sample_train.jsonl'

    # MAX_STEP=10000
    # SAVE_STEP=1000
    MAX_STEP=10000
    SAVE_STEP=10
    Dataset_name='Record'
    ServerIP="10.143.12.73:7080"

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
    nohup python src/flserver.py \
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
        --modelA_layers $ModelA_layers \
        --modelC_layers $ModelC_layers \
        --output_dir "$OUTPUT_DIR" > "$LOG_FILE_1" 2>&1 &
    
    # 等待服务器加载完成
    sleep 110
    # 第一次运行flclient_id_0.py请注释from_pickle
    python src/flclient_id_0.py \
        --do_train \
        --from_pickle \
        --data2pic \
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
        --modelA_layers $ModelA_layers \
        --modelC_layers $ModelC_layers \
        --output_dir "$OUTPUT_DIR" > "$LOG_FILE_2" 2>&1 
    
    # 等待当前超参数训练完成
    wait

    echo "正在训练:Lr=$LearningRate | Lora_r=$Lora_r | Lora_alpha=$Lora_alpha | version=$Version | ModelA_layers=$ModelA_layers | ModelC_layers=$ModelC_layers"
    echo "----------------------------------------"

done < "$HYPERPARAMS_FILE"


