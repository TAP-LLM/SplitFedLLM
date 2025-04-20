#!/bin/bash


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
HYPERPARAMS_FILE="/home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module-autosplit/batch_parallel_training/serial_2.txt"

# 逐行读取超参数文件
while IFS= read -r line || [[ -n "$line" ]]; do
    # 跳过注释行和空行
    [[ "$line" =~ ^# ]] || [[ -z "$line" ]] && continue
    
    # 解析超参数
    Version=$(echo $line | awk '{print $1}')
    # 动态选择分割点
    ModelA_layers=1
    ModelC_layers=1
    Add_DP_hidden=false
    Grad_noise=0
    Max_step=500
    Save_step=1000
    Lora_r=8
    Lora_alpha=16

    echo "正在训练: Version=$Version | A_layers=$ModelA_layers | C_layers=$ModelC_layers"
    # 获取当前时间戳
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    Server_device="cuda:1"
    Client_0="cuda:1"
    Client_1="cuda:1"
    Server_ip="10.143.12.74:8080"

    OUTPUT_DIR="/home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module-autosplit/output/For_BatchTrain/Version_${Version}"
    mkdir -p "$OUTPUT_DIR"
    # 定义日志文件路径
    LOG_FILE_1="$OUTPUT_DIR/trian_server_output.log"
    # 检查日志文件是否存在，如果存在则删除
    if [ -f "$LOG_FILE_1" ]; then
        rm "$LOG_FILE_1"
    fi
    LOG_FILE_2="$OUTPUT_DIR/trian_client_0_output.log"
    # 检查日志文件是否存在，如果存在则删除
    if [ -f "$LOG_FILE_2" ]; then
        rm "$LOG_FILE_2"
    fi
    # LOG_FILE_3="$OUTPUT_DIR/trian_client_1_output.log"
    # # 检查日志文件是否存在，如果存在则删除
    # if [ -f "$LOG_FILE_3" ]; then
    #     rm "$LOG_FILE_3"
    # fi
    # LOG_FILE_4="$OUTPUT_DIR/trian_client_2_output.log"
    # # 检查日志文件是否存在，如果存在则删除
    # if [ -f "$LOG_FILE_4" ]; then
    #     rm "$LOG_FILE_4"
    # fi
    # LOG_FILE_5="$OUTPUT_DIR/trian_client_3_output.log"
    # # 检查日志文件是否存在，如果存在则删除
    # if [ -f "$LOG_FILE_5" ]; then
    #     rm "$LOG_FILE_5"
    # fi

    # 复制运行脚本到输出文件夹
    cp "$0" "$OUTPUT_DIR/${Script_name%.*}_$TIMESTAMP.sh"

    # ----------------- 执行训练 -----------------
    # 使用 nohup 在后台运行 Python 脚本
    nohup python /home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module-autosplit/flserver_llama10_only_for_batchtrain.py \
        --modelA_layers $ModelA_layers \
        --modelC_layers $ModelC_layers \
        --device $Server_device \
        --server_ip $Server_ip \
        --lora_r $Lora_r \
        --lora_alpha $Lora_alpha \
        --max_step $Max_step \
        --save_step $Save_step \
        --add_DP_hidden $Add_DP_hidden \
        --grad_noise $Grad_noise \
        --output_dir $OUTPUT_DIR > "$LOG_FILE_1" 2>&1 &

    sleep 80

    nohup python /home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module-autosplit/flclient_llama10_Client_0_only_for_batchtrain.py \
        --modelA_layers $ModelA_layers \
        --modelC_layers $ModelC_layers \
        --device $Client_0 \
        --server_ip $Server_ip \
        --lora_r $Lora_r \
        --lora_alpha $Lora_alpha \
        --max_step $Max_step \
        --save_step $Save_step \
        --output_dir $OUTPUT_DIR \
        --grad_noise $Grad_noise > "$LOG_FILE_2" 2>&1 &


    sleep 15   
    nohup python /home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module-autosplit/flclient_llama10_Client_1_only_for_batchtrain.py \
        --modelA_layers $ModelA_layers \
        --modelC_layers $ModelC_layers \
        --device $Client_1 \
        --server_ip $Server_ip \
        --lora_r $Lora_r \
        --lora_alpha $Lora_alpha \
        --max_step $Max_step \
        --save_step $Save_step \
        --output_dir $OUTPUT_DIR \
        --grad_noise $Grad_noise > "$LOG_FILE_3" 2>&1 &

    # sleep 15
    
    # nohup python /home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module-autosplit/flclient_llama10_Client_2_only_for_batchtrain.py \
    #     --modelA_layers $ModelA_layers \
    #     --modelC_layers $ModelC_layers \
    #     --device $Client_1 \
    #     --server_ip $Server_ip \
    #     --lora_r $Lora_r \
    #     --lora_alpha $Lora_alpha \
    #     --max_step $Max_step \
    #     --save_step $Save_step \
    #     --output_dir $OUTPUT_DIR \
    #     --grad_noise $Grad_noise > "$LOG_FILE_4" 2>&1 &

    # sleep 15
    
    # nohup python /home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module-autosplit/flclient_llama10_Client_3_only_for_batchtrain.py \
    #     --modelA_layers $ModelA_layers \
    #     --modelC_layers $ModelC_layers \
    #     --device $Client_1 \
    #     --server_ip $Server_ip \
    #     --lora_r $Lora_r \
    #     --lora_alpha $Lora_alpha \
    #     --max_step $Max_step \
    #     --save_step $Save_step \
    #     --output_dir $OUTPUT_DIR \
    #     --grad_noise $Grad_noise > "$LOG_FILE_5" 2>&1 &

    wait
    echo "完成训练: Version=$Version | A_layers=$ModelA_layers | C_layers=$ModelC_layers"
    echo "----------------------------------------"
done < "$HYPERPARAMS_FILE"

# ----------------- 训练结束通知 -----------------
nohup python /home/zhangzishuai/web_login/buaa_login.py &
nohup python /home/zhangzishuai/Tools/mail_reminder.py \
    --mail_finished \
    --script_name $Script_name \
    --script_abs_path $Script_abs_path &

