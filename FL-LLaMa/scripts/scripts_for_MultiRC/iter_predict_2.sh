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

# 连接网络
nohup python /home/zhangzishuai/web_login/buaa_login.py &
# 启动邮件服务
Script_name=$(basename "$0")
Script_abs_path=$(cd "$(dirname "$0")"; pwd -P)/$(basename "$0")
nohup python /home/zhangzishuai/Tools/mail_reminder.py \
    --mail_start \
    --script_name $Script_name \
    --script_abs_path $Script_abs_path &

# 定义版本路径
Version_dir="/home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module/output/MultiRC/version_10_r32_alpha32_lr0.00001"

# 遍历所有检查点
for CheckpointPath in $(ls -d ${Version_dir}/model-A/checkpoint-*); do
    # 提取检查点编号
    CheckpointStep=$(basename $CheckpointPath | sed 's/checkpoint-//')
    # 跳过检查点编号小于15000的检查点
    if [ "$CheckpointStep" -lt 15000 ]; then
        continue
    fi
    echo "进行检查点 ${CheckpointStep} 的测试"

    # 删除目标文件（使用绝对路径，添加错误抑制）
    rm -f "${CheckpointPath}/Fl-llama_prediction.json" \
          "${CheckpointPath}/Fl-llama_prediction_for_eval.json" 2>/dev/null
    
    echo "已清理检查点 $CheckpointStep 的两个文件"

    # 获取当前时间戳
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    Data_fold="/home/zhangzishuai/SplitFederated-LLaMA/Dataset/MultiRC"
    Max_source_length=910
    Max_target_length=10
    Max_output_length=10
    Test_file="sample_val.jsonl"
    Dataset_name="MultiRC"
    Passage_column="passage"
    BatchSize=1 #测试一般batchsize为1
    Seed=42
    Torch_dtype="float16" #训练的时候必须要用float32不然学习率太大了
    #只在特定GPU上运行 ,单个服务器上运行多个程序需要更改GPU
    Device="cuda:1"
    #单个服务器上运行多个程序需要更改IP
    ServerIP="10.143.12.74:4070" 
    # GPU 0上占用8080 6080 GPU 1 占用4070 5040

    # Max_predict_samples必须等于实际的大小，客户端生成的回答数量等于Max_predict_samples才计算指标
    Max_predict_samples=950
    # 更新模型路径
    Lora_modelA_path="${Version_dir}/model-A/checkpoint-${CheckpointStep}"
    Lora_modelB_path="${Version_dir}/model-B/checkpoint-${CheckpointStep}"
    Lora_modelC_path="${Version_dir}/model-C/checkpoint-${CheckpointStep}"
    Output_dir="${Lora_modelA_path}"

    # 定义日志文件路径
    LOG_FILE_1="$Output_dir/iter_ckpt_output_server.log"
    # 检查日志文件是否存在，如果存在则删除
    if [ -f "$LOG_FILE_1" ]; then
        rm "$LOG_FILE_1"
    fi

    LOG_FILE_2="$Output_dir/iter_ckpt_output_client.log"
    # 检查日志文件是否存在，如果存在则删除
    if [ -f "$LOG_FILE_2" ]; then
        rm "$LOG_FILE_2"
    fi

    # 复制运行脚本到输出文件夹
    SCRIPT_NAME=$(basename "$0")
    cp "$0" "$Lora_modelA_path/${SCRIPT_NAME%.*}_$TIMESTAMP.sh"

    # 使用 nohup 在后台运行 Python 脚本
    nohup python /home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module/flserver_llama7.py \
        --do_predict \
        --device $Device \
        --lora_modelB_path $Lora_modelB_path \
        --server_ip $ServerIP \
        --max_predict_samples $Max_predict_samples \
        --seed $Seed \
        --batch_size $BatchSize \
        --torch_dtype $Torch_dtype \
        --output_dir "$Output_dir" > "$LOG_FILE_1" 2>&1 &

    sleep 115

    nohup python /home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module/flclient_llama7.py \
        --do_predict \
        --device $Device \
        --server_ip $ServerIP \
        --max_predict_samples $Max_predict_samples \
        --seed $Seed \
        --batch_size $BatchSize \
        --dataset_name $Dataset_name \
        --torch_dtype $Torch_dtype \
        --max_source_length $Max_source_length \
        --max_target_length $Max_target_length \
        --max_output_length $Max_output_length \
        --lora_modelA_path $Lora_modelA_path \
        --lora_modelC_path $Lora_modelC_path \
        --test_file $Test_file \
        --passage_column $Passage_column \
        --output_dir "$Output_dir" > "$LOG_FILE_2" 2>&1 &
    wait
    echo "完成检查点 ${CheckpointStep} 的测试。"

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