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

HYPERPARAMS_FILE="/home/XXXX/SplitFederated-LLaMA/Fed-Llama-module/scripts_for_Record/IterTest_hyperparams_2.txt"
# 逐行读取超参数文件
while IFS= read -r line || [[ -n "$line" ]]; do
    # 跳过注释行和空行
    [[ "$line" =~ ^# ]] || [[ -z "$line" ]] && continue
    
    Version_dir=$(echo $line | awk '{print $1}') # 遍历不同文件夹要改的地方

    echo "正在测试:Version_dir=$Version_dir"

    for CheckpointPath in $(ls -d ${Version_dir}/model-A/checkpoint-*); do   
        # 提取检查点编号
        CheckpointStep=$(basename $CheckpointPath | sed 's/checkpoint-//')
        # 跳过检查点编号小于15000的检查点
        if [ "$CheckpointStep" -lt 24000 ]; then
            continue
        fi
        echo "进行检查点 ${CheckpointStep} 的测试"

        # 获取当前时间戳
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        # Max_predict_samples必须等于实际的大小，客户端生成的回答数量等于Max_predict_samples才计算指标
        Max_predict_samples=900
        Data_fold="/home/XXXX/SplitFederated-LLaMA/Dataset/Record/"
        Dataset_name="Record"
        Max_source_length=700
        Device_server="cuda:1"
        Device_client="cuda:1"
        Torch_dtype="float16"
        Test_file="val.jsonl" 
        # 这个是为了遍历检查点寻找最优检查点
        Passage_column="passage"
        Question_column='qas'
        Max_target_length=20
        Max_output_length=20
        Nohup_log="$OUTPUT_DIR/central_iter_predict.log"
        BatchSize=1
        Seed=42
        ServerIP="10.143.12.74:4070" 
        # GPU 0上占用8080 6080 GPU 1 占用4070 5040

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
        cp "$0" "$Lora_modelA_path/${Script_name%.*}_$TIMESTAMP.sh"

        # 使用 nohup 在后台运行 Python 脚本
        nohup python /home/XXXX/SplitFederated-LLaMA/Fed-Llama-module/flserver.py \
            --do_predict \
            --device $Device_server \
            --lora_modelB_path $Lora_modelB_path \
            --server_ip $ServerIP \
            --max_predict_samples $Max_predict_samples \
            --max_output_length $Max_output_length \
            --seed $Seed \
            --batch_size $BatchSize \
            --torch_dtype $Torch_dtype \
            --output_dir "$Output_dir" > "$LOG_FILE_1" 2>&1 &

        sleep 115

        python /home/XXXX/SplitFederated-LLaMA/Fed-Llama-module/flclient_id_0.py \
            --do_predict \
            --from_pickle \
            --iter_test \
            --device $Device_client \
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
            --question_column $Question_column \
            --output_dir "$Output_dir" > "$LOG_FILE_2" 2>&1
        wait
        echo "完成检查点 ${CheckpointStep} 的测试。"

    done

    echo "完成测试:Version_dir=$Version_dir"
    echo "----------------------------------------"

done < "$HYPERPARAMS_FILE"

