#!/bin/bash

# 定义日志文件路径
LOG_FILE_1="/home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module/output_1.log"

# 检查日志文件是否存在，如果存在则删除
if [ -f "$LOG_FILE_1" ]; then
    rm "$LOG_FILE_1"
fi

LOG_FILE_2="/home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module/output_2.log"

# 检查日志文件是否存在，如果存在则删除
if [ -f "$LOG_FILE_2" ]; then
    rm "$LOG_FILE_2"
fi

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

# 使用 nohup 在后台运行 Python 脚本
nohup python /home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module/flserver_llama2.py \
    --do_train \
    --save_step 50 \
    --max_step 100000 \
    --output_dir /home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module/output/Friends/ReadCompre/version_4 > output_1.log 2>&1 &

sleep 10

nohup python /home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module/flclient_llama2.py \
    --do_train \
    --save_step 50 \
    --max_step 100000 \
    --output_dir /home/zhangzishuai/SplitFederated-LLaMA/Fed-Llama-module/output/Friends/ReadCompre/version_4 > output_2.log 2>&1 &