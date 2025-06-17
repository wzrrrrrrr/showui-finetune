#!/bin/bash

# ShowUI-2B 阿里云Linux微调训练脚本
# 适用于 NVIDIA A10 GPU
# 使用方法: ./run_linux_training.sh

set -e  # 遇到错误立即退出

# 生成实验ID
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXP_ID="showui_linux_${TIMESTAMP}"

echo "🚀 开始ShowUI-2B微调训练 (阿里云Linux版本)..."
echo "📅 实验ID: ${EXP_ID}"

# 激活虚拟环境
echo "🐍 激活虚拟环境..."
source showui_env/bin/activate

# 检查CUDA环境
echo "🖥️ 检查CUDA环境..."
echo "🐧 Linux系统 - 使用CUDA加速"
nvidia-smi
nvcc --version || echo "⚠️ nvcc未找到"

# 检查Python环境
echo "🐍 检查Python环境..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda}'); print(f'GPU数量: {torch.cuda.device_count()}')"

# 检查模型文件
echo "📁 检查模型文件..."
if [ -d "./models/ShowUI-2B" ]; then
    echo "✅ 找到本地模型: ./models/ShowUI-2B"
    ls -la ./models/ShowUI-2B/ | head -5
else
    echo "❌ 未找到本地模型路径: ./models/ShowUI-2B"
    echo "请确保模型已下载到正确位置"
    exit 1
fi

# 检查数据文件
echo "📊 检查数据文件..."
if [ -f "data/my_dataset/metadata.json" ]; then
    echo "✅ 找到训练数据: data/my_dataset/metadata.json"
    echo "数据条数: $(python -c "import json; print(len(json.load(open('data/my_dataset/metadata.json'))))")"
else
    echo "❌ 未找到训练数据: data/my_dataset/metadata.json"
    exit 1
fi

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0

# 训练参数
TRAIN_ARGS="\
    --model_id showlab/ShowUI-2B \
    --local_weight \
    --local_weight_dir ./models \
    --precision bf16 \
    --use_qlora \
    --load_in_4bit \
    --use_deepspeed \
    --ds_config ds_config.json \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --dataset_dir ./data \
    --train_json my_dataset/metadata.json \
    --model_max_length 2048 \
    --batch_size 1 \
    --grad_accumulation_steps 8 \
    --lr 2e-4 \
    --warmup_steps 100 \
    --epochs 3 \
    --max_steps 1000 \
    --log_base_dir ./logs \
    --exp_id ${EXP_ID} \
    --print_freq 10 \
    --save_steps 500 \
    --use_text_only"

echo "🏃 开始训练..."
echo "📋 训练参数: $TRAIN_ARGS"

export HF_HUB_OFFLINE=1

# 使用DeepSpeed运行训练
deepspeed train_linux.py $TRAIN_ARGS

echo "🎉 训练脚本执行完成！"
echo "📁 日志保存在: ./logs/${EXP_ID}"
