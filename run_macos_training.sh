#!/bin/bash

# ShowUI-2B macOS微调训练脚本
# 使用方法: ./run_macos_training.sh

set -e  # 遇到错误立即退出

# 生成实验ID
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXP_ID="showui_macos_${TIMESTAMP}"

echo "🚀 开始ShowUI-2B微调训练 (macOS版本)..."
echo "📅 实验ID: ${EXP_ID}"

# 激活虚拟环境
echo "🐍 激活虚拟环境..."
source showui_env/bin/activate

# 检查GPU状态
echo "🖥️ 检查GPU状态..."
echo "🍎 macOS系统 - 使用MPS加速"
python -c "import torch; print(f'MPS可用: {torch.backends.mps.is_available()}')"

# 检查模型文件
echo "📁 检查模型文件..."
if [ -d "models/ShowUI-2B" ]; then
    echo "✅ 找到本地模型: models/ShowUI-2B"
    ls -la models/ShowUI-2B/ | head -5
else
    echo "❌ 未找到本地模型路径: models/ShowUI-2B"
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
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 训练参数
TRAIN_ARGS="\
    --model_id showlab/ShowUI-2B \
    --local_weight \
    --local_weight_dir ./models \
    --precision fp32 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --dataset_dir ./data \
    --train_json my_dataset/metadata.json \
    --model_max_length 1024 \
    --batch_size 1 \
    --grad_accumulation_steps 1 \
    --lr 5e-4 \
    --warmup_steps 2 \
    --epochs 1 \
    --max_steps 5 \
    --log_base_dir ./logs \
    --exp_id ${EXP_ID} \
    --print_freq 1 \
    --save_steps 5"

echo "🏃 开始训练..."
echo "📋 训练参数: $TRAIN_ARGS"

# 运行训练
python train_macos.py $TRAIN_ARGS

echo "🎉 训练脚本执行完成！"
echo "📁 日志保存在: ./logs/${EXP_ID}"
