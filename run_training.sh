#!/bin/bash

# ShowUI-2B 微调训练启动脚本
# 适用于单张 NVIDIA GPU
# 使用方法: ./run_training.sh

set -e  # 遇到任何错误立即退出

# --- 1. 实验设置 ---
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXP_ID="showui_finetune_${TIMESTAMP}"

echo "🚀 开始 ShowUI-2B 微调训练..."
echo "📅 实验ID: ${EXP_ID}"
echo "--------------------------------------------------"

# ... [环境检查部分保持不变] ...
# --- 2. 环境检查 ---
# 激活虚拟环境 (如果需要)
if [ -d "showui_env/bin" ]; then
    echo "🐍 正在激活虚拟环境..."
    source showui_env/bin/activate
fi
# 检查CUDA和PyTorch
echo "🖥️ 正在检查环境..."
nvidia-smi
echo ""
python -c "import torch; print(f'PyTorch 版本: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}'); print(f'Torch CUDA 版本: {torch.version.cuda}'); print(f'可用 GPU 数量: {torch.cuda.device_count()}')"
echo "--------------------------------------------------"
# 检查模型文件
echo "📁 正在检查模型文件..."
MODEL_NAME="ShowUI-2B"
if [ -d "./models/${MODEL_NAME}" ]; then
    echo "✅ 找到本地模型: ./models/${MODEL_NAME}"
    ls -lah ./models/${MODEL_NAME}/ | head -n 5
else
    echo "❌ 未找到本地模型路径: ./models/${MODEL_NAME}"
    echo "💡 请确保模型已下载到正确位置，或修改脚本中的 MODEL_NAME。"
    exit 1
fi
# 检查数据文件
DATA_JSON_PATH="data/my_dataset/metadata.json"
echo "📊 正在检查数据文件..."
if [ -f "${DATA_JSON_PATH}" ]; then
    DATA_COUNT=$(python -c "import json; print(len(json.load(open('${DATA_JSON_PATH}'))))")
    echo "✅ 找到训练数据: ${DATA_JSON_PATH} (共 ${DATA_COUNT} 条)"
else
    echo "❌ 未找到训练数据: ${DATA_JSON_PATH}"
    exit 1
fi
echo "--------------------------------------------------"

# --- 3. 设置环境变量 ---
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export HF_HUB_OFFLINE=1

# --- 4. 定义训练参数 (使用数组) ---
# 定义一个数组来存储参数，每个参数或值都是一个独立的元素
TRAIN_ARGS=(
    # -- 模型与路径参数 --
    --model_id "showlab/ShowUI-2B"
    --local_weight_dir "./models"

    # -- 精度与量化参数 --
    --precision "bf16"
    --use_qlora
    --load_in_4bit

    # -- LoRA 微调参数 --
    --lora_r 16
    --lora_alpha 32
    --lora_dropout 0.1

    # -- 数据集参数 --
    --dataset_dir "./data"
    --train_json "my_dataset/metadata.json"
    --model_max_length 2048

    # -- 训练超参数 --
    --batch_size 1
    --grad_accumulation_steps 8
    --lr 2e-4
    --warmup_steps 100
    --epochs 3
    --max_steps 1000

    # -- 日志与保存参数 --
    --log_base_dir "./logs"
    --exp_id "${EXP_ID}"
    --print_freq 10
    --save_steps 500
)

# --- 5. 执行训练 ---
echo "🏃 即将开始训练..."
echo "📋 最终训练参数:"
# 打印数组内容以供调试
printf " %q" "${TRAIN_ARGS[@]}"
echo ""
echo "--------------------------------------------------"

# 运行Python训练脚本
# 使用 "${TRAIN_ARGS[@]}" 来正确展开数组
python train.py "${TRAIN_ARGS[@]}"

echo "🎉 训练脚本执行完成！"
echo "📁 日志和模型权重保存在: ./logs/${EXP_ID}"