#!/bin/bash

# ShowUI-2B 微调训练脚本 - 适用于阿里云A10 GPU
# 使用方法: ./run_training.sh

set -e  # 遇到错误立即退出

# 生成实验ID
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXP_ID="showui_2b_finetune_${TIMESTAMP}"

echo "🚀 开始ShowUI-2B微调训练..."
echo "📅 实验ID: ${EXP_ID}"

# 激活虚拟环境
echo "🐍 激活虚拟环境..."
source showui_env/bin/activate

# 检查GPU状态
echo "🖥️ 检查GPU状态..."
nvidia-smi

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)/showui_core"

# 训练参数
TRAIN_ARGS="

    --model_id showlab/ShowUI-2B \
    --local_weight \
    --local_weight_dir /models \
    --precision bf16 \

    --use_qlora \
    --load_in_4bit \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_target_modules qkv_proj \
    --dataset_dir ./data \
    --train_dataset custom \
    --train_json metadata.json \
    --val_dataset custom \
    --val_json metadata.json \
    --batch_size 1 \
    --grad_accumulation_steps 8 \
    --lr 2e-4 \
    --warmup_steps 100 \
    --epochs 3 \
    --steps_per_epoch 500 \
    --gradient_checkpointing \
    --max_visual_tokens 1024 \
    --model_max_length 4096 \
    --log_base_dir ./logs \
    --exp_id ${EXP_ID} \
    --print_freq 10 \
    --workers 4 \
    --tune_visual_encoder false \
    --freeze_lm_embed
"

echo "🏃 开始训练..."
cd showui_core

# 使用tmux运行训练（可选，便于后台运行）
if command -v tmux &> /dev/null; then
    echo "📺 在tmux会话中运行训练..."
    tmux new-session -d -s "showui_training_${TIMESTAMP}" \
        "python train.py ${TRAIN_ARGS}; read -p 'Training finished. Press enter to exit...'"
    echo "✅ 训练已在tmux会话 'showui_training_${TIMESTAMP}' 中启动"
    echo "📋 使用以下命令查看训练进度:"
    echo "   tmux attach -t showui_training_${TIMESTAMP}"
else
    echo "🔄 直接运行训练..."
    python train.py ${TRAIN_ARGS}
fi

echo "🎉 训练脚本执行完成！"