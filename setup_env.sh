#!/bin/bash

# ShowUI微调环境设置脚本 - 适用于阿里云A10 GPU服务器
# 硬件规格: ecs.gn7i-c8g1.2xlarge (1x NVIDIA A10, 8 CPU cores, 30GB RAM)

echo "🚀 开始设置ShowUI微调环境..."

# 检查系统环境
echo "📋 检查系统环境..."
echo "🐧 Linux系统"
# 检查CUDA环境
nvidia-smi || echo "未检测到NVIDIA GPU"
nvcc --version || echo "未检测到CUDA"

# 更新系统包
echo "📦 更新系统包..."
sudo apt update && sudo apt upgrade -y

# 安装系统依赖
echo "🔧 安装系统依赖..."
sudo apt install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    build-essential \
    python3-dev \
    python3-pip \
    python3-venv

# 创建Python虚拟环境
echo "🐍 创建Python虚拟环境..."
python3 -m venv showui_env
source showui_env/bin/activate

# 升级pip
echo "⬆️ 升级pip..."
pip install --upgrade pip setuptools wheel

# 安装PyTorch (CUDA版本)
echo "🔥 安装PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装ShowUI核心依赖
echo "📚 安装ShowUI依赖..."
cd showui_core
pip install -r requirements.txt

# 安装额外的优化库
echo "⚡ 安装性能优化库..."
echo "⚡ 安装Flash Attention (可能需要较长时间)..."
pip install flash-attn --no-build-isolation || echo "⚠️ Flash Attention安装失败，继续..."

echo "🚀 安装Liger Kernel..."
pip install liger-kernel || echo "⚠️ Liger Kernel安装失败，继续..."

# 验证安装
echo "✅ 验证安装..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU数量: {torch.cuda.device_count()}')"

# 检查本地模型路径
echo "🔍 检查本地模型路径..."
if [ -d "/models/ShowUI-2B" ]; then
    echo "✅ 找到本地模型: /models/ShowUI-2B"
    ls -la /models/ShowUI-2B/ | head -10
else
    echo "⚠️ 未找到本地模型路径: /models/ShowUI-2B"
    echo "请确保模型已下载到正确位置"
fi

if [ $? -eq 0 ]; then
    echo "🎉 环境设置完成！"
else
    echo "❌ 环境设置失败，请检查错误信息"
fi

# 创建必要的目录
echo "📁 创建项目目录..."
mkdir -p logs output data/my_dataset

echo "📝 环境设置完成！请运行以下命令激活环境："
echo "source showui_env/bin/activate"