#!/bin/bash

# ShowUI微调环境设置脚本 - 适用于阿里云A10 GPU服务器
# 硬件规格: ecs.gn7i-c8g1.2xlarge (1x NVIDIA A10, 8 CPU cores, 30GB RAM)

echo "🚀 开始设置ShowUI微调环境..."

# 检查系统环境
echo "📋 检查系统环境..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 检测到macOS系统"
    # 检查是否有GPU (对于macOS，通常是Metal)
    system_profiler SPDisplaysDataType | grep -i metal || echo "未检测到Metal GPU支持"
else
    echo "🐧 检测到Linux系统"
    # 检查CUDA环境
    nvidia-smi || echo "未检测到NVIDIA GPU"
    nvcc --version || echo "未检测到CUDA"
fi

# 更新系统包
echo "📦 更新系统包..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - 检查Homebrew
    if ! command -v brew &> /dev/null; then
        echo "⚠️ 建议安装Homebrew来管理依赖"
        echo "访问: https://brew.sh"
    else
        brew update
    fi
else
    # Linux
    sudo apt update && sudo apt upgrade -y
fi

# 安装系统依赖
echo "🔧 安装系统依赖..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS依赖 (大部分已预装或通过Xcode Command Line Tools提供)
    echo "macOS系统依赖检查完成"
else
    # Linux依赖
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
fi

# 创建Python虚拟环境
echo "🐍 创建Python虚拟环境..."
python3 -m venv showui_env
source showui_env/bin/activate

# 升级pip
echo "⬆️ 升级pip..."
pip install --upgrade pip setuptools wheel

# 安装PyTorch (macOS版本)
echo "🔥 安装PyTorch..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    pip install torch torchvision torchaudio
else
    # Linux with CUDA
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

# 安装ShowUI核心依赖
echo "📚 安装ShowUI依赖..."
cd showui_core
pip install -r requirements.txt

# 安装额外的优化库
echo "⚡ 安装性能优化库..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "⚠️ macOS系统跳过CUDA特定的优化库安装"
    echo "flash-attn和liger-kernel主要用于CUDA环境"
else
    # Linux with CUDA
    echo "⚡ 安装Flash Attention (可能需要较长时间)..."
    pip install flash-attn --no-build-isolation || echo "⚠️ Flash Attention安装失败，继续..."

    echo "🚀 安装Liger Kernel..."
    pip install liger-kernel || echo "⚠️ Liger Kernel安装失败，继续..."
fi

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