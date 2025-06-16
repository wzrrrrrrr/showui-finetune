# ShowUI-2B 微调项目

本项目用于在阿里云A10 GPU上微调ShowUI-2B模型，实现GUI自动化任务。

## 🚀 快速开始

### 1. 环境准备（阿里云服务器）

```bash
# 克隆项目
git clone <your-repo-url>
cd showui-finetune

# 运行环境设置脚本
./setup_env.sh

# 激活虚拟环境
source showui_env/bin/activate
```

### 2. 模型准备

确保ShowUI-2B模型已下载到服务器的 `/models` 目录：

```bash
# 检查模型是否存在
ls -la /models/ShowUI-2B/

# 如果不存在，下载模型
mkdir -p /models
cd /models
git clone https://huggingface.co/showlab/ShowUI-2B
```

### 3. 数据准备

1. 将你的截图放入 `data/my_dataset/` 目录
2. 编辑 `data/my_dataset/metadata.json` 文件，添加训练数据
3. 参考 `data/README.md` 了解数据格式

### 4. 开始训练

```bash
# 获取你的WandB API key: https://wandb.ai/authorize
./run_training.sh YOUR_WANDB_KEY
```

## 📁 项目结构

```
showui-finetune/
├── .gitignore                    # Git忽略文件
├── README.md                     # 项目说明
├── setup_env.sh                  # 环境设置脚本
├── run_training.sh               # 训练启动脚本
├── requirements.txt              # Python依赖
│
├── data/                         # 数据目录
│   ├── README.md                 # 数据格式说明
│   ├── metadata.json            # 训练数据元数据
│   └── my_dataset/               # 存放训练图片
│
├── showui_core/                  # ShowUI核心代码
│   ├── train.py                  # 主训练脚本
│   ├── model/                    # 模型定义
│   ├── data/                     # 数据处理
│   ├── main/                     # 训练器和评估器
│   ├── utils/                    # 工具函数
│   ├── ds_configs/               # DeepSpeed配置
│   └── requirements.txt          # 核心依赖
│
├── custom_configs/               # 自定义配置
│   └── my_finetune_config.yaml   # 微调配置文件
│
├── logs/                         # 训练日志（自动创建）
└── output/                       # 训练输出（自动创建）
```

## ⚙️ 硬件配置

- **服务器规格**: ecs.gn7i-c8g1.2xlarge
- **GPU**: 1x NVIDIA A10 (24GB显存)
- **CPU**: 8核心
- **内存**: 30GB RAM

## 🔧 配置说明

### 微调配置 (`custom_configs/my_finetune_config.yaml`)
- 使用QLoRA进行高效微调
- 4bit量化减少显存占用
- Flash Attention提升训练效率
- 针对A10 GPU优化的参数设置

### 训练参数
- **批次大小**: 1 (单GPU)
- **梯度累积**: 8步
- **学习率**: 2e-4
- **训练轮数**: 3
- **LoRA rank**: 16

## 📊 监控训练

训练过程会自动记录到WandB，你可以通过以下方式监控：

1. **WandB Dashboard**: 在线查看训练指标
2. **TensorBoard**: 本地日志查看
3. **tmux会话**: 实时查看训练输出

```bash
# 查看tmux会话
tmux list-sessions
tmux attach -t showui_training_YYYYMMDD_HHMMSS
```

## 🎯 使用场景

- 网页自动化操作
- 移动应用UI测试
- 桌面应用自动化
- GUI元素识别和交互

## 📝 注意事项

1. 确保WandB API key有效
2. 训练数据格式正确
3. 图片分辨率适中（建议≤1920x1080）
4. 定期保存检查点
5. 监控GPU显存使用情况

## 🔗 相关链接

- [ShowUI官方仓库](https://github.com/showlab/ShowUI)
- [WandB官网](https://wandb.ai/)
- [阿里云ECS文档](https://help.aliyun.com/product/25365.html)