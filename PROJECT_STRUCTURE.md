# ShowUI-2B 项目结构

## 📁 核心文件

```
showui-finetune/
├── train.py                    # 主训练脚本
├── run_training.sh            # 训练启动脚本
├── setup_env.sh              # 环境配置脚本
├── requirements.txt          # Python依赖
├── README.md                 # 使用文档
└── CHANGELOG.md              # 更新日志
```

## 📂 数据目录

```
data/
├── my_dataset/
│   ├── metadata.json         # 训练数据标注
│   └── images/              # 图片文件
│       ├── img.png
│       └── img_1.png
└── README.md                # 数据说明
```

## 🤖 模型目录

```
models/
└── ShowUI-2B/               # 预训练模型
    ├── config.json
    ├── model.safetensors.index.json
    ├── model-*.safetensors
    └── ...
```

## 📊 输出目录

```
logs/
└── showui_YYYYMMDD_HHMMSS/  # 训练输出
    ├── adapter_model.safetensors
    ├── adapter_config.json
    └── README.md
```

## 🔧 环境目录

```
showui_env/                  # Python虚拟环境
├── bin/
├── lib/
└── ...
```

## 📚 原始代码

```
showui_core/                 # 原始ShowUI代码 (参考用)
├── train.py                # 原始训练脚本
├── requirements.txt        # 原始依赖
└── ...
```

## 🚀 快速开始

1. **环境设置**: `./setup_env.sh`
2. **开始训练**: `./run_training.sh`
3. **查看结果**: `ls logs/`

## 📝 文件说明

### 核心脚本
- **train.py**: 主训练脚本，支持QLoRA微调
- **run_training.sh**: 训练启动脚本，包含所有训练参数
- **setup_env.sh**: 自动化环境配置，安装所有依赖

### 配置文件
- **requirements.txt**: Python依赖列表
- **README.md**: 详细使用说明
- **CHANGELOG.md**: 版本更新记录

### 数据文件
- **metadata.json**: 训练数据标注文件
- **images/**: 训练图片目录

## 🎯 设计原则

- **简洁性**: 只保留必要的核心文件
- **专一性**: 专注Linux/CUDA环境
- **易用性**: 一键环境配置和训练启动
- **兼容性**: 完美支持peft + bitsandbytes
