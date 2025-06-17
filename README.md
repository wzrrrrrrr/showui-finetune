# ShowUI-2B 微调训练

基于ShowUI-2B模型的高效微调训练框架，支持QLoRA 4bit量化微调。

## 🖥️ 环境要求

- **GPU**: NVIDIA GPU (推荐A10/V100/A100)
- **显存**: 16GB+ (推荐24GB+)
- **系统**: Linux (Ubuntu 18.04+/CentOS 7+)
- **CUDA**: 11.8+
- **Python**: 3.8+

## 🔧 技术特性

- ✅ **QLoRA微调**: 4bit量化，大幅降低显存需求
- ✅ **peft兼容**: 完美兼容peft和bitsandbytes
- ✅ **标准PyTorch**: 移除DeepSpeed，避免兼容性问题
- ✅ **混合精度**: 支持BF16/FP16训练
- ✅ **梯度累积**: 模拟大批次训练效果

## 🚀 快速开始

### 1. 环境准备

```bash
# 1. 克隆项目
git clone <your-repo>
cd showui-finetune

# 2. 运行自动化环境设置
chmod +x setup_env.sh
./setup_env.sh

# 或手动安装
python -m venv showui_env
source showui_env/bin/activate
pip install -r requirements.txt
```

### 2. 模型准备

确保ShowUI-2B模型在 `./models/ShowUI-2B/` 目录下：

```bash
ls ./models/ShowUI-2B/
# 应该包含：
# - config.json
# - model.safetensors.index.json
# - model-*.safetensors
# - tokenizer.json
# - 等文件
```

### 3. 数据准备

确保训练数据在 `./data/my_dataset/` 目录下：

```bash
ls ./data/my_dataset/
# 应该包含：
# - metadata.json  (训练数据标注)
# - images/        (图片文件夹)
```

### 4. 开始训练

```bash
# 给脚本执行权限
chmod +x run_training.sh

# 开始训练
./run_training.sh
```

## ⚙️ 配置说明

### 训练参数

主要训练参数在 `run_training.sh` 中：

```bash
--model_id showlab/ShowUI-2B          # 模型ID
--local_weight_dir ./models           # 本地模型路径
--precision bf16                      # 精度 (bf16/fp16/fp32)
--use_qlora                          # 使用QLoRA
--load_in_4bit                       # 4bit量化
--lora_r 16                          # LoRA rank
--lora_alpha 32                      # LoRA alpha
--batch_size 1                       # 批次大小
--grad_accumulation_steps 8          # 梯度累积步数
--lr 2e-4                           # 学习率
--epochs 3                          # 训练轮数
--max_steps 1000                    # 最大步数
```

### 训练配置

标准PyTorch训练配置：

- **QLoRA**: 4bit量化微调
- **BF16**: 混合精度训练
- **梯度累积**: 模拟大批次训练
- **学习率调度**: 线性warmup

## 📊 监控训练

### 1. 终端输出

训练过程中会显示：
- 损失值变化
- 训练步数
- GPU内存使用
- 训练速度

### 2. 日志文件

训练日志保存在 `./logs/showui_YYYYMMDD_HHMMSS/`

## 🔧 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减少批次大小
   --batch_size 1
   --grad_accumulation_steps 4
   ```

2. **CUDA环境问题**
   ```bash
   # 检查CUDA环境
   nvidia-smi
   nvcc --version
   ```

3. **模型加载失败**
   ```bash
   # 检查模型文件完整性
   ls -la ./models/ShowUI-2B/
   ```

4. **依赖安装失败**
   ```bash
   # 使用国内镜像
   pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
   ```

### 性能优化

1. **启用Flash Attention**
   ```bash
   pip install flash-attn --no-build-isolation
   ```

2. **调整批次大小**
   - A10 24GB: batch_size=1, grad_accumulation_steps=8
   - 更大GPU: 可以增加batch_size

3. **使用混合精度**
   ```bash
   --precision bf16  # 推荐
   --precision fp16  # 备选
   ```

## 📈 训练结果

训练完成后：

在 `./logs/showui_YYYYMMDD_HHMMSS/` 目录下包含：

- **adapter_model.safetensors**: LoRA适配器权重
- **adapter_config.json**: LoRA配置文件
- **README.md**: 训练信息说明

## 🎯 下一步

1. **模型推理**: 使用训练好的LoRA权重进行推理
2. **效果评估**: 在测试集上评估模型性能
3. **参数调优**: 根据结果调整超参数
4. **数据扩充**: 添加更多训练数据

## 📞 支持

如遇问题，请检查：
1. GPU驱动和CUDA版本
2. Python环境和依赖版本
3. 模型和数据文件完整性
4. 系统资源使用情况
