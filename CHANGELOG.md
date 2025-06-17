# ShowUI Fine-tuning 更新日志

## 2025-06-16 - DeepSpeed移除更新

### 🗑️ 移除的功能
- **DeepSpeed**: 完全移除了DeepSpeed分布式训练框架
- **ZeRO优化**: 移除ZeRO内存优化
- **DeepSpeed配置**: 删除ds_config.json配置文件

### ✅ 保留的功能
- **QLoRA微调**: 4bit量化微调功能完全保留
- **混合精度**: BF16/FP16混合精度训练
- **梯度累积**: 模拟大批次训练
- **标准优化器**: AdamW + 线性学习率调度

### 🔧 解决的问题
- **兼容性问题**: 解决了peft与bitsandbytes的通信冲突
- **依赖简化**: 减少了复杂的分布式训练依赖
- **稳定性提升**: 使用标准PyTorch训练更稳定

### 📝 修改的文件
1. **train_linux.py** - 移除DeepSpeed相关代码，使用标准PyTorch训练
2. **run_linux_training.sh** - 删除DeepSpeed参数和启动方式
3. **run_training.sh** - 删除ds_zero参数
4. **requirements_linux.txt** - 删除deepspeed依赖
5. **setup_env.sh** - 删除DeepSpeed安装
6. **ds_config.json** - 删除配置文件
7. **README_LINUX.md** - 更新架构说明

## 2025-06-16 - WandB移除更新

### 🗑️ 移除的功能
- **WandB监控**: 完全移除了所有WandB相关代码和依赖
- **在线日志**: 不再支持在线训练监控

### ✅ 保留的功能
- **本地日志**: 训练过程仍会在终端显示详细信息
- **TensorBoard**: 仍支持TensorBoard本地监控
- **模型保存**: 训练完成后正常保存LoRA权重
- **所有训练功能**: 训练逻辑完全不变

### 📝 修改的文件
1. **train_linux.py** - 删除wandb导入和相关代码
2. **train_macos.py** - 删除wandb导入和相关代码
3. **run_linux_training.sh** - 删除wandb_key参数
4. **run_macos_training.sh** - 删除wandb_key参数
5. **run_training.sh** - 删除wandb_key参数
6. **requirements_linux.txt** - 删除wandb依赖
7. **showui_core/requirements.txt** - 删除wandb依赖
8. **README_LINUX.md** - 更新使用说明

### 🚀 新的使用方式

#### macOS:
```bash
./run_macos_training.sh
```

#### Linux (阿里云):
```bash
./run_linux_training.sh
```

#### 原始脚本:
```bash
./run_training.sh
```

### 📊 监控训练进度

现在可以通过以下方式监控训练：

1. **终端输出**: 实时显示损失值、学习率、训练步数
2. **TensorBoard**: 
   ```bash
   tensorboard --logdir=./logs
   ```
3. **日志文件**: 保存在 `./logs/实验ID/` 目录下

### 💡 优势
- **更简洁**: 减少了外部依赖
- **更稳定**: 避免网络连接问题
- **更快速**: 减少了网络上传开销
- **更私密**: 训练数据不会上传到第三方服务

### 🔄 如果需要恢复WandB
如果后续需要重新添加WandB支持，可以：
1. 在requirements中添加 `wandb>=0.15.0`
2. 在训练脚本中重新添加wandb相关代码
3. 在运行脚本中添加wandb_key参数
