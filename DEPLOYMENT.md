# 阿里云部署指南

本文档详细说明如何在阿里云A10 GPU服务器上部署和运行ShowUI微调项目。

## 🖥️ 服务器规格

- **实例规格**: ecs.gn7i-c8g1.2xlarge
- **GPU**: 1x NVIDIA A10 (24GB显存)
- **CPU**: 8核心
- **内存**: 30GB RAM
- **操作系统**: Ubuntu 20.04 LTS (推荐)

## 📋 部署步骤

### 1. 连接服务器

```bash
# 使用SSH连接到阿里云服务器
ssh root@YOUR_SERVER_IP

# 或使用密钥文件
ssh -i your-key.pem root@YOUR_SERVER_IP
```

### 2. 系统初始化

```bash
# 更新系统
apt update && apt upgrade -y

# 安装基础工具
apt install -y git wget curl vim htop tmux

# 检查GPU驱动
nvidia-smi
```

### 3. 克隆项目

```bash
# 克隆你的项目仓库
git clone https://github.com/YOUR_USERNAME/showui-finetune.git
cd showui-finetune

# 检查项目结构
ls -la
```

### 4. 环境设置

```bash
# 运行环境设置脚本
./setup_env.sh

# 激活虚拟环境
source showui_env/bin/activate

# 测试环境
python test_environment.py
```

### 5. 模型准备

```bash
# 确保ShowUI-2B模型已下载到 /models 目录
# 如果还没有，可以使用以下命令下载：
mkdir -p /models
cd /models
git clone https://huggingface.co/showlab/ShowUI-2B

# 或者从本地上传模型文件
scp -r local_models/ShowUI-2B root@YOUR_SERVER_IP:/models/

# 检查模型文件
ls -la /models/ShowUI-2B/
```

### 6. 数据准备

```bash
# 上传你的训练数据到 data/my_dataset/ 目录
# 可以使用scp、rsync或其他方式

# 示例：使用scp上传数据
scp -r local_data/* root@YOUR_SERVER_IP:/path/to/showui-finetune/data/my_dataset/

# 检查数据
ls data/my_dataset/
head data/my_dataset/metadata.json
```

### 7. 开始训练

```bash
# 获取WandB API key: https://wandb.ai/authorize
# 运行训练
./run_training.sh YOUR_WANDB_KEY
```

## 🔧 配置优化

### GPU内存优化

如果遇到显存不足，可以调整以下参数：

```yaml
# 在 custom_configs/my_finetune_config.yaml 中
batch_size: 1                    # 减小批次大小
grad_accumulation_steps: 16      # 增加梯度累积
max_visual_tokens: 512           # 减少视觉token数量
model_max_length: 2048           # 减少序列长度
```

### 网络优化

```bash
# 如果下载模型较慢，可以设置代理
export HF_ENDPOINT=https://hf-mirror.com

# 或者预先下载模型到本地
huggingface-cli download showlab/ShowUI-2B --local-dir ./models/ShowUI-2B
```

## 📊 监控和调试

### 查看训练进度

```bash
# 查看tmux会话
tmux list-sessions
tmux attach -t showui_training_YYYYMMDD_HHMMSS

# 查看GPU使用情况
watch -n 1 nvidia-smi

# 查看日志
tail -f logs/showui_2b_finetune_*/tensorboard/events.out.tfevents.*
```

### 常见问题解决

1. **CUDA内存不足**
   ```bash
   # 清理GPU缓存
   python -c "import torch; torch.cuda.empty_cache()"
   ```

2. **网络连接问题**
   ```bash
   # 检查网络连接
   ping huggingface.co
   curl -I https://wandb.ai
   ```

3. **权限问题**
   ```bash
   # 确保脚本有执行权限
   chmod +x setup_env.sh run_training.sh test_environment.py
   ```

## 💾 数据备份

### 定期备份训练结果

```bash
# 创建备份脚本
cat > backup_training.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
tar -czf backup_${DATE}.tar.gz logs/ output/ data/my_dataset/metadata.json
echo "备份完成: backup_${DATE}.tar.gz"
EOF

chmod +x backup_training.sh
```

### 同步到对象存储

```bash
# 安装阿里云CLI工具
wget https://aliyuncli.alicdn.com/aliyun-cli-linux-latest-amd64.tgz
tar -xzf aliyun-cli-linux-latest-amd64.tgz
sudo mv aliyun /usr/local/bin/

# 配置阿里云CLI
aliyun configure

# 上传备份到OSS
aliyun oss cp backup_*.tar.gz oss://your-bucket/showui-backups/
```

## 🚀 性能优化建议

1. **使用Flash Attention**: 已在配置中启用
2. **启用梯度检查点**: 减少内存使用
3. **使用混合精度**: bf16精度训练
4. **优化数据加载**: 调整workers数量
5. **定期清理缓存**: 避免内存泄漏

## 📞 技术支持

如果遇到问题，可以：

1. 查看项目README.md
2. 检查日志文件
3. 运行环境测试脚本
4. 查看ShowUI官方文档
5. 在GitHub上提交Issue
