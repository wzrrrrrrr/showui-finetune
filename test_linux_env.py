#!/usr/bin/env python3
"""
测试Linux环境是否准备就绪
"""

import sys
import torch
import json
import os
from PIL import Image

def test_environment():
    print("🧪 测试Linux环境...")
    
    # 1. 检查Python版本
    print(f"🐍 Python版本: {sys.version}")
    
    # 2. 检查PyTorch和CUDA
    print(f"🔥 PyTorch版本: {torch.__version__}")
    print(f"🚀 CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎯 CUDA版本: {torch.version.cuda}")
        print(f"💾 GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")
    
    # 3. 检查关键依赖
    try:
        import transformers
        print(f"🤗 Transformers版本: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers未安装")
    
    try:
        import peft
        print(f"🔧 PEFT版本: {peft.__version__}")
    except ImportError:
        print("❌ PEFT未安装")
    
    try:
        import deepspeed
        print(f"⚡ DeepSpeed版本: {deepspeed.__version__}")
    except ImportError:
        print("❌ DeepSpeed未安装")
    
    try:
        import bitsandbytes
        print(f"🔢 BitsAndBytes版本: {bitsandbytes.__version__}")
    except ImportError:
        print("❌ BitsAndBytes未安装")
    
    # 4. 检查模型文件
    model_path = "./models/ShowUI-2B"
    if os.path.exists(model_path):
        print(f"✅ 模型文件存在: {model_path}")
        files = os.listdir(model_path)
        print(f"  文件数量: {len(files)}")
        key_files = ['config.json', 'model.safetensors.index.json']
        for key_file in key_files:
            if key_file in files:
                print(f"  ✅ {key_file}")
            else:
                print(f"  ❌ {key_file}")
    else:
        print(f"❌ 模型文件不存在: {model_path}")
    
    # 5. 检查数据文件
    data_path = "./data/my_dataset/metadata.json"
    if os.path.exists(data_path):
        print(f"✅ 数据文件存在: {data_path}")
        with open(data_path, 'r') as f:
            data = json.load(f)
        print(f"  数据条数: {len(data)}")
        
        # 检查图片文件
        img_dir = "./data/my_dataset/images"
        if os.path.exists(img_dir):
            img_files = os.listdir(img_dir)
            print(f"  图片文件数量: {len(img_files)}")
        else:
            print(f"  ❌ 图片目录不存在: {img_dir}")
    else:
        print(f"❌ 数据文件不存在: {data_path}")
    
    # 6. 测试GPU内存
    if torch.cuda.is_available():
        print("\n💾 测试GPU内存...")
        device = torch.device("cuda:0")
        
        # 分配一些内存测试
        try:
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            z = torch.matmul(x, y)
            print("✅ GPU计算测试通过")
            
            # 显示内存使用
            allocated = torch.cuda.memory_allocated(0) / 1e9
            cached = torch.cuda.memory_reserved(0) / 1e9
            print(f"  已分配内存: {allocated:.2f}GB")
            print(f"  缓存内存: {cached:.2f}GB")
            
            # 清理
            del x, y, z
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ GPU测试失败: {e}")
    
    print("\n🎉 环境检查完成!")

if __name__ == "__main__":
    test_environment()
