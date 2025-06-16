#!/usr/bin/env python3
"""
模型路径验证脚本
检查本地模型文件是否完整
"""

import os
import json
import sys

def check_model_files():
    """检查模型文件完整性"""
    model_path = "models/ShowUI-2B"
    
    print(f"🔍 检查模型路径: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        print("💡 请确保模型已下载到正确位置")
        return False
    
    print(f"✅ 模型路径存在: {model_path}")
    
    # 检查必需的配置文件
    required_files = [
        "config.json",
        "tokenizer.json", 
        "tokenizer_config.json",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "added_tokens.json"
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = os.path.join(model_path, file_name)
        if os.path.exists(file_path):
            print(f"✅ 配置文件存在: {file_name}")
        else:
            print(f"❌ 配置文件缺失: {file_name}")
            missing_files.append(file_name)
    
    # 检查模型权重文件
    weight_files = []
    for file_name in os.listdir(model_path):
        if file_name.endswith(('.safetensors', '.bin', '.pth')):
            weight_files.append(file_name)
            print(f"✅ 权重文件: {file_name}")
    
    if not weight_files:
        print("❌ 未找到模型权重文件 (.safetensors, .bin, .pth)")
        missing_files.append("model weights")
    
    # 检查config.json内容
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"✅ 模型类型: {config.get('model_type', 'unknown')}")
            print(f"✅ 架构: {config.get('architectures', ['unknown'])[0]}")
            print(f"✅ 隐藏层大小: {config.get('hidden_size', 'unknown')}")
        except Exception as e:
            print(f"⚠️ 读取config.json时出错: {e}")
    
    if missing_files:
        print(f"\n❌ 缺失文件: {', '.join(missing_files)}")
        return False
    else:
        print(f"\n✅ 模型文件检查完成，所有必需文件都存在")
        return True

def check_disk_space():
    """检查磁盘空间"""
    try:
        import shutil
        total, used, free = shutil.disk_usage("/models")
        
        print(f"\n💾 磁盘空间信息:")
        print(f"   总空间: {total // (1024**3):.1f} GB")
        print(f"   已使用: {used // (1024**3):.1f} GB")
        print(f"   可用空间: {free // (1024**3):.1f} GB")
        
        if free < 10 * 1024**3:  # 少于10GB
            print("⚠️ 可用磁盘空间不足10GB，可能影响训练")
        else:
            print("✅ 磁盘空间充足")
            
    except Exception as e:
        print(f"⚠️ 无法获取磁盘空间信息: {e}")

def check_permissions():
    """检查文件权限"""
    model_path = "/models/ShowUI-2B"
    
    if not os.path.exists(model_path):
        return False
        
    try:
        # 检查读权限
        test_file = os.path.join(model_path, "config.json")
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                f.read(100)  # 读取前100个字符
            print("✅ 模型文件读权限正常")
        return True
    except PermissionError:
        print("❌ 模型文件读权限不足")
        return False
    except Exception as e:
        print(f"⚠️ 检查权限时出错: {e}")
        return False

def main():
    """主函数"""
    print("🔧 ShowUI模型路径验证工具\n")
    
    all_good = True
    
    # 检查模型文件
    if not check_model_files():
        all_good = False
    
    # 检查磁盘空间
    check_disk_space()
    
    # 检查权限
    if not check_permissions():
        all_good = False
    
    print("\n" + "="*50)
    if all_good:
        print("🎉 模型路径验证通过！可以开始训练。")
        print("💡 运行 python test_environment.py 进行完整环境测试")
    else:
        print("❌ 模型路径验证失败，请检查上述问题。")
        print("💡 确保模型已正确下载到 /models/ShowUI-2B 目录")
        sys.exit(1)

if __name__ == "__main__":
    main()
