#!/usr/bin/env python3
"""
ShowUI环境测试脚本
用于验证环境是否正确安装
"""

import sys
import os

def test_python_version():
    """测试Python版本"""
    print(f"🐍 Python版本: {sys.version}")
    if sys.version_info < (3, 8):
        print("❌ Python版本过低，需要3.8+")
        return False
    print("✅ Python版本符合要求")
    return True

def test_pytorch():
    """测试PyTorch安装"""
    try:
        import torch
        print(f"🔥 PyTorch版本: {torch.__version__}")
        print(f"🖥️ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🎮 GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        print("✅ PyTorch安装正常")
        return True
    except ImportError:
        print("❌ PyTorch未安装")
        return False

def test_transformers():
    """测试Transformers库"""
    try:
        import transformers
        print(f"🤗 Transformers版本: {transformers.__version__}")
        print("✅ Transformers安装正常")
        return True
    except ImportError:
        print("❌ Transformers未安装")
        return False

def test_other_dependencies():
    """测试其他依赖"""
    dependencies = [
        ("deepspeed", "DeepSpeed"),
        ("peft", "PEFT"),
        ("wandb", "WandB"),
        ("datasets", "Datasets"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV")
    ]
    
    all_good = True
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name}安装正常")
        except ImportError:
            print(f"❌ {name}未安装")
            all_good = False
    
    return all_good

def test_project_structure():
    """测试项目结构"""
    required_dirs = [
        "showui_core",
        "data",
        "custom_configs"
    ]

    required_files = [
        "showui_core/train.py",
        "setup_env.sh",
        "run_training.sh",
        "data/my_dataset/metadata.json"
    ]

    all_good = True

    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ 目录存在: {dir_path}")
        else:
            print(f"❌ 目录缺失: {dir_path}")
            all_good = False

    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ 文件存在: {file_path}")
        else:
            print(f"❌ 文件缺失: {file_path}")
            all_good = False

    return all_good

def test_local_model():
    """测试本地模型路径"""
    model_path = "/models/ShowUI-2B"
    if os.path.exists(model_path):
        print(f"✅ 本地模型路径存在: {model_path}")

        # 检查关键文件
        key_files = ["config.json", "tokenizer.json", "preprocessor_config.json"]
        for file_name in key_files:
            file_path = os.path.join(model_path, file_name)
            if os.path.exists(file_path):
                print(f"✅ 模型文件存在: {file_name}")
            else:
                print(f"❌ 模型文件缺失: {file_name}")
                return False
        return True
    else:
        print(f"❌ 本地模型路径不存在: {model_path}")
        print("💡 请确保模型已下载到 /models/ShowUI-2B 目录")
        return False

def main():
    """主测试函数"""
    print("🧪 开始环境测试...\n")
    
    tests = [
        ("Python版本", test_python_version),
        ("PyTorch", test_pytorch),
        ("Transformers", test_transformers),
        ("其他依赖", test_other_dependencies),
        ("项目结构", test_project_structure),
        ("本地模型", test_local_model)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 测试 {test_name}:")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "="*50)
    print("📊 测试结果汇总:")
    all_passed = True
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有测试通过！环境配置正确。")
        print("💡 你可以开始准备数据并运行训练了。")
    else:
        print("\n⚠️ 部分测试失败，请检查环境配置。")
        print("💡 运行 ./setup_env.sh 重新配置环境。")

if __name__ == "__main__":
    main()
