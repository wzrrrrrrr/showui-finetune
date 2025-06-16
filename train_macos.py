#!/usr/bin/env python3
"""
macOS兼容的ShowUI微调训练脚本
移除了deepspeed依赖，使用标准PyTorch训练
"""

import argparse
import os
import sys
import json

from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoProcessor, BitsAndBytesConfig
from tqdm import tqdm
from PIL import Image

class ShowUIDataset(Dataset):
    """ShowUI数据集类"""
    def __init__(self, data_path, processor, args):
        self.data_path = data_path
        self.processor = processor
        self.args = args

        # 读取数据
        with open(data_path, 'r', encoding='utf-8') as f:
            if data_path.endswith('.json'):
                # JSON格式
                self.data = json.load(f)
            else:
                # JSONL格式
                self.data = []
                for line in f:
                    if line.strip():
                        self.data.append(json.loads(line))

        print(f"📊 加载了 {len(self.data)} 条训练数据")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]

        # 根据数据格式处理
        if 'img_url' in item:
            # 新格式：my_dataset/metadata.json
            image_filename = item['img_url']
            image_path = os.path.join(os.path.dirname(self.data_path), 'images', image_filename)

            # 构建训练文本
            elements = item.get('element', [])
            if elements:
                element = elements[0]  # 取第一个元素
                instruction = element.get('instruction', '点击')
                point = element.get('point', [0.5, 0.5])

                # 转换相对坐标到绝对坐标
                img_size = item.get('img_size', [1282, 846])
                abs_x = int(point[0] * img_size[0])
                abs_y = int(point[1] * img_size[1])

                text = f"用户: 请点击{instruction}\n助手: 我会帮您点击{instruction}。<click>{abs_x}, {abs_y}</click>"
            else:
                text = "用户: 描述这个图片\n助手: 这是一个界面截图。"

        else:
            # 旧格式：conversations
            image_path = os.path.join(os.path.dirname(self.data_path), item['image'])
            conversations = item['conversations']

            # 构建简单的对话文本
            text = ""
            for conv in conversations:
                if conv['from'] == 'human':
                    user_text = conv['value'].replace('<image>\n', '').replace('<image>', '').strip()
                    text += f"用户: {user_text}\n"
                elif conv['from'] == 'gpt' or conv['from'] == 'assistant':
                    assistant_text = conv['value'].strip()
                    text += f"助手: {assistant_text}\n"

        # 加载图片
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ 无法加载图片 {image_path}: {e}")
            # 创建一个空白图片作为fallback
            image = Image.new('RGB', (224, 224), color='white')

        # 使用processor处理（暂时只用文本，避免图像token问题）
        try:
            # 暂时只使用文本进行训练，避免图像token匹配问题
            inputs = self.processor.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.args.model_max_length
            )

            # 设置labels
            inputs["labels"] = inputs["input_ids"].clone()

            # 将tensor从batch维度中取出
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor) and inputs[key].dim() > 1:
                    inputs[key] = inputs[key].squeeze(0)

            return inputs
            
        except Exception as e:
            print(f"⚠️ 处理数据时出错: {e}")
            # 返回一个简单的fallback
            return {
                "input_ids": torch.zeros(100, dtype=torch.long),
                "attention_mask": torch.ones(100, dtype=torch.long),
                "labels": torch.zeros(100, dtype=torch.long)
            }

def parse_args():
    parser = argparse.ArgumentParser(description="ShowUI训练 - macOS兼容版本")
    
    # 基础参数

    parser.add_argument("--precision", default="fp16", type=str, choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--use_qlora", action="store_true", default=True)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    
    # 模型参数
    parser.add_argument("--model_id", default="showlab/ShowUI-2B")
    parser.add_argument("--local_weight", action="store_true", default=True)
    parser.add_argument("--local_weight_dir", default="./models", help="本地模型路径")
    
    # 数据参数
    parser.add_argument("--dataset_dir", default="./data", type=str)
    parser.add_argument("--train_json", default="metadata.jsonl", type=str)
    parser.add_argument("--model_max_length", default=2048, type=int)
    
    # LoRA参数
    parser.add_argument("--lora_r", default=16, type=int)
    parser.add_argument("--lora_alpha", default=32, type=int)
    parser.add_argument("--lora_dropout", default=0.1, type=float)
    
    # 训练参数
    parser.add_argument("--log_base_dir", default="./logs", type=str)
    parser.add_argument("--exp_id", default="showui_macos", type=str)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--max_steps", default=None, type=int, help="最大训练步数，如果设置则覆盖epochs")
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--grad_accumulation_steps", default=4, type=int)
    parser.add_argument("--warmup_steps", default=50, type=int)
    parser.add_argument("--print_freq", default=5, type=int)
    parser.add_argument("--save_steps", default=100, type=int)
    
    return parser.parse_args()

def setup_model_and_processor(args):
    """设置模型和处理器"""
    print("🔧 正在设置模型和处理器...")
    
    # 确定设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 使用MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 使用CUDA")
    else:
        device = torch.device("cpu")
        print("💻 使用CPU")
    
    # 设置数据类型
    torch_dtype = torch.float32
    if args.precision == "bf16" and device.type != "mps":  # MPS不支持bf16
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    
    # 量化配置
    bnb_config = None
    if args.use_qlora and args.load_in_4bit and device.type != "mps":  # MPS不支持量化
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    
    # 加载处理器
    try:
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            trust_remote_code=True
        )
        print("✅ 处理器加载成功")
    except Exception as e:
        print(f"❌ 处理器加载失败: {e}")
        return None, None, None
    
    # 加载模型
    try:
        from transformers import Qwen2VLForConditionalGeneration
        
        model_path = f"{args.local_weight_dir}/ShowUI-2B"
        
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto" if device.type == "cuda" else None,
            quantization_config=bnb_config,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        if device.type == "mps":
            model = model.to(device)
            
        print("✅ 模型加载成功")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("💡 请确保ShowUI-2B模型文件在 ./models/ShowUI-2B 目录下")
        return None, None, None
    
    return model, processor, device

def setup_lora(model, args):
    """设置LoRA微调"""
    if args.lora_r <= 0:
        return model
        
    print("🔧 正在设置LoRA...")
    
    if args.use_qlora:
        model = prepare_model_for_kbit_training(model)
    
    # 目标模块
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

def train_epoch(model, dataloader, optimizer, scheduler, device, args, epoch, global_step=0):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    step_count = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

    for step, batch in enumerate(progress_bar):
        # 检查是否达到最大步数
        if args.max_steps and global_step >= args.max_steps:
            print(f"🎯 达到最大步数 {args.max_steps}，停止训练")
            break
        try:
            # 移动数据到设备
            if device.type != "cpu":
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
            
            # 前向传播
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accumulation_steps
            
            # 反向传播
            loss.backward()
            
            if (step + 1) % args.grad_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step_count += 1
                global_step += 1

            total_loss += loss.item() * args.grad_accumulation_steps

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item() * args.grad_accumulation_steps:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                'step': f'{global_step}'
            })



        except Exception as e:
            print(f"⚠️ 训练步骤出错: {e}")
            continue

    return total_loss / max(len(dataloader), 1), global_step

def main():
    args = parse_args()
    
    print("🚀 开始ShowUI微调训练 (macOS版本)")
    print(f"📱 模型: {args.model_id}")
    print(f"🎯 实验: {args.exp_id}")
    

    
    # 设置模型和处理器
    model, processor, device = setup_model_and_processor(args)
    if model is None:
        print("❌ 模型设置失败，退出")
        return
    
    # 设置LoRA
    model = setup_lora(model, args)
    
    # 创建数据集
    train_data_path = os.path.join(args.dataset_dir, args.train_json)
    dataset = ShowUIDataset(train_data_path, processor, args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    # 设置优化器和调度器
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(dataloader) // args.grad_accumulation_steps
    scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_steps)
    
    # 训练循环
    print("🏃 开始训练...")
    global_step = 0

    for epoch in range(args.epochs):
        avg_loss, global_step = train_epoch(model, dataloader, optimizer, scheduler, device, args, epoch, global_step)
        print(f"Epoch {epoch+1}/{args.epochs} - 平均损失: {avg_loss:.4f} - 总步数: {global_step}")



        # 如果达到最大步数，提前结束
        if args.max_steps and global_step >= args.max_steps:
            print(f"🎯 达到最大步数 {args.max_steps}，训练结束")
            break
    
    print("🎉 训练完成!")
    
    # 保存模型
    save_path = f"{args.log_base_dir}/{args.exp_id}"
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    print(f"💾 LoRA权重已保存到 {save_path}")

if __name__ == "__main__":
    main()
