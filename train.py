#!/usr/bin/env python3
"""
ShowUI-2B微调训练脚本
支持NVIDIA GPU + CUDA，兼容peft和bitsandbytes
"""
import bitsandbytes as bnb
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
from functools import partial  # <--- 在这里加上这行


class ShowUIDataset(Dataset):
    """ShowUI数据集类"""

    def __init__(self, data_path, processor, args):
        self.data_path = data_path
        self.processor = processor
        self.args = args

        # 将系统提示定义为类的属性
        self.system_prompt = "Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1."

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

    # 在你的 ShowUIDataset 类中
    # 在你的 ShowUIDataset 类中
    # 在你的 ShowUIDataset 类中
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = "未定义"  # 初始化

        try:
            # 1. 提取信息 (这部分不变)
            image_filename = item['img_url']
            image_path = os.path.join(os.path.dirname(self.data_path), 'images', image_filename)
            element = item['element'][0]
            instruction = element.get('instruction', '目标区域')
            point = element['point']

            # 2. 构建 messages 列表 (用于生成文本模板)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.system_prompt},
                        {"type": "image"},  # 这里只需要一个占位符
                        {"type": "text", "text": instruction}
                    ]
                },
                {
                    "role": "assistant",
                    "content": str(point)
                }
            ]

            # ================ [ 全新、最关键的修改 ] ================
            # 严格遵循官方文档的“手动三步法”

            # 步骤 A: 像官方一样，用 apply_chat_template 只生成文本部分
            text = self.processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            # 步骤 B: 手动模拟 process_vision_info 的核心功能
            #   B.1 加载图片
            image = Image.open(image_path).convert('RGB')
            #   B.2 使用 processor 内部的 image_processor 对图片进行预处理，得到图片张量
            #       这是我们之前所有方案都缺失的最关键一步！
            image_inputs = self.processor.image_processor([image], return_tensors="pt")['pixel_values']

            # 步骤 C: 将最终的文本和图片张量一起送入 tokenizer 进行最后处理
            #       这里我们只用 tokenizer，因为它负责将文本和视觉占位符合并
            inputs = self.processor.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.args.model_max_length
                # padding 在 collate_fn 中处理
            )

            # 步骤 D: 将预处理好的图片张量添加到 inputs 字典中
            inputs['pixel_values'] = image_inputs

            # ================== [ 修改结束 ] ==================

            # 5. 后续处理 (这部分不变)
            inputs["labels"] = inputs["input_ids"].clone()
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor) and inputs[key].dim() > 1:
                    inputs[key] = inputs[key].squeeze(0)

            return inputs

        except Exception as e:
            # 错误处理逻辑保持不变
            import traceback
            print(f"❌ 处理数据时出错! 尝试的图片路径: {image_path}")
            print(f"错误类型: {type(e).__name__}, 错误信息: {e}")
            traceback.print_exc()  # 打印完整的堆栈，帮助我们看到底是哪一步错了

            # 2. 返回一个包含所有必要键的、完整的 fallback 字典
            # 创建一个虚拟的 pixel_values 张量
            dummy_pixel_values = torch.zeros((3, 448, 448), dtype=torch.float)
            dummy_input_ids = torch.zeros(100, dtype=torch.long)

            return {
                "pixel_values": dummy_pixel_values,
                "input_ids": dummy_input_ids,
                "attention_mask": torch.ones_like(dummy_input_ids),
                "labels": torch.full_like(dummy_input_ids, -100)  # 标签用-100填充
            }
        # ================ [ 修改结束 ] ================

def parse_args():
    parser = argparse.ArgumentParser(description="ShowUI-2B微调训练")

    # 基础参数
    parser.add_argument("--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--use_qlora", action="store_true", default=True)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--use_text_only", action="store_true", default=False, help="仅使用文本训练")

    # 模型参数
    parser.add_argument("--model_id", default="showlab/ShowUI-2B")
    parser.add_argument("--local_weight", action="store_true", default=True)
    parser.add_argument("--local_weight_dir", default="./models", help="本地模型路径")

    # 数据参数
    parser.add_argument("--dataset_dir", default="./data", type=str)
    parser.add_argument("--train_json", default="my_dataset/metadata.json", type=str)
    parser.add_argument("--model_max_length", default=2048, type=int)

    # LoRA参数
    parser.add_argument("--lora_r", default=16, type=int)
    parser.add_argument("--lora_alpha", default=32, type=int)
    parser.add_argument("--lora_dropout", default=0.1, type=float)

    # 训练参数
    parser.add_argument("--log_base_dir", default="./logs", type=str)
    parser.add_argument("--exp_id", default="showui", type=str)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--max_steps", default=None, type=int, help="最大训练步数")
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--grad_accumulation_steps", default=8, type=int)
    parser.add_argument("--warmup_steps", default=100, type=int)
    parser.add_argument("--print_freq", default=10, type=int)
    parser.add_argument("--save_steps", default=500, type=int)

    return parser.parse_args()


def setup_model_and_processor(args):
    """设置模型和处理器"""
    print("🔧 正在设置模型和处理器...")

    # 确定设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 使用CUDA: {torch.cuda.get_device_name()}")
        print(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        device = torch.device("cpu")
        print("💻 使用CPU")

    # 设置数据类型
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    # 量化配置
    bnb_config = None
    if args.use_qlora and args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    # 加载处理器
    try:
        print(f"🔧 正在从 '{args.model_id}' 加载处理器...")

        # 定义 Qwen2-VL 专用的尺寸参数
        min_pixels = 256 * 28 * 28
        max_pixels = 1344 * 28 * 28

        processor = AutoProcessor.from_pretrained(
            args.model_id,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            trust_remote_code=True
        )
        print("✅ 处理器加载成功，并已使用正确的 min/max_pixels 配置！")
        # size = {"shortest_edge": 448, "longest_edge": 448},

        CHAT_TEMPLATE = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        processor.chat_template = CHAT_TEMPLATE
        processor.tokenizer.chat_template = CHAT_TEMPLATE
        print("👍 已成功设置官方 ShowUI 聊天模板")

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
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

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

    target_modules = find_all_linear_names(model)
    print(f"🎯 自动查找到的LoRA目标模块: {target_modules}")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,  # <--- 使用自动查找到的列表
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def find_all_linear_names(model):
    """
    自动查找所有可应用LoRA的线性层名称。
    这次的实现更安全，只考虑了常见的Attention和MLP层名。
    """
    # 目标模块的常见名称
    # 对于Qwen2系列，常见的线性层在qkv_proj, o_proj, up_proj, gate_proj, down_proj
    # 我们这里列一个更通用的列表
    supported_lora_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "qkv_proj", "out_proj", "in_proj",  # 适用于其他模型的名字
        "fc1", "fc2"  # Vision Transformer 中的名字
    ]

    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt, torch.nn.Linear)):
            # 获取模块名的最后一部分
            module_name = name.split('.')[-1]
            # 只有当这个名字在我们支持的列表中时，才添加它
            if module_name in supported_lora_modules:
                lora_module_names.add(module_name)

    # 不对视觉编码器的投影层和语言模型的输出层应用LoRA
    # 这是一种常见的、能提高稳定性的做法
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    if 'proj' in lora_module_names:
        lora_module_names.remove('proj')  # 通常是ViT的输出投影，不建议LoRA

    return list(lora_module_names)


def main():
    # 问题: 当前的 padding 是在 __getitem__ 中通过 padding="max_length" 实现的。这意味着每个样本都会被填充到 model_max_length，可能会浪费大量显存和计算。
    # 建议: 使用动态批处理填充（Dynamic Padding）。这需要自定义一个 collate_fn。
    def collate_fn(batch, processor):
        # 将批次中的样本解构
        pixel_values = torch.cat([item['pixel_values'] for item in batch], dim=0)

        # 对文本部分进行动态填充
        text_inputs = processor.tokenizer.pad(
            [{"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]} for item in batch],
            return_tensors="pt",
            padding=True
        )

        # 对标签也进行填充，使用 -100 忽略 padding 部分的损失
        labels = processor.tokenizer.pad(
            [{"input_ids": item["labels"]} for item in batch],
            return_tensors="pt",
            padding=True
        )["input_ids"]
        labels[labels == processor.tokenizer.pad_token_id] = -100

        return {
            'pixel_values': pixel_values,
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask'],
            'labels': labels
        }

    args = parse_args()

    print("🚀 开始ShowUI-2B微调训练")
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
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                            collate_fn=partial(collate_fn, processor=processor))

    # 设置优化器和调度器
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(dataloader) // args.grad_accumulation_steps
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_steps)

    # 训练循环
    print("🏃 开始训练...")
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

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

        avg_loss = total_loss / max(len(dataloader), 1)
        print(f"Epoch {epoch + 1}/{args.epochs} - 平均损失: {avg_loss:.4f} - 总步数: {global_step}")

        # 如果达到最大步数，提前结束
        if args.max_steps and global_step >= args.max_steps:
            break

    print("🎉 训练完成!")

    # 保存模型
    save_path = f"{args.log_base_dir}/{args.exp_id}"
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    print(f"💾 模型权重已保存到 {save_path}")


if __name__ == "__main__":
    main()
