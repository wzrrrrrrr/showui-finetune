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
from functools import partial


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
            # 注意：这里我们直接用 tokenizer 的模板功能，更底层也更稳定
            text = self.processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False  # 训练时设为False，模型需要学习预测<|im_start|>assistant
            )

            # 步骤 B: 手动模拟 process_vision_info 的核心功能
            #   B.1 加载图片
            image = Image.open(image_path).convert('RGB')
            #   B.2 使用 processor 内部的 image_processor 对图片进行预处理，得到图片张量
            #       这是我们之前所有方案都缺失的最关键一步！
            image_inputs_dict = self.processor.image_processor(images=image, return_tensors="pt")

            # 步骤 C: 将最终的文本和图片占位符合并，进行最后处理
            #       这里我们只用 tokenizer，因为它负责将文本和视觉占位符合并
            inputs = self.processor.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.args.model_max_length
                # padding 在 collate_fn 中处理
            )

            # 步骤 D: 将预处理好的图片张量添加到 inputs 字典中
            # Qwen-VL模型期望这个键名为 'pixel_values'
            inputs['pixel_values'] = image_inputs
            inputs['image_grid_thw'] = image_inputs_dict['image_grid_thw']
            # ================== [ 修改结束 ] ==================

            # 5. 后续处理 (这部分不变)
            inputs["labels"] = inputs["input_ids"].clone()
            # 从 PyTorch 张量中移除批次维度（如果存在），因为DataLoader会自动添加
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor) and inputs[key].dim() > 1:
                    inputs[key] = inputs[key].squeeze(0)

            return inputs

        except Exception as e:
            # 错误处理逻辑保持不变
            import traceback
            print(f"❌ 处理数据时出错! Item index: {idx}, 尝试的图片路径: {image_path}")
            print(f"错误类型: {type(e).__name__}, 错误信息: {e}")
            traceback.print_exc()

            # 返回一个None，让collate_fn可以过滤掉它
            return None


def parse_args():
    parser = argparse.ArgumentParser(description="ShowUI-2B微调训练")

    # 基础参数
    parser.add_argument("--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--use_qlora", action="store_true", default=True)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)

    # 模型参数
    parser.add_argument("--model_id", default="showlab/ShowUI-2B")
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
    parser.add_argument("--exp_id", default="showui_finetune", type=str)
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

    model_path = os.path.join(args.local_weight_dir, args.model_id.split('/')[-1])

    # ==================== [ 全新、最关键的修改 ] ====================
    # 加载处理器
    try:
        print(f"🔧 正在从 '{model_path}' 加载处理器...")

        # 根据官方文档和代码，计算像素值
        # min_visual_tokens = 256, max_visual_tokens = 1344
        # 每个visual token对应一个 28x28 的patch
        min_pixels = 256 * 28 * 28  # 200704
        max_pixels = 1344 * 28 * 28  # 1053696

        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            # 传入这两个关键参数，确保image_processor能正确处理不同尺寸的图片
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            size = {'shortest_edge' : 448, 'longest_edge' : 448}
        )
        print("✅ 处理器加载成功，并已设置 min/max_pixels。")

        # 设置聊天模板 (这部分保持不变，做得很好)
        CHAT_TEMPLATE = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        processor.chat_template = CHAT_TEMPLATE
        processor.tokenizer.chat_template = CHAT_TEMPLATE
        print("👍 已成功设置官方 ShowUI 聊天模板")

    except Exception as e:
        print(f"❌ 处理器加载失败: {e}")
        return None, None, None
    # ==================== [ 修改结束 ] ====================

    # 加载模型
    try:
        from transformers import Qwen2VLForConditionalGeneration

        print(f"🔧 正在从 '{model_path}' 加载模型...")
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
        print(f"💡 请确保ShowUI-2B模型文件在 {model_path} 目录下")
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
        target_modules=target_modules,
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
    这个实现很好，既通用又安全。
    """
    supported_lora_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "qkv_proj"
    ]

    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt, torch.nn.Linear)):
            module_name = name.split('.')[-1]
            if module_name in supported_lora_modules:
                lora_module_names.add(module_name)

    # 不对语言模型的输出层应用LoRA，这是一种常见的、能提高稳定性的做法
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')

    # ShowUI中没有这些，但保留是好的实践
    if 'proj' in lora_module_names:
        lora_module_names.remove('proj')

    return list(lora_module_names)


def main():
    def collate_fn(batch, processor):
        # 过滤掉因为读取错误等原因返回的 None 样本
        batch = [item for item in batch if item is not None]
        if not batch:
            return None

        # 您的collate_fn实现得很好，可以保持原样
        try:
            keys = batch[0].keys()
            padded_batch = {}

            for key in keys:
                values = [item[key] for item in batch]

                if key == 'pixel_values':
                    # pixel_values 都是相同尺寸的，直接用 stack 合并
                    padded_batch[key] = torch.stack(values, dim=0)
                elif key in ['input_ids', 'attention_mask', 'labels']:
                    # 文本相关张量需要填充到批内最大长度
                    padding_value = -100 if key == 'labels' else processor.tokenizer.pad_token_id

                    # 使用 PyTorch 自带的 pad_sequence 进行填充，非常高效
                    padded_batch[key] = torch.nn.utils.rnn.pad_sequence(
                        values, batch_first=True, padding_value=padding_value
                    )
                else:
                    padded_batch[key] = values

            return padded_batch

        except Exception as e:
            import traceback
            print(f"❌ collate_fn 中出错!")
            traceback.print_exc()
            return None

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

    # 计算总步数
    num_update_steps_per_epoch = len(dataloader) // args.grad_accumulation_steps
    if args.max_steps is None:
        total_steps = args.epochs * num_update_steps_per_epoch
    else:
        total_steps = args.max_steps

    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )

    # 训练循环
    print("🏃 开始训练...")
    global_step = 0
    model.train()

    for epoch in range(args.epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}", total=len(dataloader))

        for step, batch in enumerate(progress_bar):
            if batch is None:
                continue

            if args.max_steps and global_step >= args.max_steps:
                print(f"🎯 达到最大步数 {args.max_steps}，停止训练")
                break

            try:
                # 移动数据到设备
                batch = {k: v.to(device) for k, v in batch.items()}

                # 前向传播
                outputs = model(**batch)
                loss = outputs.loss

                # 梯度累积
                loss = loss / args.grad_accumulation_steps

                # 反向传播
                loss.backward()

                if (step + 1) % args.grad_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # 仅在实际更新后记录日志
                    current_lr = scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({
                        'loss': f'{loss.item() * args.grad_accumulation_steps:.4f}',
                        'lr': f'{current_lr:.2e}',
                        'step': global_step
                    })

            except Exception as e:
                print(f"⚠️ 训练步骤出错: {e}")
                import traceback
                traceback.print_exc()
                # 清空梯度以防万一
                optimizer.zero_grad()
                continue

        # 如果达到最大步数，提前结束外层循环
        if args.max_steps and global_step >= args.max_steps:
            break

    print("🎉 训练完成!")

    # 保存模型
    save_path = os.path.join(args.log_base_dir, args.exp_id, f"checkpoint-final")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f"💾 模型和处理器已保存到 {save_path}")


if __name__ == "__main__":
    main()