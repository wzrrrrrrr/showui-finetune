#!/usr/bin/env python3
"""
一个通用的、配置驱动的Hugging Face模型微调脚本。
支持多模态模型（如ShowUI-2B）和纯文本模型。
支持QLoRA和混合精度训练。
"""
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from PIL import Image
from datetime import datetime
import pprint


# ========================================================================================
# 1. 配置中心 (Config): 这是你唯一需要修改的地方！
#    我们以微调 ShowUI-2B 为例来填充它。
# ========================================================================================
class Config:
    """
    通用微调配置类。
    修改这里的参数以适应你的模型和数据。
    """
    # --- [侦查步骤1: 模型身份] ---
    # Hugging Face模型ID
    MODEL_ID = "showlab/ShowUI-2B"
    # 本地模型权重的根目录
    LOCAL_MODEL_DIR = "./models"
    # 是否信任远程代码 (对于自定义模型架构如Qwen-VL, Llava是必须的)
    TRUST_REMOTE_CODE = True
    # 模型类型: 'vision' 或 'text'。这会影响数据处理和模型加载方式。
    MODEL_TYPE = "vision"  # 'vision' or 'text'

    # --- [侦查步骤2: 处理器特殊癖好] ---
    # 这个字典会被直接传递给 AutoProcessor.from_pretrained
    # 对于纯文本模型，这个可以留空 {}
    PROCESSOR_KWARGS = {
        "min_pixels": 256 * 28 * 28,
        "max_pixels": 1344 * 28 * 28,
        "size": {'shortest_edge': 448, 'longest_edge': 448},
        "uigraph_train": True,  # ShowUI 特有参数
    }

    # --- [侦查步骤3: 数据格式与对话模板] ---
    # 数据集根目录
    DATASET_DIR = "./data"
    # 训练数据JSON/JSONL文件名
    TRAIN_JSON = "my_dataset/metadata.json"
    # 图片文件夹相对于JSON文件的路径 (仅对视觉模型有效)
    IMAGE_SUBDIR = "images"
    # 如果为None，则依赖processor自动加载。如果是字符串，则强制设置。
    # 对于ShowUI-2B，最好手动设置以确保一致性。
    CHAT_TEMPLATE = None

    # --- [侦查步骤4: LoRA靶心] ---
    # LoRA目标模块，通常需要根据模型检查来确定
    LORA_TARGET_MODULES = []  # 留空，触发自动检测！

    # --- [训练超参数] ---
    EXP_ID = f"finetune_{MODEL_ID.split('/')[-1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    EPOCHS = 3
    LR = 2e-4
    BATCH_SIZE = 1
    GRAD_ACCUMULATION_STEPS = 8
    MODEL_MAX_LENGTH = 2048

    # --- [硬件与性能配置] ---
    USE_QLORA = True
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    PRECISION = "bf16"  # "bf16", "fp16", "fp32"

    # --- [数据处理核心逻辑] ---
    # 这个方法定义了如何从一个JSON item转换成模型需要的输入格式
    @staticmethod
    def get_model_inputs(item, processor, cfg):
        # --- 适用于 ShowUI-2B / Qwen-VL 的逻辑 ---
        if cfg.MODEL_TYPE == 'vision':
            image_path = os.path.join(cfg.DATASET_DIR, os.path.dirname(cfg.TRAIN_JSON), cfg.IMAGE_SUBDIR,
                                      item['img_url'])
            instruction = item['element'][0]['instruction']
            point = str(item['element'][0]['point'])
            system_prompt = "Based on the screenshot of the page, I give a text description and you give its corresponding location..."

            messages = [
                {"role": "user", "content": [{"type": "text", "text": system_prompt}, {"type": "image"},
                                             {"type": "text", "text": instruction}]},
                {"role": "assistant", "content": point}
            ]

            text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            image = Image.open(image_path).convert('RGB')

            image_inputs_dict = processor.image_processor(images=image, return_tensors="pt")
            text_inputs = processor.tokenizer(text, return_tensors="pt", truncation=True,
                                              max_length=cfg.MODEL_MAX_LENGTH)

            inputs = {**text_inputs, **image_inputs_dict}
            inputs["labels"] = inputs["input_ids"].clone()
            return inputs

        # --- 适用于纯文本模型的逻辑 (例如: Llama, Mistral) ---
        elif cfg.MODEL_TYPE == 'text':
            # 假设你的JSON是 { "instruction": "...", "input": "...", "output": "..." } 格式
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')
            output_text = item.get('output', '')

            # 使用 Alpaca 格式的模板
            if input_text:
                prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            else:
                prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"

            full_text = prompt + output_text
            inputs = processor(full_text, return_tensors="pt", truncation=True, max_length=cfg.MODEL_MAX_LENGTH)
            inputs["labels"] = inputs["input_ids"].clone()
            return inputs

        else:
            raise ValueError(f"Unsupported MODEL_TYPE: {cfg.MODEL_TYPE}")


# ========================================================================================
# 2. 数据集类 (通用，无需修改)
# ========================================================================================
class UniversalDataset(Dataset):
    def __init__(self, cfg, processor):
        self.cfg = cfg
        self.processor = processor
        data_path = os.path.join(cfg.DATASET_DIR, cfg.TRAIN_JSON)
        with open(data_path, 'r', encoding='utf-8') as f:
            # 支持 .jsonl 和 .json
            if data_path.endswith('.jsonl'):
                self.data = [json.loads(line) for line in f if line.strip()]
            else:
                self.data = json.load(f)
        print(f"📊 加载了 {len(self.data)} 条训练数据从 {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            inputs = self.cfg.get_model_inputs(self.data[idx], self.processor, self.cfg)
            # DataLoader 会自动添加批次维度，所以我们移除它
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor) and inputs[key].dim() > 1:
                    inputs[key] = inputs[key].squeeze(0)
            return inputs
        except Exception as e:
            print(f"❌ 处理数据时出错! Item index: {idx}. Error: {e}")
            return None


# ========================================================================================
# 3. 训练器 (通用，无需修改)
# ========================================================================================
# ========================================================================================
# 3. 训练器 (通用，已修正模型加载逻辑)
# ========================================================================================
class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._print_config()
        self.model, self.processor = self._setup_model_and_processor()

    def _print_config(self):
        """打印所有配置参数，方便调试和记录。"""
        print("=" * 80)
        print("🚀 Universal Finetuner: Configuration Overview 🚀")
        print("=" * 80)
        config_dict = {k: v for k, v in self.cfg.__dict__.items() if not k.startswith('__') and not callable(v)}
        pprint.pprint(config_dict, indent=2, width=120)
        print("=" * 80)

    def _get_model_class(self):
        """
        根据Config智能判断并返回正确的模型类。
        这是确保能加载多模态模型的关键。
        """
        print(f"🧠 根据 MODEL_TYPE='{self.cfg.MODEL_TYPE}' 和 MODEL_ID='{self.cfg.MODEL_ID}' 判断模型类...")

        # 对于多模态模型，需要指定具体的类
        if self.cfg.MODEL_TYPE == 'vision':
            if "qwen2-vl" in self.cfg.MODEL_ID.lower() or "showui" in self.cfg.MODEL_ID.lower():
                from transformers import Qwen2VLForConditionalGeneration
                print(" -> 识别为 Qwen2VL 模型，使用 Qwen2VLForConditionalGeneration。")
                return Qwen2VLForConditionalGeneration
            # 在这里可以为其他视觉模型添加 elif 分支
            # elif "llava" in self.cfg.MODEL_ID.lower():
            #     from transformers import LlavaForConditionalGeneration
            #     print(" -> 识别为 Llava 模型，使用 LlavaForConditionalGeneration。")
            #     return LlavaForConditionalGeneration
            else:
                raise ValueError(f"未知的视觉模型类型: {self.cfg.MODEL_ID}。请在 _get_model_class 中添加支持。")

        # 对于纯文本模型，AutoModelForCausalLM 通常是安全的
        elif self.cfg.MODEL_TYPE == 'text':
            print(" -> 识别为纯文本模型，使用 AutoModelForCausalLM。")
            return AutoModelForCausalLM

        else:
            raise ValueError(f"不支持的 MODEL_TYPE: {self.cfg.MODEL_TYPE}")

    def _find_lora_target_modules(self, model):
        """
        自动查找所有可应用LoRA的线性层名称。
        """
        import bitsandbytes as bnb # 在方法内部导入，确保bnb可用

        print("🎯 正在自动检测LoRA目标模块...")
        lora_module_names = set()
        # 通用的、可能成为LoRA目标的模块名
        supported_lora_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "qkv_proj"
        ]

        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)):
                module_name = name.split('.')[-1]
                if module_name in supported_lora_modules:
                    lora_module_names.add(module_name)

        if 'lm_head' in lora_module_names:
            lora_module_names.remove('lm_head')

        print(f"✅ 自动查找到的LoRA目标模块: {list(lora_module_names)}")
        return list(lora_module_names)

    def _setup_model_and_processor(self):
        print("🔧 正在设置模型和处理器...")

        torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.half, "fp32": torch.float32}[self.cfg.PRECISION]
        bnb_config = None
        if self.cfg.USE_QLORA:
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype,
                                            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")

        model_path = os.path.join(self.cfg.LOCAL_MODEL_DIR, self.cfg.MODEL_ID.split('/')[-1])
        if not os.path.isdir(model_path):
            print(f"⚠️ 本地路径 {model_path} 不存在或不是一个目录，将尝试从 Hub 加载 {self.cfg.MODEL_ID}")
            model_path = self.cfg.MODEL_ID

        if self.cfg.MODEL_TYPE == 'vision':
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=self.cfg.TRUST_REMOTE_CODE,
                                                      **self.cfg.PROCESSOR_KWARGS)
        else:  # text
            processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=self.cfg.TRUST_REMOTE_CODE,
                                                      use_fast=False)

        if self.cfg.CHAT_TEMPLATE:
            print("👍 正在手动设置聊天模板...")
            processor.chat_template = self.cfg.CHAT_TEMPLATE
            if hasattr(processor, 'tokenizer'):  # for vision model
                processor.tokenizer.chat_template = self.cfg.CHAT_TEMPLATE

        # 【核心修正】使用 _get_model_class 获取正确的模型类
        model_class = self._get_model_class()
        model = model_class.from_pretrained(
            model_path, torch_dtype=torch_dtype, quantization_config=bnb_config,
            trust_remote_code=self.cfg.TRUST_REMOTE_CODE, device_map="auto", low_cpu_mem_usage=True
        )

        if self.cfg.USE_QLORA:
            model = prepare_model_for_kbit_training(model)

            if not self.cfg.LORA_TARGET_MODULES:
                target_modules = self._find_lora_target_modules(model)
            else:
                print(f"🎯 使用Config中指定的LoRA目标模块: {self.cfg.LORA_TARGET_MODULES}")
                target_modules = self.cfg.LORA_TARGET_MODULES

            lora_config = LoraConfig(
                r=self.cfg.LORA_R, lora_alpha=self.cfg.LORA_ALPHA,
                target_modules=target_modules,
                lora_dropout=self.cfg.LORA_DROPOUT, bias="none", task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        return model, processor

    def _collate_fn(self, batch):
        batch = [item for item in batch if item is not None]
        if not batch: return None

        keys = batch[0].keys()
        padded_batch = {}
        for key in keys:
            values = [item[key] for item in batch]
            if key in ['pixel_values', 'image_grid_thw']:
                padded_batch[key] = torch.stack(values, dim=0)
            elif key in ['input_ids', 'attention_mask', 'labels']:
                tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.processor
                padding_value = -100 if key == 'labels' else tokenizer.pad_token_id
                if padding_value is None: padding_value = 0  # Fallback for tokenizers without a pad_token

                padded_batch[key] = torch.nn.utils.rnn.pad_sequence(values, batch_first=True,
                                                                    padding_value=padding_value)
        return padded_batch

    def train(self):
        dataset = UniversalDataset(self.cfg, self.processor)
        dataloader = DataLoader(dataset, batch_size=self.cfg.BATCH_SIZE, shuffle=True, collate_fn=self._collate_fn,
                                num_workers=2)

        total_steps = (len(dataloader) // self.cfg.GRAD_ACCUMULATION_STEPS) * self.cfg.EPOCHS
        optimizer = AdamW(self.model.parameters(), lr=self.cfg.LR)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=min(100, total_steps // 10),
                                                    num_training_steps=total_steps)

        print("🏃 开始训练...")
        self.model.train()
        for epoch in range(self.cfg.EPOCHS):
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.cfg.EPOCHS}")
            for i, batch in enumerate(progress_bar):
                if batch is None: continue

                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

                try:
                    outputs = self.model(**batch)
                    loss = outputs.loss / self.cfg.GRAD_ACCUMULATION_STEPS
                    loss.backward()

                    if (i + 1) % self.cfg.GRAD_ACCUMULATION_STEPS == 0:
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                    progress_bar.set_postfix({'loss': f'{loss.item() * self.cfg.GRAD_ACCUMULATION_STEPS:.4f}'})
                except Exception as e:
                    print(f"\n⚠️ 训练步骤出错: {e}")
                    import traceback
                    traceback.print_exc()
                    optimizer.zero_grad()  # 清空梯度以防万一
                    continue

        print("🎉 训练完成!")
        save_path = f"./logs/{self.cfg.EXP_ID}"
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
        print(f"💾 模型和处理器已保存到 {save_path}")

    def _find_lora_target_modules(self, model):
        """
        自动查找所有可应用LoRA的线性层名称。
        """
        print("🎯 正在自动检测LoRA目标模块...")
        lora_module_names = set()
        # 我们只关心常见的Attention和MLP层名，以提高稳定性
        # 对于Qwen2系列，常见的线性层在qkv_proj, o_proj, up_proj, gate_proj, down_proj
        supported_lora_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "qkv_proj"
        ]

        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)):
                module_name = name.split('.')[-1]
                if module_name in supported_lora_modules:
                    lora_module_names.add(module_name)

        # 通常不建议对视觉模型的输出投影层和语言模型的输出层应用LoRA
        if 'lm_head' in lora_module_names:
            lora_module_names.remove('lm_head')

        print(f"✅ 自动查找到的LoRA目标模块: {list(lora_module_names)}")
        return list(lora_module_names)


# ========================================================================================
# 4. 执行入口
# ========================================================================================
if __name__ == "__main__":
    config = Config()
    trainer = Trainer(config)
    trainer.train()