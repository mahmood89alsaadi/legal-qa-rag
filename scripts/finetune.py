"""
Fine-tuning script using LoRA/QLoRA for legal MCQ reasoning.
Supports DeepSeek-R1 and other open-source models via HuggingFace.

Usage:
    python scripts/finetune.py \
        --config configs/finetune_lora.yaml \
        --base_model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --data data/processed/train_with_reasoning.json
"""

import json
import argparse
import yaml
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_training_data(path: str):
    with open(path) as f:
        samples = json.load(f)
    return Dataset.from_list([
        {"text": s["prompt"] + s["reasoning"] + "\nAnswer: " + s["answer"]}
        for s in samples
    ])


def build_qlora_model(model_name: str, lora_config: dict):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config.get("rank", 16),
        lora_alpha=lora_config.get("alpha", 32),
        lora_dropout=lora_config.get("dropout", 0.05),
        target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    train_dataset = load_training_data(args.data)
    model, tokenizer = build_qlora_model(args.base_model, config.get("lora", {}))

    training_args = TrainingArguments(
        output_dir=config.get("output_dir", "results/finetuned"),
        num_train_epochs=config.get("epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 4),
        gradient_accumulation_steps=config.get("grad_accum", 4),
        learning_rate=config.get("learning_rate", 2e-4),
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=config.get("max_seq_length", 2048),
        args=training_args,
    )

    print("Starting fine-tuning...")
    trainer.train()
    trainer.save_model()
    print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
