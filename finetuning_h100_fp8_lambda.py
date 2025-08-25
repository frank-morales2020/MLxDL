import torch
import os
import sys
import traceback
import argparse
from datasets import load_dataset
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    AutoConfig,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model
import torch.nn as nn
import numpy as np

import warnings
warnings.filterwarnings("ignore")


def custom_find_all_linear_names(model):
    """
    Dynamically finds all linear layer names in a PyTorch model.
    """
    linear_layer_names = set()
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear)):
            names = name.split(".")
            linear_layer_names.add(names[-1])
            
    if "lm_head" in linear_layer_names:
        linear_layer_names.remove("lm_head")
    
    return list(linear_layer_names)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    predictions = np.argmax(predictions, axis=-1)
    predictions = predictions[labels != -100]
    labels = labels[labels != -100]

    accuracy = (predictions == labels).astype(float).mean()
    
    return {"accuracy": accuracy}

def run_sft_gptoss(args):
    """
    Supervised fine-tuning script for GPT-OSS-20B using Hugging Face TRL and PEFT.
    """
    print("Setting up model and tokenizer loading...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map='auto',
        )
        
        model.config.use_cache = False

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            use_fast=True,
            padding_side='right',
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

        print("Model and Tokenizer loaded successfully.")

    except Exception as e:
        print(f"An error occurred during model or tokenizer loading: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("\nLoading and processing dataset...")
    try:
        dataset = load_dataset(args.dataset_name_or_path)
        print("Dataset loaded successfully:", dataset)

        def formatting_func(example):
            instruction = example.get('instruction', '')
            input_text = example.get('input', '')
            output_text = example.get('output', '')
            
            if input_text:
                full_prompt = f"{instruction}\n\n{input_text}"
            else:
                full_prompt = instruction

            return f"<|user|>\n{full_prompt}\n<|assistant|>\n{output_text}\n<|end|>"

        # FIX: The Alpaca dataset has columns: ['output', 'input', 'instruction']
        # The script now correctly removes these columns before processing.
        processed_dataset = dataset.map(
            lambda x: {"text": formatting_func(x)},
            remove_columns=['output', 'input', 'instruction']
        )
        print("Dataset formatted successfully.")

        train_test_split = processed_dataset['train'].train_test_split(test_size=0.1, seed=42)
        train_dataset = train_test_split['train']
        eval_dataset = train_test_split['test']
        print(f"Dataset split: Training examples = {len(train_dataset)}, Evaluation examples = {len(eval_dataset)}")

    except Exception as e:
        print(f"An error occurred during dataset loading or formatting: {e}")
        traceback.print_exc()
        sys.exit(1)

    if args.use_peft:
        print("\nSetting up PEFT (LoRA)...")
        try:
            target_modules = custom_find_all_linear_names(model)
            if target_modules:
                print(f"Found target modules for LoRA: {target_modules}")
            else:
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
                print(f"Using hard-coded target modules for LoRA: {target_modules}")
            
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            print("PEFT model setup complete.")

        except Exception as e:
            print(f"An error occurred during PEFT setup: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("\nSetting up TrainingArguments...")
    try:
        training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1, 
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=100,
        save_steps=5000,
        save_total_limit=3,
        bf16=False,
        fp16=True, 
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        report_to="tensorboard",
        push_to_hub=False,
        eval_strategy="steps",
        eval_steps=5000,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # REMOVE this line
        # predict_with_generate=True,
        )

        print("TrainingArguments setup complete.")

        # In your script, find the TrainingArguments section
    except Exception as e:
        print(f"An error occurred during TrainingArguments setup: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("\nInitializing SFTTrainer...")
    try:
        trainer = SFTTrainer(
            model=model,
            #tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            #max_seq_length=args.max_seq_length,
            formatting_func=formatting_func,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        print("SFTTrainer initialized.")

    except Exception as e:
        print(f"An error occurred during SFTTrainer initialization: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("\nStarting training...")
    try:
        trainer.train()
        print("Training finished.")

    except Exception as e:
        print(f"An error occurred during training: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("\nScript execution finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning Script for GPT-OSS-20B")
    parser.add_argument('--model_name_or_path', type=str, default='openai/gpt-oss-20b', help='Hugging Face model ID or local path')
    parser.add_argument('--dataset_name_or_path', type=str, default='yahma/alpaca-cleaned', help='Hugging Face dataset ID')
    parser.add_argument('--output_dir', type=str, default="./sft_results", help='Output directory for checkpoints and logs')
    parser.add_argument('--max_seq_length', type=int, default=128, help='Maximum sequence length (reduced for memory)')
    parser.add_argument('--per_device_train_batch_size', type=int, default=8, help='Batch size per GPU')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--num_train_epochs', type=float, default=0.1, help='Total number of training epochs')
    parser.add_argument('--use_peft', action='store_true', help='Use PEFT (LoRA) for fine-tuning')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA attention dimension')
    parser.add_argument('--lora_alpha', type=int, default=16, help='Alpha parameter for LoRA scaling')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='Dropout probability for LoRA layers')
    
    args, _ = parser.parse_known_args()
    
    run_sft_gptoss(args)