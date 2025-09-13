# This script demonstrates how to distill knowledge from a teacher model
# (Qwen2.5-7B-Instruct as fallback) into a student model (Mistral 7B)
# using Parameter-Efficient Fine-Tuning (PEFT) with LoRA,
# and upload the fine-tuned model to Hugging Face Hub.

#pip install wheel
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
#pip install git+https://github.com/huggingface/transformers.git@main -q  # CRITICAL: Latest for Qwen3-Next support 
#pip install flash-attn --no-build-isolation -q  # For faster/stable attention
#pip install triton==3.2.0 -q 
#pip install trl peft huggingface_hub -q 
#pip install bitsandbytes accelerate -q 
#pip install datasets -q 

#pip install git+https://github.com/Dao-AILab/causal-conv1d.git -q  # For Qwen stability (if using Qwen3-Next) (optional)


import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from huggingface_hub import login, HfApi
from tqdm import tqdm
import os
import gc  # For memory management
from warnings import filterwarnings
filterwarnings('ignore')

# --- 1. CONFIGURATION ---
# Define the models
TEACHER_MODEL_NAME = "Qwen/Qwen3-Next-80B-A3B-Instruct"  # New model; requires latest Transformers (run pip upgrade above)
#TEACHER_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # Fallback: Stable, fast alternative (uncomment above once fixed)
STUDENT_MODEL_NAME = "mistralai/Mistral-7B-v0.1"

# Define the output directory for the fine-tuned model (local)
OUTPUT_DIR = "./mistral-7b-qwen-Next-80B-A3B-Instruct-distilled"

# Define Hugging Face Hub repository for upload
HF_REPO_ID = "frankmorales2020/mistral-7b-qwen-Next-80B-A3B-Instruct-distilled"


# Define LoRA configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Define training hyperparameters (LOWERED for stability)
LEARNING_RATE = 2e-4
BATCH_SIZE = 2  # Reduced from 4
NUM_EPOCHS = 3
GRADIENT_ACCUMULATION_STEPS = 1  # Reduced from 4

# --- 2. HUGGING FACE LOGIN ---
# Log in to Hugging Face Hub
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    HF_TOKEN = input("Enter your Hugging Face access token: ")
try:
    login(token=HF_TOKEN)
    print(f"Successfully logged in to Hugging Face Hub.")
except Exception as e:
    raise ValueError(f"Failed to log in to Hugging Face Hub: {e}")

# --- 3. LOAD TEACHER MODEL ---
print(f"Loading teacher model: {TEACHER_MODEL_NAME} with quantization...")
teacher_quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

try:
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_NAME,
        quantization_config=teacher_quant_config,
        device_map="auto",
        dtype=torch.bfloat16,  # Changed to bfloat16 for stability
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # For speed/stability (requires flash-attn)
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME)
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
except Exception as e:
    error_msg = f"Failed to load teacher model or tokenizer: {e}\nThis may be due to an outdated Transformers version. Run: pip install git+https://github.com/huggingface/transformers.git@main -q --upgrade"
    raise RuntimeError(error_msg)

# --- 4. DATASET CREATION (KNOWLEDGE DISTILLATION) ---
def generate_distillation_dataset(num_samples=20):  # Reduced for quick testing; increase to 100 once stable
    """
    Generates a distillation dataset using the local teacher model.
    """
    print("Generating distillation dataset with teacher model...")
    
    base_prompts = [
        "Explain the concept of quantum computing in one sentence.",
        "Write a Python function to compute Fibonacci numbers efficiently.",
        "Describe the process of photosynthesis.",
        "What are the key principles of object-oriented programming?",
    ]
    prompts = (base_prompts * (num_samples // len(base_prompts) + 1))[:num_samples]
    
    dataset_data = []
    teacher_model.eval()
    
    gen_batch_size = 4  # Increased for speed
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), gen_batch_size), desc="Generating batches"):
            batch_prompts = prompts[i:i + gen_batch_size]
            
            inputs = teacher_tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,  # Reduced for speed
            ).to(teacher_model.device)
            
            try:
                outputs = teacher_model.generate(
                    **inputs,
                    max_new_tokens=128,  # Reduced for speed
                    temperature=0.7,  # Slightly higher for variety, but faster
                    top_p=0.9,  # Nucleus sampling for efficiency
                    do_sample=True,
                    pad_token_id=teacher_tokenizer.eos_token_id,
                )
                
                for j, output in enumerate(outputs):
                    prompt_tokens = inputs.input_ids[j]
                    full_text = teacher_tokenizer.decode(output, skip_special_tokens=True)
                    response = full_text[len(teacher_tokenizer.decode(prompt_tokens, skip_special_tokens=True)):]
                    response = response.strip()
                    
                    text = f"### Instruction:\n{batch_prompts[j]}\n\n### Response:\n{response}\n"
                    dataset_data.append({"text": text})
            except Exception as e:
                print(f"Error during batch generation at index {i}: {e}")
                continue
    
    print(f"Dataset with {len(dataset_data)} samples created.")
    return Dataset.from_list(dataset_data)

# Generate the dataset
try:
    distillation_dataset = generate_distillation_dataset(num_samples=20)
except Exception as e:
    raise RuntimeError(f"Failed to generate distillation dataset: {e}")

# Early cleanup of teacher to free VRAM
print("Cleaning up teacher model to free memory...")
del teacher_model
torch.cuda.empty_cache()
gc.collect()

# --- 5. FINE-TUNING ---
print(f"Loading student model: {STUDENT_MODEL_NAME}...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,  # <-- Key: Change from bfloat16 to float16
    bnb_4bit_use_double_quant=True,
)

try:
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        dtype=torch.float16,  # <-- Change from bfloat16 to float16
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # For speed/stability
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Explicit padding

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )

    # Explicitly wrap model with LoRA before trainer (stabilizes init)
    model = get_peft_model(model, lora_config)

    from trl import SFTTrainer, SFTConfig

    from transformers import TrainingArguments

    # TrainingArguments: Standard training hyperparameters
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=5,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=1,  # Verbose for tracking
        optim="paged_adamw_8bit",
        fp16=True,  # Keep dtype fix
        bf16=False,
        report_to="none",
        remove_unused_columns=False,
    )

    print("Setting up the SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=distillation_dataset,
        peft_config=lora_config,
        args=training_args,  # Use TrainingArguments
        tokenizer=tokenizer,  # Legacy for v0.9.6
        max_seq_length=1024,  # SFT-specific: Passed to trainer
        dataset_text_field="text",
        packing=True,
    )

    
except Exception as e:
    raise RuntimeError(f"Failed to set up fine-tuning: {e}")

# --- 6. EXECUTE TRAINING ---
print(f"Executing fine-tuning for {NUM_EPOCHS} epochs...")
print("Starting fine-tuning...")
try:
    trainer.train()
except Exception as e:
    raise RuntimeError(f"Training failed: {e}")

# --- 7. SAVE AND UPLOAD MODEL ---
try:
    print(f"Saving the fine-tuned model locally to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Uploading model to Hugging Face Hub: {HF_REPO_ID}...")
    trainer.model.push_to_hub(HF_REPO_ID, use_auth_token=HF_TOKEN)
    tokenizer.push_to_hub(HF_REPO_ID, use_auth_token=HF_TOKEN)
    api = HfApi()
    model_card_content = """
# Mistral-7B-Qwen-Distilled

This model is a fine-tuned version of {STUDENT_MODEL_NAME}, distilled from {TEACHER_MODEL_NAME} using LoRA and SFTTrainer.

## Training Details
- **Teacher Model**: {TEACHER_MODEL_NAME}
- **Dataset**: Synthetic dataset of {len} samples generated by the teacher.
- **LoRA Config**: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}
- **Training Hyperparams**: {NUM_EPOCHS} epochs, learning rate {LEARNING_RATE}, batch size {BATCH_SIZE}

## Usage
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("{STUDENT_MODEL_NAME}", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("{STUDENT_MODEL_NAME}")
model = PeftModel.from_pretrained(model, "{HF_REPO_ID}")
```

Built with ❤️ using Hugging Face Transformers.
""".format(
        STUDENT_MODEL_NAME=STUDENT_MODEL_NAME,
        TEACHER_MODEL_NAME=TEACHER_MODEL_NAME,
        LORA_R=LORA_R,
        LORA_ALPHA=LORA_ALPHA,
        LORA_DROPOUT=LORA_DROPOUT,
        NUM_EPOCHS=NUM_EPOCHS,
        LEARNING_RATE=LEARNING_RATE,
        BATCH_SIZE=BATCH_SIZE,
        HF_REPO_ID=HF_REPO_ID,
        len=len(distillation_dataset)
    )
    api.upload_file(
        path_or_fileobj=model_card_content.encode(),
        path_in_repo="README.md",
        repo_id=HF_REPO_ID,
        repo_type="model",
        token=HF_TOKEN
    )
    print(f"Model and model card successfully uploaded to {HF_REPO_ID}!")
except Exception as e:
    print(f"Error during save or upload: {e}")
    raise


# --- 8. CLEANUP ---
# Free memory after training and upload
del model
torch.cuda.empty_cache()
gc.collect()
print("Distillation, fine-tuning, and upload complete!")