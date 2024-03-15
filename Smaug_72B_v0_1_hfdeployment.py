import os
print('LIBRARY INSTALATION STARTED')
os.system('python3 -m pip install torch tensorboard --quiet')
os.system('python3 -m pip install  --upgrade transformers datasets accelerate evaluate bitsandbytes --quiet')
os.system('python3 -m pip install -U flash-attn --no-build-isolation --quiet')
os.system('python3 -m pip install datasets trl ninja packaging peft --quiet')
os.system('python3 -m pip install diffusers safetensors  --quiet')
print('LIBRARY INSTALATION ENDED')


import os
from huggingface_hub import login

access_token_write = os.getenv("HUGGINGFACE_ACCESS_TOKEN_WRITE")


login(
  token=access_token_write,
  add_to_git_credential=True
)


import torch
import os
import sys
import json
import IPython
from datetime import datetime
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
    pipeline
)
from trl import SFTTrainer

# set device
device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
print(f'device: {device}')


def create_prompt_formats_squad2(sample):
    """
    Format various fields of the sample ('instruction','output')
    Then concatenate them using two newline characters
    :param sample: Sample dictionnary
    """
    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### Instruct: Summarize the below conversation."
    RESPONSE_KEY = "### Output:"
    END_KEY = "### End"

    blurb = f"\n{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY}"
    input_context = f"{sample['question']}" if sample["question"] else None
    response = f"{RESPONSE_KEY}\n{sample['answers']['text']}"
    end = f"{END_KEY}"

    parts = [part for part in [blurb, instruction, input_context, response, end] if part]

    formatted_prompt = "\n\n".join(parts)
    sample["text"] = formatted_prompt

    return sample


from datasets import load_dataset

# Load dataset from the hub
print("Preprocessing dataset squad-2")

dataset = load_dataset("rajpurkar/squad_v2")
#dataset = dataset.shuffle()
dataset_squad2 = dataset.map(create_prompt_formats_squad2) #for "rajpurkar/squad_v2"
# save datasets to disk
dataset_squad2["train"].to_json("train_dataset_squad2.json", orient="records")
dataset_squad2["validation"].to_json("test_dataset_squad2.json", orient="records")

print(dataset_squad2)


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import setup_chat_format

# Hugging Face model id
model_id = "abacusai/Smaug-72B-v0.1" #04 march 2024 and 08 march 2024 NO MEMORY

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

print('Load model and tokenizer')
# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(model_id,use_fast=True)
tokenizer.padding_side = 'right' # to prevent warnings

# We redefine the pad_token and pad_token_id with out of vocabulary token (unk_token)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.unk_token_id

# # set chat template to OAI chatML, remove if you start from a fine-tuned model
model, tokenizer = setup_chat_format(model, tokenizer)


text = "What is the capital of Canada?"

#device = 'cuda'
model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

generated_ids = model.generate(**model_inputs, temperature=0.8, top_k=1, top_p=1.0, repetition_penalty=1.4, min_new_tokens=16, max_new_tokens=128, do_sample=True)
decoded = tokenizer.decode(generated_ids[0])
print(decoded)


eval_tokenizer = AutoTokenizer.from_pretrained(model_id, add_bos_token=True, trust_remote_code=True, use_fast=False)
eval_tokenizer.pad_token = eval_tokenizer.eos_token

def gen(model,p, maxlen=1024, sample=True):
    toks = eval_tokenizer(p, return_tensors="pt")
    res = model.generate(**toks.to("cuda"), max_new_tokens=maxlen, do_sample=sample,num_return_sequences=1,temperature=0.9,num_beams=1,top_p=0.95,).to('cuda')
    return eval_tokenizer.batch_decode(res,skip_special_tokens=True)


print('ODEL GENERATION - ZERO SHOT')

index=10
dataset = dataset_squad2
#features: ['id', 'title', 'context', 'question', 'answers'],
prompt = dataset[index]['question']
summary = dataset[index]['answers']['text']

original_model = model
formatted_prompt = f"Instruct: Answer the following question.\n{prompt}\nOutput:\n" # for dataset_squad2
res = gen(original_model,formatted_prompt,100,)
output = res[0].split('Output:\n')[1]

dash_line = '-'.join('' for x in range(100))
print(dash_line)

print(f'QUESTION :\n{formatted_prompt}') # for dataset_squad2

print(dash_line)
print(f'ANSWER BY HUMAN:\n{summary}\n') # for dataset_squad2
print(dash_line)
print(f'MODEL GENERATION - ZERO SHOT:\n{output}') # for dataset_dialogsum_test AND dataset_squad2



print(len(dataset_squad2))

print()

print(model)


from peft import LoraConfig

# LoRA config based on QLoRA paper & Sebastian Raschka experiment
peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
)


from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="Smaug-72B-v0.1-hf-squad2-flash-attention-2", # directory to save and repository id
    num_train_epochs=3,                     # number of training epochs for POC
    per_device_train_batch_size=3,          # batch size per device during training
    gradient_accumulation_steps=2,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=10,                       # log every 10 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
    bf16=True,                              # use bfloat16 precision
    tf32=True,                              # use tf32 precision
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    push_to_hub=True,                       # push model to hub
    report_to="tensorboard",                # report metrics to tensorboard
)


#from trl import SFTTrainer

import trl

# explicitly import SFTTrainer
from trl.trainer import SFTTrainer

max_seq_length = 3072 # max sequence length for model and packing of the dataset

trainer = SFTTrainer(
    model=model,
    args=args,
    #train_dataset=dataset,
    train_dataset=dataset_squad2,
    #dataset_text_field="text", ### added for the summarization dataset
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    dataset_kwargs={
        "add_special_tokens": False,  # We template with special tokens
        "append_concat_token": False, # No need to add additional separator token
    }
)

print('TRAINING Started')
# start training, the model will be automatically saved to the hub and the output directory
trainer.train()

# save model
trainer.save_model()

print('TRAINING Ended')

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, pipeline

peft_model_id = "./Smaug-72B-v0.1-hf-squad2-flash-attention-2"

# Load Model with PEFT adapter
model = AutoPeftModelForCausalLM.from_pretrained(
  peft_model_id,
  device_map="auto",
  torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

# load into pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)


print('NEW MODEL: %s'%model)

print()

print('ORIGINAL MODEL-ID: %s'%model_id)

eval_tokenizer = AutoTokenizer.from_pretrained(model_id, add_bos_token=True, trust_remote_code=True, use_fast=False)
eval_tokenizer.pad_token = eval_tokenizer.eos_token

def gen_after_tunning(model,p, maxlen=1024, sample=True):
    toks = eval_tokenizer(p, return_tensors="pt")
    res = model.generate(**toks.to("cuda"), max_new_tokens=maxlen, do_sample=sample,num_return_sequences=1,temperature=0.9,num_beams=1,top_p=0.95,).to('cuda')
    return eval_tokenizer.batch_decode(res,skip_special_tokens=True)


print('VALIDATION Started')

index=10
dataset = dataset_squad2
TUNE_model = model

## squad2.0
#features: ['id', 'title', 'context', 'question', 'answers'],
prompt = dataset[index]['question']
summary = dataset[index]['answers']['text']

formatted_prompt = f"Instruct: Answer the following question.\n{prompt}\nOutput:\n" # for dataset_squad2

res = gen_after_tunning(TUNE_model,formatted_prompt,1024,)
#print(res[0])
output = res[0].split('Output:\n')[1]


dash_line = '-'.join('' for x in range(100))
print(dash_line)
print(f'QUESTION :\n{formatted_prompt}') # for dataset_squad2

print(dash_line)
print(f'ANSWER BY HUMAN:\n{summary}\n') # for dataset_squad2
print(dash_line)
print(f'MODEL GENERATION - AFTER THE TUNNING:\n{output}') # for dataset_squad2
