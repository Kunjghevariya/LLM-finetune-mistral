import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from config import *

# 1️⃣ Hugging Face login
from huggingface_hub import login
login(HF_TOKEN)

# 2️⃣ Load dataset
dataset = load_dataset("tatsu-lab/alpaca")["train"]

def format_example(example):
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    response = example.get("output", "")
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{response}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
    return {"text": prompt}

processed_dataset = dataset.map(format_example)

# Split
split_point = len(processed_dataset)//2
from datasets import DatasetDict
processed_dataset = DatasetDict({
    "train": processed_dataset.select(range(split_point)),
    "test": processed_dataset.select(range(split_point, len(processed_dataset)))
})

# Tokenize
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = processed_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 4️⃣ Load model with 4-bit quantization
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True
)
base_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)

# LoRA
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(base_model, lora_config)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Trainer
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    warmup_steps=20,
    max_steps=MAX_STEPS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=5,
    save_strategy="steps",
    save_steps=25,
    report_to="wandb"
)

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset["train"], data_collator=data_collator)
trainer.train()
trainer.save_model(f"{OUTPUT_DIR}/final")
