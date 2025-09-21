import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from config import *

model_name = "mistralai/Mistral-7B-v0.1"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

# Base model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True
)
base_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map="auto", trust_remote_code=True)

# Load LoRA fine-tuned model
ft_model = PeftModel.from_pretrained(base_model, f"{OUTPUT_DIR}/final")
ft_model.eval()

def chat_with_model(prompt, max_new_tokens=128):
    torch.cuda.empty_cache()
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = ft_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.9, temperature=0.7, use_cache=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test
print(chat_with_model("Hello, how are you?"))
