# LLM-finetune-mistral
# Mistral 7B LoRA Fine-Tuning

This repository demonstrates **LoRA fine-tuning** of the Mistral 7B model using Hugging Face Transformers and PEFT.

---

## **Setup**

1. Clone the repo:
```bash
git clone <repo_url>
cd mistral-lora
```
2.Install dependencies:
```bash
pip install -r requirements.txt
```
3. Set Hugging Face token
```bash
export HF_TOKEN="your_hf_token_here"
```
## **Training**
Run the training script:
```bash
python train.py
```
The fine-tuned model will be saved in:
output/mistral_lora_out/final
## **Inference / Chat**
Run inference with:
```bash
python inference.py
```
You can modify prompts in inference.py 
