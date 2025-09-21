import os

# Hugging Face API token
HF_TOKEN = os.getenv("HF_TOKEN", "your_token_here")  # set in GitHub secrets

# Paths
OUTPUT_DIR = "./output/mistral_lora_out"

# Training hyperparameters
TRAIN_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
MAX_STEPS = 50
LEARNING_RATE = 2e-4
