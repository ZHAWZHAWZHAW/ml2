import os
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load model and tokenizer
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Ensure the save directory exists
model_save_path = 'fine-tuned_model/t5_base'
os.makedirs(model_save_path, exist_ok=True)

# Save model and tokenizer
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"Model and tokenizer saved to {model_save_path}")
