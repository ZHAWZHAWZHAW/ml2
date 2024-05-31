import os
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Modell und Tokenizer laden
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Sicherstellen, dass der Speichern-Ordner existiert
model_save_path = 'fine-tuned_model/t5_base'
os.makedirs(model_save_path, exist_ok=True)

# Modell und Tokenizer speichern
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"Model and tokenizer saved to {model_save_path}")
