import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer, T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
import gdown
import os

# Google Drive IDs zu den BART Modell-Dateien
bart_config_json_id = "1L-bA0x_L3x-5g6zv7bx0x_8rYAIf0Ycu"
bart_generation_config_json_id = "1DQm7qXQu0fr8nzBpwjr6oKy-cZ6-BSwl"
bart_merges_txt_id = "1frNG28RZqCLp2KtP3MA0ibQvhzjPcLtH"
bart_model_safetensors_id = "14A5Xk090NjQWJxJ4Oca7iIIxYBdnz6Kw"
bart_special_tokens_map_json_id = "1XS66vZ-V05PSKtz70PvBYX_rxFozFgzp"
bart_tokenizer_config_json_id = "10Xtby-l0a-VjpKvvIOP8zVd1fPfEeAoT"
bart_vocab_json_id = "1bolVhYKdd4VCgph7_xTbUaxBle4Zf2_W"

# Google Drive IDs zu den T5 Modell-Dateien
t5_config_json_id = "1pu0fbzGhAbgQ35D82f_hKiMa7u-YiXOK"
t5_generation_config_json_id = "1hCHgSwSByIoU_kdHZlQsH0xBnY9Jd7hH"
t5_special_tokens_map_json_id = "1OpiuSOzybkEUn_aj6nsFQ_fbgMWHSAxt"
t5_model_safetensors_id = "1Xfx-zFjZP7Xgo13OSh62w4Z9Hwq9YnIL"
t5_added_tokens_json_id = "1gcnNSPX9HZV3tb4EPoXt3OI9k2fE77pG"
t5_spiece_model_id = "165Wij00He548AGeTfC8556_kWUlyRppZ"
t5_tokenizer_config_json_id = "1VVM-8twIRxfuIguD4HUvMS1A6EetsGTQ"

# Google Drive IDs zu den fine-tuned T5 Modell-Dateien
fine_tuned_config_json_id = "1ptB72riEKrQjwfnZDZ4QI1JkSnG40e9A"
fine_tuned_generation_config_json_id = "1Zdurv0fhhvJypZJQYQSS2T1WZRJUpYjb"
fine_tuned_special_tokens_map_json_id = "1zAy2hTS9Mgj00_Q0x8lchZHwJyWo-g1F"
fine_tuned_model_safetensors_id = "1OPWgs1-UeRSpeCmOfPoj4cmQymKMjWer"
fine_tuned_added_tokens_json_id = "1glNnIyCp-op7RbmS-8GFXCl4thAgHKA8"
fine_tuned_spiece_model_id = "1PcLiN4muOaZbB5sBTE_qrStT-r6b6KkJ"
fine_tuned_tokenizer_config_json_id = "1VtOSN74rhuqoYtV084u9qwZ8I6tmBy5y"

# Google Drive URLs zu den BART Modell-Dateien
bart_config_json_url = f"https://drive.google.com/uc?id={bart_config_json_id}"
bart_generation_config_json_url = f"https://drive.google.com/uc?id={bart_generation_config_json_id}"
bart_merges_txt_url = f"https://drive.google.com/uc?id={bart_merges_txt_id}"
bart_model_safetensors_url = f"https://drive.google.com/uc?id={bart_model_safetensors_id}"
bart_special_tokens_map_json_url = f"https://drive.google.com/uc?id={bart_special_tokens_map_json_id}"
bart_tokenizer_config_json_url = f"https://drive.google.com/uc?id={bart_tokenizer_config_json_id}"
bart_vocab_json_url = f"https://drive.google.com/uc?id={bart_vocab_json_id}"

# Google Drive URLs zu den T5 Modell-Dateien
t5_config_json_url = f"https://drive.google.com/uc?id={t5_config_json_id}"
t5_generation_config_json_url = f"https://drive.google.com/uc?id={t5_generation_config_json_id}"
t5_special_tokens_map_json_url = f"https://drive.google.com/uc?id={t5_special_tokens_map_json_id}"
t5_model_safetensors_url = f"https://drive.google.com/uc?id={t5_model_safetensors_id}"
t5_added_tokens_json_url = f"https://drive.google.com/uc?id={t5_added_tokens_json_id}"
t5_spiece_model_url = f"https://drive.google.com/uc?id={t5_spiece_model_id}"
t5_tokenizer_config_json_url = f"https://drive.google.com/uc?id={t5_tokenizer_config_json_id}"

# Google Drive URLs zu den fine-tuned T5 Modell-Dateien
fine_tuned_config_json_url = f"https://drive.google.com/uc?id={fine_tuned_config_json_id}"
fine_tuned_generation_config_json_url = f"https://drive.google.com/uc?id={fine_tuned_generation_config_json_id}"
fine_tuned_special_tokens_map_json_url = f"https://drive.google.com/uc?id={fine_tuned_special_tokens_map_json_id}"
fine_tuned_model_safetensors_url = f"https://drive.google.com/uc?id={fine_tuned_model_safetensors_id}"
fine_tuned_added_tokens_json_url = f"https://drive.google.com/uc?id={fine_tuned_added_tokens_json_id}"
fine_tuned_spiece_model_url = f"https://drive.google.com/uc?id={fine_tuned_spiece_model_id}"
fine_tuned_tokenizer_config_json_url = f"https://drive.google.com/uc?id={fine_tuned_tokenizer_config_json_id}"

# Lokale Pfade zum Speichern der Modell-Dateien
bart_model_path = "streamlit_models/bart_model"
t5_model_path = "streamlit_models/t5_base_model"
fine_tuned_model_path = "streamlit_models/fine_tuned_model"

# Funktion zum Herunterladen der Dateien von Google Drive
def download_file(url, output):
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

# Erstellen der Verzeichnisse und Herunterladen der Dateien, falls nicht vorhanden
os.makedirs(bart_model_path, exist_ok=True)
os.makedirs(t5_model_path, exist_ok=True)
os.makedirs(fine_tuned_model_path, exist_ok=True)

# Download der BART Modell-Dateien
download_file(bart_config_json_url, os.path.join(bart_model_path, "config.json"))
download_file(bart_generation_config_json_url, os.path.join(bart_model_path, "generation_config.json"))
download_file(bart_merges_txt_url, os.path.join(bart_model_path, "merges.txt"))
download_file(bart_model_safetensors_url, os.path.join(bart_model_path, "model.safetensors"))
download_file(bart_special_tokens_map_json_url, os.path.join(bart_model_path, "special_tokens_map.json"))
download_file(bart_tokenizer_config_json_url, os.path.join(bart_model_path, "tokenizer_config.json"))
download_file(bart_vocab_json_url, os.path.join(bart_model_path, "vocab.json"))

# Download der T5 Modell-Dateien
download_file(t5_config_json_url, os.path.join(t5_model_path, "config.json"))
download_file(t5_generation_config_json_url, os.path.join(t5_model_path, "generation_config.json"))
download_file(t5_special_tokens_map_json_url, os.path.join(t5_model_path, "special_tokens_map.json"))
download_file(t5_model_safetensors_url, os.path.join(t5_model_path, "model.safetensors"))
download_file(t5_added_tokens_json_url, os.path.join(t5_model_path, "added_tokens.json"))
download_file(t5_spiece_model_url, os.path.join(t5_model_path, "spiece.model"))
download_file(t5_tokenizer_config_json_url, os.path.join(t5_model_path, "tokenizer_config.json"))

# Download der fine-tuned T5 Modell-Dateien
download_file(fine_tuned_config_json_url, os.path.join(fine_tuned_model_path, "config.json"))
download_file(fine_tuned_generation_config_json_url, os.path.join(fine_tuned_model_path, "generation_config.json"))
download_file(fine_tuned_special_tokens_map_json_url, os.path.join(fine_tuned_model_path, "special_tokens_map.json"))
download_file(fine_tuned_model_safetensors_url, os.path.join(fine_tuned_model_path, "model.safetensors"))
download_file(fine_tuned_added_tokens_json_url, os.path.join(fine_tuned_model_path, "added_tokens.json"))
download_file(fine_tuned_spiece_model_url, os.path.join(fine_tuned_model_path, "spiece.model"))
download_file(fine_tuned_tokenizer_config_json_url, os.path.join(fine_tuned_model_path, "tokenizer_config.json"))

# Laden der Modelle und Tokenizer
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_path)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_path, use_safetensors=True)

t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_path, use_safetensors=True)

fine_tuned_tokenizer = T5Tokenizer.from_pretrained(fine_tuned_model_path)
fine_tuned_model = T5ForConditionalGeneration.from_pretrained(fine_tuned_model_path, use_safetensors=True)

def get_model_info(model):
    return {
        "Model Name": model.config.name_or_path,
        "Model Type": model.config.model_type,
        "Number of Parameters": f"{model.num_parameters():,}",
        "Max Length": str(getattr(model.config, 'n_positions', 'N/A') if hasattr(model.config, 'n_positions') else getattr(model.config, 'max_position_embeddings', 'N/A')),
        "Hidden Size": str(model.config.d_model),
        "Number of Attention Heads": str(model.config.num_heads if hasattr(model.config, 'num_heads') else model.config.encoder_attention_heads),
        "Number of Layers": str(model.config.num_layers if hasattr(model.config, 'num_layers') else model.config.encoder_layers),
        "Vocabulary Size": str(model.config.vocab_size),
    }

def show_model_info_page():
    st.header("📊 Model Information")

    bart_info = get_model_info(bart_model)
    t5_info = get_model_info(t5_model)
    fine_tuned_info = get_model_info(fine_tuned_model)

    df = pd.DataFrame([bart_info, t5_info, fine_tuned_info], index=["BART Model", "T5 Base Model", "Fine-tuned T5 Model"]).transpose()
    st.table(df)

    st.write("""
    ### Difference between T5 Base Model and Fine-tuned T5 Model
    The T5 Base model and the fine-tuned T5 model have the same architecture parameters because the fine-tuned model is based on the T5 Base model. 
    However, there are significant differences between the two models:
    
    - **Purpose:** 
        - **T5 Base Model:** This is the pre-trained version of the T5 model. It is trained on a large and diverse dataset to understand general language patterns.
        - **Fine-tuned T5 Model:** This model has been further trained (fine-tuned) on the XSum dataset, which is specifically designed for abstractive text summarization. The fine-tuning process adjusts the weights of the model to perform better on this specific task.
    
    - **Performance:** 
        - **T5 Base Model:** Good for a variety of general language tasks, but not specialized.
        - **Fine-tuned T5 Model:** Optimized for text summarization tasks, leading to better performance in those areas compared to the base model.
    
    - **Training Data:** 
        - **T5 Base Model:** Trained on a broad dataset, including the Colossal Clean Crawled Corpus (C4).
        - **Fine-tuned T5 Model:** Initially trained on the same broad dataset as the T5 Base Model, but further trained (fine-tuned) on the XSum dataset, which is a large dataset specifically for summarization tasks.

    In summary, while the architecture of both models is the same, the fine-tuned model has learned to better handle summarization tasks due to its specialized training on the XSum dataset.
    """)

if __name__ == "__main__":
    show_model_info_page()
