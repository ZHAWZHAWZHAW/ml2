import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer
import pandas as pd
import gdown
import os

# Google Drive IDs zu den Modell-Dateien
config_json_id = "1L-bA0x_L3x-5g6zv7bx0x_8rYAIf0Ycu"
generation_config_json_id = "1DQm7qXQu0fr8nzBpwjr6oKy-cZ6-BSwl"
merges_txt_id = "1frNG28RZqCLp2KtP3MA0ibQvhzjPcLtH"
model_safetensors_id = "14A5Xk090NjQWJxJ4Oca7iIIxYBdnz6Kw"
special_tokens_map_json_id = "1XS66vZ-V05PSKtz70PvBYX_rxFozFgzp"
tokenizer_config_json_id = "10Xtby-l0a-VjpKvvIOP8zVd1fPfEeAoT"
vocab_json_id = "1bolVhYKdd4VCgph7_xTbUaxBle4Zf2_W"

# Google Drive URLs zu den Modell-Dateien
config_json_url = f"https://drive.google.com/uc?id={config_json_id}"
generation_config_json_url = f"https://drive.google.com/uc?id={generation_config_json_id}"
merges_txt_url = f"https://drive.google.com/uc?id={merges_txt_id}"
model_safetensors_url = f"https://drive.google.com/uc?id={model_safetensors_id}"
special_tokens_map_json_url = f"https://drive.google.com/uc?id={special_tokens_map_json_id}"
tokenizer_config_json_url = f"https://drive.google.com/uc?id={tokenizer_config_json_id}"
vocab_json_url = f"https://drive.google.com/uc?id={vocab_json_id}"

# Lokale Pfade zum Speichern der Modell-Dateien
bart_model_path = "streamlit_models/bart_model"

# Funktion zum Herunterladen der Dateien von Google Drive
def download_file(url, output):
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

# Erstellen des Verzeichnisses und Herunterladen der Dateien, falls nicht vorhanden
os.makedirs(bart_model_path, exist_ok=True)

download_file(config_json_url, os.path.join(bart_model_path, "config.json"))
download_file(generation_config_json_url, os.path.join(bart_model_path, "generation_config.json"))
download_file(merges_txt_url, os.path.join(bart_model_path, "merges.txt"))
download_file(model_safetensors_url, os.path.join(bart_model_path, "model.safetensors"))
download_file(special_tokens_map_json_url, os.path.join(bart_model_path, "special_tokens_map.json"))
download_file(tokenizer_config_json_url, os.path.join(bart_model_path, "tokenizer_config.json"))
download_file(vocab_json_url, os.path.join(bart_model_path, "vocab.json"))

# Laden des Modells und Tokenizers
tokenizer = BartTokenizer.from_pretrained(bart_model_path)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_path, use_safetensors=True)

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

    df = pd.DataFrame([bart_info], index=["BART Model"]).transpose()
    st.table(df)

    st.write("### Model Information")
    st.write(f"**Model Name:** `{bart_info['Model Name']}`")
    st.write(f"**Model Type:** `{bart_info['Model Type']}`")
    st.write(f"**Number of Parameters:** `{bart_info['Number of Parameters']}`")
    st.write(f"**Max Length:** `{bart_info['Max Length']}`")
    st.write(f"**Hidden Size:** `{bart_info['Hidden Size']}`")
    st.write(f"**Number of Attention Heads:** `{bart_info['Number of Attention Heads']}`")
    st.write(f"**Number of Layers:** `{bart_info['Number of Layers']}`")
    st.write(f"**Vocabulary Size:** `{bart_info['Vocabulary Size']}`")

if __name__ == "__main__":
    show_model_info_page()
