import streamlit as st
import logging
import os
import gdown

# Set logging level to suppress the special tokens warning
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

###################################################################################################
# Function to download files from Google Drive
def download_file(url, output):
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

# Google Drive URLs for the model files
model_files = {
    "bart": {
        "config.json": "https://drive.google.com/uc?id=1L-bA0x_L3x-5g6zv7bx0x_8rYAIf0Ycu",
        "generation_config.json": "https://drive.google.com/uc?id=1DQm7qXQu0fr8nzBpwjr6oKy-cZ6-BSwl",
        "merges.txt": "https://drive.google.com/uc?id=1frNG28RZqCLp2KtP3MA0ibQvhzjPcLtH",
        "model.safetensors": "https://drive.google.com/uc?id=14A5Xk090NjQWJxJ4Oca7iIIxYBdnz6Kw",
        "special_tokens_map.json": "https://drive.google.com/uc?id=1XS66vZ-V05PSKtz70PvBYX_rxFozFgzp",
        "tokenizer_config.json": "https://drive.google.com/uc?id=10Xtby-l0a-VjpKvvIOP8zVd1fPfEeAoT",
        "vocab.json": "https://drive.google.com/uc?id=1bolVhYKdd4VCgph7_xTbUaxBle4Zf2_W"
    },
    "t5_base": {
        "config.json": "https://drive.google.com/uc?id=1pu0fbzGhAbgQ35D82f_hKiMa7u-YiXOK",
        "generation_config.json": "https://drive.google.com/uc?id=1hCHgSwSByIoU_kdHZlQsH0xBnY9Jd7hH",
        "special_tokens_map.json": "https://drive.google.com/uc?id=1OpiuSOzybkEUn_aj6nsFQ_fbgMWHSAxt",
        "model.safetensors": "https://drive.google.com/uc?id=1Xfx-zFjZP7Xgo13OSh62w4Z9Hwq9YnIL",
        "added_tokens.json": "https://drive.google.com/uc?id=1gcnNSPX9HZV3tb4EPoXt3OI9k2fE77pG",
        "spiece.model": "https://drive.google.com/uc?id=165Wij00He548AGeTfC8556_kWUlyRppZ",
        "tokenizer_config.json": "https://drive.google.com/uc?id=1VVM-8twIRxfuIguD4HUvMS1A6EetsGTQ"
    },
    "fine_tuned": {
        "config.json": "https://drive.google.com/uc?id=1ptB72riEKrQjwfnZDZ4QI1JkSnG40e9A",
        "generation_config.json": "https://drive.google.com/uc?id=1Zdurv0fhhvJypZJQYQSS2T1WZRJUpYjb",
        "special_tokens_map.json": "https://drive.google.com/uc?id=1zAy2hTS9Mgj00_Q0x8lchZHwJyWo-g1F",
        "model.safetensors": "https://drive.google.com/uc?id=1OPWgs1-UeRSpeCmOfPoj4cmQymKMjWer",
        "added_tokens.json": "https://drive.google.com/uc?id=1glNnIyCp-op7RbmS-8GFXCl4thAgHKA8",
        "spiece.model": "https://drive.google.com/uc?id=1PcLiN4muOaZbB5sBTE_qrStT-r6b6KkJ",
        "tokenizer_config.json": "https://drive.google.com/uc?id=1VtOSN74rhuqoYtV084u9qwZ8I6tmBy5y"
    }
}

# Local paths to save the model files
model_paths = {
    "bart": "streamlit_models/bart_model",
    "t5_base": "streamlit_models/t5_base_model",
    "fine_tuned": "streamlit_models/fine_tuned_model"
}

# Create directories and download files if they don't exist
for model, files in model_files.items():
    os.makedirs(model_paths[model], exist_ok=True)
    for file_name, url in files.items():
        download_file(url, os.path.join(model_paths[model], file_name))
###################################################################################################
        
from streamlit_pages.summarizepdf_page import show_summarizepdf_page
from streamlit_pages.model_info_page import show_model_info_page
from streamlit_pages.home_page import show_home_page
from streamlit_pages.summarizeurl_page import show_summarizeurl_page
from streamlit_pages.model_validation_page import show_model_validation_page

st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
            color: #333;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 24px;
            font-size: 16px;
            cursor: pointer;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
        .stTextInput > div > div > input {
            border: 1px solid #ccc;
            padding: 10px;
            font-size: 16px;
            border-radius: 4px;
        }
        .stTextInput > div > div > input:focus {
            border-color: #4CAF50;
            box-shadow: 0 0 5px rgba(81, 203, 238, 1);
        }
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Summarize PDFs", "Summarize URLs", "Model Info", "Model Validation"])
if page == "Home":
    show_home_page()
elif page == "Summarize PDFs":
    show_summarizepdf_page()
elif page == "Summarize URLs":
    show_summarizeurl_page()
elif page == "Model Info":
    show_model_info_page()
elif page == "Model Validation":
    show_model_validation_page()

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("FS 2024: ML2 Project, Linus Schneeberger", unsafe_allow_html=True)
