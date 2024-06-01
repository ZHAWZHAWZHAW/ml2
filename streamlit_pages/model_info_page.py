import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer, T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
import os

###################################################################################################
# Paths to the models, which were independently created and trained
    #bart_model_path = "pre-trained_model/bart_model"
    #t5_model_path = "fine-tuned_model/t5_base"
    #fine_tuned_model_path = "fine-tuned_model/results/final_model"

# Paths to the models, which were loaded from Google Drive
bart_model_path = "streamlit_models/bart_model"
t5_model_path = "streamlit_models/t5_base_model"
fine_tuned_model_path = "streamlit_models/fine_tuned_model"

# Load models and tokenizers
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_path)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_path, use_safetensors=True)
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_path, use_safetensors=True)
fine_tuned_tokenizer = T5Tokenizer.from_pretrained(fine_tuned_model_path)
fine_tuned_model = T5ForConditionalGeneration.from_pretrained(fine_tuned_model_path, use_safetensors=True)

# Function to retrieve model information
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
###################################################################################################

# Streamlit page to display model information
###################################################################################################
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
###################################################################################################

if __name__ == "__main__":
    show_model_info_page()