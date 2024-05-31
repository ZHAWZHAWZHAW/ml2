
import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import BartTokenizer, BartForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
# Pfade zu den Modellen
bart_model_path = "streamlit_models/bart_model"
t5_model_path = "streamlit_models/t5_base_model"
fine_tuned_model_path = "streamlit_models/fine_tuned_model"
# Laden der Modelle und Tokenizer
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_path)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_path, use_safetensors=True)
t5_base_tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
t5_base_model = T5ForConditionalGeneration.from_pretrained(t5_model_path, use_safetensors=True)
fine_tuned_tokenizer = T5Tokenizer.from_pretrained(fine_tuned_model_path)
fine_tuned_model = T5ForConditionalGeneration.from_pretrained(fine_tuned_model_path, use_safetensors=True)
def fetch_url_content(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            text = ' '.join([para.get_text() for para in paragraphs])
            return text
        else:
            st.error(f"Failed to fetch the URL content. Status code: {response.status_code}")
            return ""
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return ""
def summarize_text_with_bart(text):
    inputs = bart_tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = bart_model.generate(
        inputs['input_ids'],
        max_length=350,
        min_length=100,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
def summarize_text_with_t5_base(text):
    inputs = t5_base_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = t5_base_model.generate(
        inputs['input_ids'],
        max_length=128,
        num_beams=4,
        early_stopping=True,
    )
    summary = t5_base_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
def summarize_text_with_fine_tuned_model(text):
    inputs = fine_tuned_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = fine_tuned_model.generate(
        inputs['input_ids'],
        max_length=128,
        num_beams=4,
        early_stopping=True,
    )
    summary = fine_tuned_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
def show_summarizeurl_page():
    st.title("🌐 URL Text Summarization")
    st.write("Enter a URL to get a summary of its content.")
    url = st.text_input("Enter URL")
    model_choice = st.selectbox("Choose the model", ["BART", "T5 Base Model", "Fine-tuned T5 Model", "Use All"])
    if url:
        with st.spinner('🔄 Fetching and processing the URL content...'):
            text = fetch_url_content(url)
        
        if st.button("Summarize"):
            if model_choice == "BART":
                with st.spinner('🔄 Generating summary with BART...'):
                    summary = summarize_text_with_bart(text)
                st.success("✅ Summary generated successfully with BART!")
                st.markdown("### ✨ Summary (BART):")
                st.write(summary)
            elif model_choice == "T5 Base Model":
                with st.spinner('🔄 Generating summary with T5 Base Model...'):
                    summary = summarize_text_with_t5_base(text)
                st.success("✅ Summary generated successfully with T5 Base Model!")
                st.markdown("### ✨ Summary (T5 Base Model):")
                st.write(summary)
            elif model_choice == "Fine-tuned T5 Model":
                with st.spinner('🔄 Generating summary with Fine-tuned T5 Model...'):
                    summary = summarize_text_with_fine_tuned_model(text)
                st.success("✅ Summary generated successfully with Fine-tuned T5 Model!")
                st.markdown("### ✨ Summary (Fine-tuned T5 Model):")
                st.write(summary)
            elif model_choice == "Use All":
                with st.spinner('🔄 Generating summaries with all models...'):
                    summary_bart = summarize_text_with_bart(text)
                    summary_t5_base = summarize_text_with_t5_base(text)
                    summary_fine_tuned = summarize_text_with_fine_tuned_model(text)
                st.success("✅ Summaries generated successfully!")
                st.markdown("### ✨ Summary (BART):")
                st.write(summary_bart)
                st.markdown("### ✨ Summary (T5 Base Model):")
                st.write(summary_t5_base)
                st.markdown("### ✨ Summary (Fine-tuned T5 Model):")
                st.write(summary_fine_tuned)
                
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(f"Read the whole story here: [Link]({url})")
if __name__ == "__main__":
    show_summarizeurl_page()

import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import BartTokenizer, BartForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
# Pfade zu den Modellen
bart_model_path = "streamlit_models/bart_model"
t5_model_path = "streamlit_models/t5_base_model"
fine_tuned_model_path = "streamlit_models/fine_tuned_model"
# Laden der Modelle und Tokenizer
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_path)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_path, use_safetensors=True)
t5_base_tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
t5_base_model = T5ForConditionalGeneration.from_pretrained(t5_model_path, use_safetensors=True)
fine_tuned_tokenizer = T5Tokenizer.from_pretrained(fine_tuned_model_path)
fine_tuned_model = T5ForConditionalGeneration.from_pretrained(fine_tuned_model_path, use_safetensors=True)
def fetch_url_content(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            text = ' '.join([para.get_text() for para in paragraphs])
            return text
        else:
            st.error(f"Failed to fetch the URL content. Status code: {response.status_code}")
            return ""
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return ""
def summarize_text_with_bart(text):
    inputs = bart_tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = bart_model.generate(
        inputs['input_ids'],
        max_length=350,
        min_length=100,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
def summarize_text_with_t5_base(text):
    inputs = t5_base_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = t5_base_model.generate(
        inputs['input_ids'],
        max_length=128,
        num_beams=4,
        early_stopping=True,
    )
    summary = t5_base_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
def summarize_text_with_fine_tuned_model(text):
    inputs = fine_tuned_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = fine_tuned_model.generate(
        inputs['input_ids'],
        max_length=128,
        num_beams=4,
        early_stopping=True,
    )
    summary = fine_tuned_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
def show_summarizeurl_page():
    st.title("🌐 URL Text Summarization")
    st.write("Enter a URL to get a summary of its content.")
    url = st.text_input("Enter URL")
    model_choice = st.selectbox("Choose the model", ["BART", "T5 Base Model", "Fine-tuned T5 Model", "Use All"])
    if url:
        with st.spinner('🔄 Fetching and processing the URL content...'):
            text = fetch_url_content(url)
        
        if st.button("Summarize"):
            if model_choice == "BART":
                with st.spinner('🔄 Generating summary with BART...'):
                    summary = summarize_text_with_bart(text)
                st.success("✅ Summary generated successfully with BART!")
                st.markdown("### ✨ Summary (BART):")
                st.write(summary)
            elif model_choice == "T5 Base Model":
                with st.spinner('🔄 Generating summary with T5 Base Model...'):
                    summary = summarize_text_with_t5_base(text)
                st.success("✅ Summary generated successfully with T5 Base Model!")
                st.markdown("### ✨ Summary (T5 Base Model):")
                st.write(summary)
            elif model_choice == "Fine-tuned T5 Model":
                with st.spinner('🔄 Generating summary with Fine-tuned T5 Model...'):
                    summary = summarize_text_with_fine_tuned_model(text)
                st.success("✅ Summary generated successfully with Fine-tuned T5 Model!")
                st.markdown("### ✨ Summary (Fine-tuned T5 Model):")
                st.write(summary)
            elif model_choice == "Use All":
                with st.spinner('🔄 Generating summaries with all models...'):
                    summary_bart = summarize_text_with_bart(text)
                    summary_t5_base = summarize_text_with_t5_base(text)
                    summary_fine_tuned = summarize_text_with_fine_tuned_model(text)
                st.success("✅ Summaries generated successfully!")
                st.markdown("### ✨ Summary (BART):")
                st.write(summary_bart)
                st.markdown("### ✨ Summary (T5 Base Model):")
                st.write(summary_t5_base)
                st.markdown("### ✨ Summary (Fine-tuned T5 Model):")
                st.write(summary_fine_tuned)
                
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(f"Read the whole story here: [Link]({url})")
if __name__ == "__main__":
    show_summarizeurl_page()
