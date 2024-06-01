import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import BartTokenizer, BartForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration

###################################################################################################

# Paths to the models, which were created and trained independently/locally
    # bart_model_path = "pre-trained_model/bart_model"
    # t5_model_path = "fine-tuned_model/t5_base"
    # fine_tuned_model_path = "fine-tuned_model/results/final_model"

# Paths to the models
bart_model_path = "streamlit_models/bart_model"
t5_model_path = "streamlit_models/t5_base_model"
fine_tuned_model_path = "streamlit_models/fine_tuned_model"

# Load models and tokenizers
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_path)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_path, use_safetensors=True)
t5_base_tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
t5_base_model = T5ForConditionalGeneration.from_pretrained(t5_model_path, use_safetensors=True)
fine_tuned_tokenizer = T5Tokenizer.from_pretrained(fine_tuned_model_path)
fine_tuned_model = T5ForConditionalGeneration.from_pretrained(fine_tuned_model_path, use_safetensors=True)

# Function to fetch text content from a URL
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
###################################################################################################

# Prompt-Engineering
###################################################################################################
def summarize_text_with_combined_prompts_bart(text):
    prompt1 = f"Summarize the following text in detail: {text}"
    prompt2 = f"Extract and summarize the most important points from the text: {text}"
    
    inputs1 = bart_tokenizer(prompt1, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids1 = bart_model.generate(
        inputs1['input_ids'],
        max_length=50,  
        min_length=25,  
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    summary1 = bart_tokenizer.decode(summary_ids1[0], skip_special_tokens=True)
    
    inputs2 = bart_tokenizer(prompt2, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids2 = bart_model.generate(
        inputs2['input_ids'],
        max_length=50,  
        min_length=25, 
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    summary2 = bart_tokenizer.decode(summary_ids2[0], skip_special_tokens=True)
    
    combined_summary = summary1 + " " + summary2
    return combined_summary

def summarize_text_with_combined_prompts_t5_base(text):
    prompt1 = f"Summarize the following text in detail: {text}"
    prompt2 = f"Extract and summarize the most important points from the text: {text}"
    
    inputs1 = t5_base_tokenizer(prompt1, return_tensors="pt", max_length=512, truncation=True)
    summary_ids1 = t5_base_model.generate(
        inputs1['input_ids'],
        max_length=50, 
        min_length=25, 
        num_beams=4,
        early_stopping=True,
    )
    summary1 = t5_base_tokenizer.decode(summary_ids1[0], skip_special_tokens=True)
    
    inputs2 = t5_base_tokenizer(prompt2, return_tensors="pt", max_length=512, truncation=True)
    summary_ids2 = t5_base_model.generate(
        inputs2['input_ids'],
        max_length=50,
        min_length=25,
        num_beams=4,
        early_stopping=True,
    )
    summary2 = t5_base_tokenizer.decode(summary_ids2[0], skip_special_tokens=True)
    
    combined_summary = summary1 + " " + summary2
    return combined_summary

def summarize_text_with_combined_prompts_fine_tuned(text):
    prompt1 = f"Summarize the following text in detail: {text}"
    prompt2 = f"Extract and summarize the most important points from the text: {text}"
    
    inputs1 = fine_tuned_tokenizer(prompt1, return_tensors="pt", max_length=512, truncation=True)
    summary_ids1 = fine_tuned_model.generate(
        inputs1['input_ids'],
        max_length=50,
        min_length=25,
        num_beams=4,
        early_stopping=True,
    )
    summary1 = fine_tuned_tokenizer.decode(summary_ids1[0], skip_special_tokens=True)
    
    inputs2 = fine_tuned_tokenizer(prompt2, return_tensors="pt", max_length=512, truncation=True)
    summary_ids2 = fine_tuned_model.generate(
        inputs2['input_ids'],
        max_length=50, 
        min_length=25,  
        num_beams=4,
        early_stopping=True,
    )
    summary2 = fine_tuned_tokenizer.decode(summary_ids2[0], skip_special_tokens=True)
    
    combined_summary = summary1 + " " + summary2
    return combined_summary
###################################################################################################

# Streamlit app for summarizing text from URLs
###################################################################################################
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
                    summary = summarize_text_with_combined_prompts_bart(text)
                st.success("✅ Summary generated successfully with BART!")
                st.markdown("### ✨ Summary (BART):")
                st.write(summary)
            elif model_choice == "T5 Base Model":
                with st.spinner('🔄 Generating summary with T5 Base Model...'):
                    summary = summarize_text_with_combined_prompts_t5_base(text)
                st.success("✅ Summary generated successfully with T5 Base Model!")
                st.markdown("### ✨ Summary (T5 Base Model):")
                st.write(summary)
            elif model_choice == "Fine-tuned T5 Model":
                with st.spinner('🔄 Generating summary with Fine-tuned T5 Model...'):
                    summary = summarize_text_with_combined_prompts_fine_tuned(text)
                st.success("✅ Summary generated successfully with Fine-tuned T5 Model!")
                st.markdown("### ✨ Summary (Fine-tuned T5 Model):")
                st.write(summary)
            elif model_choice == "Use All":
                with st.spinner('🔄 Generating summaries with all models...'):
                    summary_bart = summarize_text_with_combined_prompts_bart(text)
                    summary_t5_base = summarize_text_with_combined_prompts_t5_base(text)
                    summary_fine_tuned = summarize_text_with_combined_prompts_fine_tuned(text)
                st.success("✅ Summaries generated successfully!")
                st.markdown("### ✨ Summary (BART):")
                st.write(summary_bart)
                st.markdown("### ✨ Summary (T5 Base Model):")
                st.write(summary_t5_base)
                st.markdown("### ✨ Summary (Fine-tuned T5 Model):")
                st.write(summary_fine_tuned)
                
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(f"Read the whole story here: [Link]({url})")
###################################################################################################

if __name__ == "__main__":
    show_summarizeurl_page()
