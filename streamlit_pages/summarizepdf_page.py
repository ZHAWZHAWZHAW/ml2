
import streamlit as st
import fitz  # PyMuPDF for PDF processing
from transformers import BartTokenizer, BartForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration

###################################################################################################

# Pfade zu den Modellen, die eigenständig / lokal erstellt & trainiert wurden
#bart_model_path = "pre-trained_model/bart_model"
#t5_model_path = "fine-tuned_model/t5_base"
#fine_tuned_model_path = "fine-tuned_model/results/final_model"

# Pfade zu den Modellen
bart_model_path = "streamlit_models/bart_model"
t5_model_path = "streamlit_models/t5_base_model"
fine_tuned_model_path = "streamlit_models/fine_tuned_model"

###################################################################################################

# Laden der Modelle und Tokenizer
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_path)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_path, use_safetensors=True)
t5_base_tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
t5_base_model = T5ForConditionalGeneration.from_pretrained(t5_model_path, use_safetensors=True)
fine_tuned_tokenizer = T5Tokenizer.from_pretrained(fine_tuned_model_path)
fine_tuned_model = T5ForConditionalGeneration.from_pretrained(fine_tuned_model_path, use_safetensors=True)
def read_pdf(uploaded_file):
    if uploaded_file is not None:
        file_stream = uploaded_file.read()
        document = fitz.open(stream=file_stream, filetype="pdf")
        text = ""
        for page in document:
            text += page.get_text()
        document.close()
        return text
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
def show_summarizepdf_page():
    st.title("📄 PDF Text Summarization")
    st.write("Upload a PDF file to get a summary.")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    model_choice = st.selectbox("Choose the model", ["BART", "T5 Base Model", "Fine-tuned T5 Model", "Use All"])
    if uploaded_file:
        with st.spinner('🔄 Reading and processing the PDF...'):
            text = read_pdf(uploaded_file)
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
if __name__ == "__main__":
    show_summarizepdf_page()
