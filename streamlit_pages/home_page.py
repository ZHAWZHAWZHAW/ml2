import streamlit as st

def show_home_page():
    st.title("📄 PDF & URL Text Summarization")
    st.write("Welcome to the Text Summarization App! Use the navigation menu to access different functionalities.")
    st.markdown("---")
    with st.container():
        st.header("📚 General Information")
        st.write(
            """
            This application uses Facebook's BART model, a custom-trained T5 model, and a fine-tuned T5 model for text summarization.
            You can upload a PDF file or enter a URL to get a concise summary within seconds.
            
            **Features:**
            - Fast and accurate summaries
            - Supports large PDF files up to 200MB
            - User-friendly interface
            - URL content summarization
            
            **How it works:**
            1. Navigate to the 'Summarize PDF' page to upload a PDF.
            2. Navigate to the 'Summarize URL' page to enter a URL.
            3. Choose a summarization model.
            4. Click 'Summarize' to get the summary.
            
            **Models Available:**
            - BART
            - T5 Base Model
            - Fine-tuned T5 Model
            """
        )

if __name__ == "__main__":
    show_home_page()
