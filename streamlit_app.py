import streamlit as st
import logging

# Set logging level to suppress the special tokens warning
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

from streamlit_pages.summarizepdf_page import show_summarizepdf_page
from streamlit_pages.model_info_page import show_model_info_page
from streamlit_pages.home_page import show_home_page
from streamlit_pages.summarizeurl_page import show_summarizeurl_page

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
page = st.sidebar.radio("Go to", ["Home", "Summarize PDFs","Summarize URLs", "Model Info"])

if page == "Home":
    show_home_page()
elif page == "Summarize PDFs":
    show_summarizepdf_page()
elif page == "Summarize URLs":
    show_summarizeurl_page()
elif page == "Model Info":
    show_model_info_page()

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("FS 2024: ML2 Project, Linus Schneeberger", unsafe_allow_html=True)