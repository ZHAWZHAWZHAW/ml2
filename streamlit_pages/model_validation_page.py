import streamlit as st

def show_model_validation_page():
    st.title("📊 Model Validation")

    st.markdown("""
    ### 🧠 Model Performance Comparison

    | Metric       | BART Model        | T5 Base Model    | Fine-tuned T5 Model |
    |--------------|-------------------|------------------|---------------------|
    | **ROUGE-1**  | 0.2011            | 0.1857           | 0.3070              |
    | **ROUGE-2**  | 0.0380            | 0.0358           | 0.0999              |
    | **ROUGE-L**  | 0.1338            | 0.1318           | 0.2363              |
    | **ROUGE-Lsum** | 0.1336          | 0.1316           | 0.2363              |
    | **Latency**  | 11.61 seconds     | 3.92 seconds     | 1.99 seconds        |
    | **Throughput** | 0.086 texts/sec | 0.255 texts/sec  | 0.503 texts/sec     |
    """)

    with st.expander("Explanation of Metrics and Scores"):
        st.markdown("""
        <div style="background-color: #f9f9f9; border-left: 6px solid #ddd; padding: 10px; font-size: 0.9em;">
        <b>Explanation of Metrics and Scores</b><br><br>
        <b>⚙️ ROUGE Scores</b><br>
        ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a set of metrics used to evaluate the quality of summaries by comparing them to reference summaries (typically human-generated). The main ROUGE metrics used are:
        <ul>
            <li><b>ROUGE-1:</b> Measures the overlap of unigrams (individual words) between the generated summary and the reference summary. It provides an indication of how many words from the reference summary are present in the generated summary.</li>
            <li><b>ROUGE-2:</b> Measures the overlap of bigrams (pairs of consecutive words) between the generated summary and the reference summary. This metric captures the ability of the summary to retain the sequence of words and hence the coherence of the text.</li>
            <li><b>ROUGE-L:</b> Measures the longest common subsequence (LCS) between the generated summary and the reference summary. It considers the order of the words and provides a more flexible measure of overlap, considering both precision and recall.</li>
            <li><b>ROUGE-Lsum:</b> A variant of ROUGE-L specifically designed for summarization tasks, focusing on the longest common subsequence at the summary level.</li>
        </ul>
        Higher ROUGE scores indicate that the generated summary has a higher overlap with the reference summary, suggesting better summarization quality.<br><br>

        <b>⚙️ Latency</b><br>
        <ul>
            <li><b>Latency:</b> Measures the average time taken by the model to generate a summary for each input text. It is calculated as the total time taken divided by the number of texts processed.</li>
            <li><b>Lower latency:</b> Indicates that the model can generate summaries quickly, making it suitable for real-time or high-throughput applications.</li>
        </ul><br>

        <b>⚙️ Throughput</b><br>
        <ul>
            <li><b>Throughput:</b> Measures the number of texts the model can process in a given time period, typically expressed as texts per second. It is calculated as the number of texts processed divided by the total time taken.</li>
            <li><b>Higher throughput:</b> Indicates that the model can handle a large number of texts efficiently in a given time frame, which is important for scalability.</li>
        </ul>
        These metrics together provide a comprehensive view of the model's performance, balancing between the quality of the summaries (ROUGE scores) and the efficiency of the model (latency and throughput).
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    ### 🔍 Interpretation and Analysis
    #### ⚙️ ROUGE Scores

    The Fine-tuned T5 Model outperforms both the BART and T5 Base models across all ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum). This indicates that the Fine-tuned T5 Model generates more accurate and relevant summaries compared to the other models.

    #### ⚙️ Latency

    The BART Model has the highest latency, taking about 11.61 seconds per text, which is significantly slower compared to the T5 models. The Fine-tuned T5 Model has the lowest latency at 1.99 seconds per text, indicating that it processes texts faster than the other models.

    #### ⚙️ Throughput

    The Fine-tuned T5 Model has the highest throughput at 0.503 texts per second, meaning it can handle more texts in a given time frame compared to the other models. The BART Model has the lowest throughput at 0.086 texts per second, making it the least efficient in terms of processing speed.

    #### ⚙️ Conclusion

    The Fine-tuned T5 Model not only generates higher quality summaries as evidenced by the higher ROUGE scores but also processes texts more efficiently with lower latency and higher throughput. This makes it the best-performing model among the three for the task of summarizing news articles.
    """)

if __name__ == "__main__":
    show_model_validation_page()
