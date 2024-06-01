# 📰 News Summarizer for PDFs and URLs

## Table of Contents
1. [🏁 Project Goal & Motivation](https://github.com/ZHAWZHAWZHAW/ml2/blob/master/README.md#-1-project-goal--motivation)
   - [1.1 Motivation](#11-motivation)
   - [1.2 Relevance](#12-relevance)
2. [🔧 Set-up the Project](https://github.com/ZHAWZHAWZHAW/ml2/blob/master/README.md#-2-set-up-the-project)
3. [📊 Data](https://github.com/ZHAWZHAWZHAW/ml2/blob/master/README.md#-3-data)
   - [3.1 Data for Training the Model](#31-data-for-training-the-model)
   - [3.2 Data for Testing the Application](#32-data-for-testing-the-application)
4. [📈 Modeling](https://github.com/ZHAWZHAWZHAW/ml2/blob/master/README.md#-4-modeling)
   - [4.1 BART Model](#41-bart-model)
   - [4.2 T5 Model](#42-t5-model)
   - [4.3 Fine-tuned T5 Model](#43-fine-tuned-t5-model)
   - [4.4 Prompt Engineering](#44-prompt-engineering)
5. [✅ Interpretation and Validation](https://github.com/ZHAWZHAWZHAW/ml2/blob/master/README.md#-5-interpretation-and-validation)
   - [5.1 Analysis](#51-analysis)
   - [5.1.1 Model Performance Comparison](#511-model-comparison)
   - [5.2 Personal reflection](#52-personal-reflection)
   - [5.3 Training Analyses](#53-training-analysis)
   - [5.4 Limitation & Next Steps](#54-limitation--next-steps)

## 🏁 Project Goal & Motivation <a name="-1-project-goal--motivation"></a>
In an era where information overload is a significant challenge. The issue isn't that there's too little information, but rather too much of it, making it hard to stay well-informed. This project aims to solve the problem of information overload by providing a tool that can summarize news articles from PDFs and URLs quickly and accurately.

### 1.1 Motivation <a name="11-motivation"></a>
The reason for this project is to make it easier to keep up with the news. In today's busy world, people need to stay informed without spending too much time reading long articles. Whether you're a busy professional, a student, or just someone who wants to stay updated, everyone can benefit from quick and accurate summaries.

### 1.2 Relevance <a name="12-relevance"></a>
This project is really useful because it tackles a common problem people face today: information overload. By using advanced Natural Language Processing (NLP) techniques, this news summarizer can pull out the key information from long articles, giving users quick and clear summaries. This helps save time and makes sure people stay informed about important topics without having to read through a lot of unnecessary details.

## 🔧 Set-up the Project <a name="-2-set-up-the-project"></a>

### ⚠️⚠️⚠️ Important Information:
I've saved the three models used in Google Drive. As soon as you start the Streamlit app (command: streamlit run streamlit_app.py), the application will download the three models and load them into the "streamlit_models" folder. The Streamlit app will then start and use the newly downloaded models.

-> So, you only need to start the Streamlit app to get the three models and have the application running.

Of course, you can also generate and train the models yourself by using the code in the "fine-tuned_model" and "pre-trained_model" folders.

### 📋 Step-by-step:
1. Git clone
2. Set up Environment:
	1. python3 -m venv venv
	2. source venv/bin/activate 
	3. pip install -r requirements.txt
3. Streamlit run streamlit_app.py (it takes a few minutes to get the models from the Google Drive)



Link to the Google Drive: https://drive.google.com/drive/folders/1eKp7-epSM1sTqh1SuVeMMdkz19c8k364?usp=sharing



   The code downloads the created models from Google Drive so that you can test the application even if you cannot run the models locally. The models will be downloaded to the 'streamlit-models' folder.	

   Download:
   
   <img width="700" alt="image" src="https://github.com/ZHAWZHAWZHAW/ml2/assets/95766456/dbb9586a-01fc-4d75-95e0-109de8994c6a">

   Folder streamlit-models:
   
   <img width="164" alt="image" src="https://github.com/ZHAWZHAWZHAW/ml2/assets/95766456/848fc0d3-a51a-4a0c-bab4-0761342b1513">


### ℹ️ Screenshots of the running Application

<details>
  <summary>Click to see screenshots</summary>
  
  <table>
    <tr>
      <td><img src="https://github.com/ZHAWZHAWZHAW/ml2/assets/95766456/99c2cb02-eaf5-41f5-8168-267c659f18fd" width="400"/></td>
      <td><img src="https://github.com/ZHAWZHAWZHAW/ml2/assets/95766456/3bfb7804-6301-48de-8b77-050067faed54" width="400"/></td>
    </tr>
    <tr>
      <td><img src="https://github.com/ZHAWZHAWZHAW/ml2/assets/95766456/7003051c-c06f-40ff-93f7-4f9ed6a7e4d2" width="400"/></td>
      <td><img src="https://github.com/ZHAWZHAWZHAW/ml2/assets/95766456/7453ecf2-94d6-4838-8111-891e0db0179a" width="400"/></td>
    </tr>
    <tr>
      <td><img src="https://github.com/ZHAWZHAWZHAW/ml2/assets/95766456/616f7ac0-6a50-4197-b12a-90c7bfc3b01e" width="400"/></td>
      <td><img src="https://github.com/ZHAWZHAWZHAW/ml2/assets/95766456/f864e71c-da2d-477c-ab94-a54b2362a581" width="400"/></td>
    </tr>
    <tr>
      <td><img src="https://github.com/ZHAWZHAWZHAW/ml2/assets/95766456/3ffe5d66-b0db-45f9-b836-dd212b74f269" width="400"/></td>
      <td><img src="https://github.com/ZHAWZHAWZHAW/ml2/assets/95766456/3e3fdc33-a9ee-45b8-b365-87fe69a6c578" width="400"/></td>
    </tr>
    <tr>
      <td colspan="2"><img src="https://github.com/ZHAWZHAWZHAW/ml2/assets/95766456/e9b73106-ebe0-4490-87d2-66926d08e56b" width="400"/></td>
    </tr>
  </table>
</details>


## 📊 Data <a name="-3-data"></a>
### 3.1 Data for Training the Model <a name="31-data-for-training-the-model"></a>
The XSum (Extreme Summarization) dataset was chosen for training the T5 model due to its suitability for generating concise and informative summaries. The XSum dataset contains a wide range of articles from the British Broadcasting Corporation (BBC) and is designed specifically for the task of abstractive summarization. Each article in the dataset is paired with a one-sentence summary, making it an ideal choice for training models aimed at creating brief and accurate news summaries.

Here are some key reasons for selecting the XSum dataset:

- **Diversity of Topics:** The dataset includes articles on a wide variety of topics, ensuring that the model can generalize well across different domains.
- **Quality of Summaries:** The one-sentence summaries provided in the XSum dataset are human-written and focus on the key information in the articles, which aligns well with the goal of creating concise and accurate summaries.
- **Size of the Dataset:** With over 200,000 article-summary pairs, the XSum dataset provides a substantial amount of data for training, validation, and testing, which helps in creating a robust and reliable model.
- **Relevance to Real-World Applications:** The format and content of the XSum dataset closely resemble real-world news articles and summaries, making it highly applicable for a news summarization tool.

By using the XSum dataset, the trained T5 model can effectively learn to generate high-quality summaries that capture the essence of news articles, thereby meeting the needs of users who require quick and comprehensive insights.

### 3.2 Data for Testing the Application <a name="32-data-for-testing-the-application"></a>
In the project folder "Test Data," you will find a collection of PDFs and URLs with which you can test my application. Based on this data, I have also conducted the interpretation and validation in the final section.

Of course, you can also upload your own URLs and PDFs and review the results.

<img width="150" alt="image" src="https://github.com/ZHAWZHAWZHAW/ml2/assets/95766456/5dc6e1dd-c11d-43cb-8d16-cba9af4d70e2">


## 📈 Modeling <a name="-4-modeling"></a>
In this project, I used the BART model and the T5 Base model in their pre-trained forms without making any further modifications. These models were directly loaded from their respective repositories and used for text summarization. Additionally, I fine-tuned the T5 model to further optimize its performance for my specific use cases. By fine-tuning the T5 model, I was able to significantly improve the quality of the generated summaries.

### 4.1 BART Model <a name="41-bart-model"></a>
BART (Bidirectional and Auto-Regressive Transformers) is a sequence-to-sequence model designed for natural language generation tasks. Developed by Facebook AI, BART combines the benefits of bidirectional and autoregressive transformers, making it highly effective for tasks like text summarization, translation, and text generation. BART is particularly strong in tasks that require understanding and generating human-like text because it leverages a denoising autoencoder approach during training, which helps the model learn robust representations of text.

<img width="311" alt="image" src="https://github.com/ZHAWZHAWZHAW/ml2/assets/95766456/9d9f30db-125a-49df-ae51-37ebdf5b9fef">

### 4.2 T5 Model <a name="42-t5-model"></a>
T5 (Text-To-Text Transfer Transformer) is a versatile model developed by Google Research that treats all NLP tasks as a text-to-text problem, allowing for a unified approach to various tasks like translation, summarization, and classification. The T5 base model, a specific variant within the T5 family, is pre-trained on a large corpus using a fill-in-the-blank objective and can be fine-tuned for specific tasks. This model's flexibility and effectiveness stem from its ability to convert every problem into a text generation task, making it highly adaptable and powerful for numerous applications.

<img width="311" alt="image" src="https://github.com/ZHAWZHAWZHAW/ml2/assets/95766456/cf6732c7-cc86-41ee-ab88-982a13b0b858">

### 4.3 Fine-tuned T5 Model <a name="43-fine-tuned-t5-model"></a>
To enhance the performance of the T5 model for summarizing news articles, I fine-tuned it using a subset of the XSum dataset. 
1. The process began with initializing the T5 model and tokenizer from the pre-trained "t5-base" model.
2. Next, I defined a tokenization function to preprocess the input data, ensuring consistency by formatting both the documents and their summaries. I loaded a subset of 1000 training examples and 500 validation examples from the XSum dataset, chosen for its diverse and concise human-written summaries.
3. I applied the tokenization function to both the training and validation datasets, transforming the raw text into tokenized inputs suitable for the T5 model. The training arguments were set to balance training efficiency and model performance, with parameters such as learning rate, batch size, number of epochs, and evaluation strategy.
4. The Seq2SeqTrainer was initialized with the model, training arguments, datasets, tokenizer, and an early stopping callback to prevent overfitting. This mechanism monitored performance and halted training if no improvement was observed.
5. The training process, managed by Seq2SeqTrainer, involved handling the training loop, evaluation, and saving the best model based on evaluation loss. Finally, I saved the fine-tuned model and tokenizer for easy deployment and further use.

By fine-tuning the T5 model with the XSum dataset, I improved its ability to generate concise and accurate summaries, addressing information overload and making it a valuable tool for efficient news consumption.

### 4.4 Prompt Engineering <a name="44-prompt-engineering"></a>
In my news summarizer application, prompt engineering is a key technique used to enhance the performance of the models for summarizing text from PDFs and URLs. Prompt engineering involves designing specific prompts that guide the models to generate high-quality summaries effectively.

#### Summarizing Text from URLs

The approach for URL summarization is similar. First, I fetched the text content from the given URL using `requests` and `BeautifulSoup`. The same prompt structure was used to guide the models in generating summaries. This consistency helps in maintaining the effectiveness of the summaries across different types of input sources.

Example function for BART:
```python
def summarize_text_with_bart(text):
    prompt = f"Summarize the following text in detail: {text}"
    inputs = bart_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = bart_model.generate(inputs['input_ids'], max_length=350, min_length=100, num_beams=4, early_stopping=True)
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
```

Example function for T5 Base:
```python
def summarize_text_with_t5_base(text):
    prompt = f"Summarize the following text in detail: {text}"
    inputs = t5_base_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = t5_base_model.generate(inputs['input_ids'], max_length=128, num_beams=4, early_stopping=True)
    return t5_base_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
```
By carefully crafting these prompts, I was able to ensure that three models effectively summarize text from various sources. Using prompt engineering is key to helping the models create clear and informative summaries. This makes the news summarizer a handy tool for users who need quick insights from a lot of content.

## ✅ Interpretation and Validation <a name="-5-interpretation-and-validation"></a>
### 5.1 Analysis <a name="51-analysis"></a>
To validate the performance of the BART, T5 Base, and Fine-tuned T5 models, I followed a systematic approach:

1. Load Models and Tokenizers: Pre-trained BART and T5 Base models, along with the Fine-tuned T5 model, were loaded with their respective tokenizers.
2. Load Dataset and Metric: A subset of 100 examples from the XSum validation set and the ROUGE metric were loaded for evaluation.
3. Summarize Texts: Each model generated summaries for the input texts from the dataset. The summaries (hypotheses) and reference summaries were collected.
4. Evaluate Performance: The ROUGE scores were computed to measure summary quality. Latency (time per text) and throughput (texts per second) were calculated to assess efficiency.

This process provided a clear comparison of each model's performance in terms of both quality and speed.

<details>
  <summary>Screenshots</summary>
  <summary>Run validation.py</summary>
  <img width="845" alt="image" src="https://github.com/ZHAWZHAWZHAW/ml2/assets/95766456/3c581ece-f888-4bc3-b707-dba5088abb68">

  <summary>Results:</summary>
  <img width="932" alt="image" src="https://github.com/ZHAWZHAWZHAW/ml2/assets/95766456/72a237a7-1493-45f8-b42d-445f306432a3">
</details>

### 5.1.1. 🧠 Model Performance Comparison <a name="511-model-performance-comparison"></a>

| Metric       | BART Model        | T5 Base Model    | Fine-tuned T5 Model |
|--------------|-------------------|------------------|---------------------|
| **ROUGE-1**  | 0.2011            | 0.1857           | 0.3070              |
| **ROUGE-2**  | 0.0380            | 0.0358           | 0.0999              |
| **ROUGE-L**  | 0.1338            | 0.1318           | 0.2363              |
| **ROUGE-Lsum** | 0.1336          | 0.1316           | 0.2363              |
| **Latency**  | 11.61 seconds     | 3.92 seconds     | 1.99 seconds        |
| **Throughput** | 0.086 texts/sec | 0.255 texts/sec  | 0.503 texts/sec     |

<details>
  <summary>Explanation of Metrics and Scores</summary>

  #### ⚙️ ROUGE Scores
  ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a set of metrics used to evaluate the quality of summaries by comparing them to reference summaries (typically human-generated). The main ROUGE metrics used are:
  - **ROUGE-1:** Measures the overlap of unigrams (individual words) between the generated summary and the reference summary. It provides an indication of how many words from the reference summary are present in the generated summary.
  - **ROUGE-2:** Measures the overlap of bigrams (pairs of consecutive words) between the generated summary and the reference summary. This metric captures the ability of the summary to retain the sequence of words and hence the coherence of the text.
  - **ROUGE-L:** Measures the longest common subsequence (LCS) between the generated summary and the reference summary. It considers the order of the words and provides a more flexible measure of overlap, considering both precision and recall.
  - **ROUGE-Lsum:** A variant of ROUGE-L specifically designed for summarization tasks, focusing on the longest common subsequence at the summary level.

  Higher ROUGE scores indicate that the generated summary has a higher overlap with the reference summary, suggesting better summarization quality.

  #### ⚙️ Latency
  - **Latency:** Measures the average time taken by the model to generate a summary for each input text. It is calculated as the total time taken divided by the number of texts processed.
    - **Lower latency** indicates that the model can generate summaries quickly, making it suitable for real-time or high-throughput applications.

  #### ⚙️ Throughput
  - **Throughput:** Measures the number of texts the model can process in a given time period, typically expressed as texts per second. It is calculated as the number of texts processed divided by the total time taken.
    - **Higher throughput** indicates that the model can handle a large number of texts efficiently in a given time frame, which is important for scalability.

  These metrics together provide a comprehensive view of the model's performance, balancing between the quality of the summaries (ROUGE scores) and the efficiency of the model (latency and throughput).
</details>

### 🔍 Interpretation and Analysis
#### ⚙️ ROUGE Scores

The Fine-tuned T5 Model outperforms both the BART and T5 Base models across all ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum). This indicates that the Fine-tuned T5 Model generates more accurate and relevant summaries compared to the other models.

#### ⚙️ Latency

The BART Model has the highest latency, taking about 11.61 seconds per text, which is significantly slower compared to the T5 models. The Fine-tuned T5 Model has the lowest latency at 1.99 seconds per text, indicating that it processes texts faster than the other models.

#### ⚙️ Throughput

The Fine-tuned T5 Model has the highest throughput at 0.503 texts per second, meaning it can handle more texts in a given time frame compared to the other models. The BART Model has the lowest throughput at 0.086 texts per second, making it the least efficient in terms of processing speed.

#### ⚙️ Conclusion

The Fine-tuned T5 Model not only generates higher quality summaries as evidenced by the higher ROUGE scores but also processes texts more efficiently with lower latency and higher throughput. This makes it the best-performing model among the three for the task of summarizing news articles.

### 5.2 📌 Personal reflection <a name="52-personal-reflection"></a>
I found that the summaries generated by the Fine-tuned T5 Model are concise and capture the essential details of the original texts effectively. In both cases, it provided clear and direct summaries, showcasing its ability to understand and distill complex information into digestible summaries.

The BART Model, while detailed, included more context and specifics, which might be useful in some scenarios but could be considered excessive in others. The T5 Base Model, on the other hand, provided more succinct summaries than BART but did not capture as many details as the Fine-tuned T5 Model.

In summary, I believe the Fine-tuned T5 Model excels in generating summaries that are both comprehensive and concise, making it my preferred choice for applications where quick and clear information retrieval is essential. The BART Model's detailed summaries could be beneficial for users needing more context, whereas the T5 Base Model strikes a balance between brevity and detail but does not perform as well as the fine-tuned variant.


| Example 1 | Example 2 |
|-------------------------------|-------------------------------|
| ![Fine-tuned T5 Model Summary 1](https://github.com/ZHAWZHAWZHAW/ml2/assets/95766456/12d39873-1ae0-4a3e-8929-6655614446f1) | ![Fine-tuned T5 Model Summary 2](https://github.com/ZHAWZHAWZHAW/ml2/assets/95766456/49d4002b-f961-40c1-901f-ac3cf1d3b42d) |


### 🛠️ 5.3 Training Analysis <a name="53-training-analysis"></a>
The training of the Fine-Tuned T5 Model was conducted over three epochs using a subset of the XSum dataset. Initially, the model started with a loss of 9.6748, which decreased significantly to 1.3579 by the third epoch. This steady reduction in loss indicates effective learning and optimization.

The evaluation loss was recorded at 1.1472, suggesting reasonable generalization to unseen validation data. Throughout the training, the gradient norms fluctuated as the model adjusted its weights. However, there was a warning regarding missing keys in the checkpoint model, specifically 'embed_tokens.weight' and 'lm_head.weight', which could have slightly impacted performance.

The training speed averaged 3.86 samples per second, which is decent but could be improved for faster training times, especially for larger datasets or more complex models.

In summary, the training of the Fine-Tuned T5 Model was successful, with significant loss reduction and good generalization. Addressing the missing keys warning and optimizing training speed could further enhance performance, making the model a valuable tool for generating high-quality summaries of news articles.

<details>
  <summary>Training Logs Screenshot</summary>
  <summary>Screenshot 1</summary>
  <img width="1013" alt="image" src="https://github.com/ZHAWZHAWZHAW/ml2/assets/95766456/c831c060-e900-4402-8710-db8fc3f57007">
  <summary>Screenshot 2</summary>
  <img width="1015" alt="image" src="https://github.com/ZHAWZHAWZHAW/ml2/assets/95766456/422bd40e-75d8-4eed-af3f-d56f92d779da">
</details>

### 🛑 5.4 Limitation & Next Steps <a name="54-limitation--next-steps"></a>
Even though the Fine-tuned T5 Model showed good results, there are some limitations. The model was trained on a small part of the XSum dataset, so it might not work as well with different texts or topics. We only used the ROUGE metric for evaluation, which doesn't measure everything like readability or accuracy.

To improve, we could train on a larger, more varied dataset and use more evaluation methods. Getting feedback from people reading the summaries could also help. Trying other advanced summarization models might lead to better performance.

By addressing these issues and exploring new options, we can make the news summarization tool even better.
