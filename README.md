# 📰 News Summarizer for PDFs and URLs
## 🏁 1) Project Goal and Motivation

In an era where information overload is a significant challenge, individuals and professionals alike struggle to keep up with the vast amounts of news published daily. The problem is not the lack of information but rather the overwhelming abundance of it, making it difficult to stay informed efficiently. This project aims to solve the problem of information overload by providing a tool that can summarize news articles from PDFs and URLs quickly and accurately.

### Motivation

The motivation behind this project stems from the need for an efficient way to digest news. In today’s fast-paced world, people need to stay updated without spending excessive time reading through long articles. Whether it's a busy professional who needs to catch up on the latest industry news, a student researching current events, or a casual reader wanting to stay informed, everyone benefits from concise, accurate summaries.

### Relevance

This project is highly relevant as it addresses a common pain point experienced by many in the digital age. By leveraging advanced Natural Language Processing (NLP) techniques, this news summarizer can distill essential information from lengthy articles, providing users with quick and comprehensive insights. This not only saves time but also ensures that users stay informed about important topics without the burden of sifting through redundant information.

## 🔧 2) Set-up the project
	1. Git clone
	2. Env setzen:
		1. python3 -m venv venv
		2. source venv/bin/activate 
		3. pip install -r requirements.txt
  	3. Streamlit run streamlit_app.py (Dauer ca. 10 Min bis die Streamlit startet.
   <img width="956" alt="image" src="https://github.com/ZHAWZHAWZHAW/ml2/assets/95766456/dbb9586a-01fc-4d75-95e0-109de8994c6a">

## 📊 3) Data
### Data for training the model
The XSum (Extreme Summarization) dataset was chosen for training the T5 model due to its suitability for generating concise and informative summaries. The XSum dataset contains a wide range of articles from the British Broadcasting Corporation (BBC) and is designed specifically for the task of abstractive summarization. Each article in the dataset is paired with a one-sentence summary, making it an ideal choice for training models aimed at creating brief and accurate news summaries.

Here are some key reasons for selecting the XSum dataset:

- **Diversity of Topics:** The dataset includes articles on a wide variety of topics, ensuring that the model can generalize well across different domains.
- **Quality of Summaries:** The one-sentence summaries provided in the XSum dataset are human-written and focus on the key information in the articles, which aligns well with the goal of creating concise and accurate summaries.
- **Size of the Dataset:** With over 200,000 article-summary pairs, the XSum dataset provides a substantial amount of data for training, validation, and testing, which helps in creating a robust and reliable model.
- **Relevance to Real-World Applications:** The format and content of the XSum dataset closely resemble real-world news articles and summaries, making it highly applicable for a news summarization tool.

By using the XSum dataset, the trained T5 model can effectively learn to generate high-quality summaries that capture the essence of news articles, thereby meeting the needs of users who require quick and comprehensive insights.

### Data for testing the application
In the project folder "Test Data," you will find a collection of PDFs and URLs with which you can test my application. Based on this data, I have also conducted the interpretation and validation in the final section.

Of course, you can also upload your own URLs and PDFs and review the results.

## 📈 4) Modeling

### BART model
BART (Bidirectional and Auto-Regressive Transformers) is a sequence-to-sequence model designed for natural language generation tasks. Developed by Facebook AI, BART combines the benefits of bidirectional and autoregressive transformers, making it highly effective for tasks like text summarization, translation, and text generation. BART is particularly strong in tasks that require understanding and generating human-like text because it leverages a denoising autoencoder approach during training, which helps the model learn robust representations of text.

### T5 model
T5 (Text-To-Text Transfer Transformer) is a versatile model developed by Google Research that treats all NLP tasks as a text-to-text problem, allowing for a unified approach to various tasks like translation, summarization, and classification. The T5 base model, a specific variant within the T5 family, is pre-trained on a large corpus using a fill-in-the-blank objective and can be fine-tuned for specific tasks. This model's flexibility and effectiveness stem from its ability to convert every problem into a text generation task, making it highly adaptable and powerful for numerous applications.

### Fine-tuned t5 model
To enhance the performance of the T5 model for summarizing news articles, I fine-tuned it using a subset of the XSum dataset. The process began with initializing the T5 model and tokenizer from the pre-trained "t5-base" model.

Next, I defined a tokenization function to preprocess the input data, ensuring consistency by formatting both the documents and their summaries. I loaded a subset of 1000 training examples and 500 validation examples from the XSum dataset, chosen for its diverse and concise human-written summaries.

I applied the tokenization function to both the training and validation datasets, transforming the raw text into tokenized inputs suitable for the T5 model. The training arguments were set to balance training efficiency and model performance, with parameters such as learning rate, batch size, number of epochs, and evaluation strategy.

The Seq2SeqTrainer was initialized with the model, training arguments, datasets, tokenizer, and an early stopping callback to prevent overfitting. This mechanism monitored performance and halted training if no improvement was observed.

The training process, managed by Seq2SeqTrainer, involved handling the training loop, evaluation, and saving the best model based on evaluation loss. Finally, I saved the fine-tuned model and tokenizer for easy deployment and further use.

By fine-tuning the T5 model with the XSum dataset, I improved its ability to generate concise and accurate summaries, addressing information overload and making it a valuable tool for efficient news consumption.

### Prompt Engineering

## ✅ 5) Interpretation and Validation
### Analyses

### Limitation & Next steps
