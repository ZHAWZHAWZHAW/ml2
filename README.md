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

## 📈 4) Modeling
### BART model

### T5 model

### Fine-tuned t5 model
#### Training

### Prompt Engineering

## ✅ 5) Interpretation and Validation
### Analyses

### Limitation & Next steps
