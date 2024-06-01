import os
import time
from datasets import load_from_disk
from transformers import BartTokenizer, BartForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
import evaluate
import torch

###################################################################################################

# Paths to the models, which were independently created and trained
    #bart_model_path = "pre-trained_model/bart_model"
    #t5_model_path = "fine-tuned_model/t5_base"
    #fine_tuned_model_path = "fine-tuned_model/results/final_model"

# Define paths to models and dataset
bart_model_path = "streamlit_models/bart_model"
t5_base_model_path = "streamlit_models/t5_base_model"
fine_tuned_model_path = "streamlit_models/fine_tuned_model"
data_folder = "fine-tuned_model/data/xsum_valid"

# Load models and tokenizers
print("Loading models and tokenizers...")
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_path)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_path)
t5_base_tokenizer = T5Tokenizer.from_pretrained(t5_base_model_path)
t5_base_model = T5ForConditionalGeneration.from_pretrained(t5_base_model_path)
fine_tuned_tokenizer = T5Tokenizer.from_pretrained(fine_tuned_model_path)
fine_tuned_model = T5ForConditionalGeneration.from_pretrained(fine_tuned_model_path)
print("Models and tokenizers loaded.")

# Load metric and dataset
print("Loading metric and dataset...")
rouge = evaluate.load("rouge")
dataset = load_from_disk(data_folder).select(range(100))  # Using a subset for quick evaluation
print("Metric and dataset loaded.")

###################################################################################################
# Function to summarize text
def summarize_text(model, tokenizer, text, max_length=512):
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=128, min_length=30, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Function to evaluate model
def evaluate_model(model, tokenizer, dataset):
    print(f"Evaluating model: {model.__class__.__name__}")
    references = []
    hypotheses = []
    start_time = time.time()
    for i, example in enumerate(dataset):
        summary = summarize_text(model, tokenizer, example["document"])
        references.append(example["summary"])
        hypotheses.append(summary)
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(dataset)} examples.")
    end_time = time.time()

    # Compute ROUGE scores
    results = rouge.compute(predictions=hypotheses, references=references)
    latency = (end_time - start_time) / len(dataset)
    throughput = len(dataset) / (end_time - start_time)
    
    return results, latency, throughput

# Evaluate BART model
bart_results, bart_latency, bart_throughput = evaluate_model(bart_model, bart_tokenizer, dataset)

# Evaluate T5 Base model
t5_base_results, t5_base_latency, t5_base_throughput = evaluate_model(t5_base_model, t5_base_tokenizer, dataset)

# Evaluate Fine-tuned T5 model
fine_tuned_results, fine_tuned_latency, fine_tuned_throughput = evaluate_model(fine_tuned_model, fine_tuned_tokenizer, dataset)

###################################################################################################
# Print results
print("BART Model Results:")
print("ROUGE:", bart_results)
print("Latency:", bart_latency, "seconds per text")
print("Throughput:", bart_throughput, "texts per second")

print("\nT5 Base Model Results:")
print("ROUGE:", t5_base_results)
print("Latency:", t5_base_latency, "seconds per text")
print("Throughput:", t5_base_throughput, "texts per second")

print("\nFine-tuned T5 Model Results:")
print("ROUGE:", fine_tuned_results)
print("Latency:", fine_tuned_latency, "seconds per text")
print("Throughput:", fine_tuned_throughput, "texts per second")
