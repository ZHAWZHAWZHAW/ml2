from transformers import BartTokenizer, BartForConditionalGeneration

def download_and_save_model(model_name, save_directory):
    # Load tokenizer and model
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    # Save tokenizer and model to save_directory
    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)
    
    print(f"Model and tokenizer saved to {save_directory}.")

if __name__ == "__main__":
    model_name = "facebook/bart-large-cnn"  # The BART model specifically for summarization
    save_directory = "pre-trained_model/bart_model"  # Adjust the path where the model should be saved
    download_and_save_model(model_name, save_directory)
