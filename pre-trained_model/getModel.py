from transformers import BartTokenizer, BartForConditionalGeneration

def download_and_save_model(model_name, save_directory):
    # Tokenizer und Modell laden
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    # Tokenizer und Modell in save_directory speichern
    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)
    
    print(f"Modell und Tokenizer wurden unter {save_directory} gespeichert.")

if __name__ == "__main__":
    model_name = "facebook/bart-large-cnn"  # Das BART-Modell speziell für Zusammenfassungen
    save_directory = "pre-trained_model/bart_model"  # Pfad anpassen, wo das Modell gespeichert werden soll
    download_and_save_model(model_name, save_directory)
