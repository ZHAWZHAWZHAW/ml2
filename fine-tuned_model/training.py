import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
from datasets import load_from_disk

# Modell und Tokenizer initialisieren
model_directory = "fine-tuned_model/t5_base"
tokenizer = T5Tokenizer.from_pretrained(model_directory)
model = T5ForConditionalGeneration.from_pretrained(model_directory)

# Tokenisierungsfunktion
def tokenize_function(examples):
    inputs = [doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")  # Verkürzte Eingabelänge

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=64, truncation=True, padding="max_length")  # Verkürzte Ziel-Länge

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Laden der gespeicherten Datensätze
data_folder = 'fine-tuned_model/data'
train_dataset = load_from_disk(os.path.join(data_folder, 'xsum_train')).shuffle(seed=42).select(range(1000))  # Verwende nur 1000 Trainingsbeispiele
eval_dataset = load_from_disk(os.path.join(data_folder, 'xsum_valid')).shuffle(seed=42).select(range(500))  # Verwende nur 500 Evaluationsbeispiele

# Datensätze tokenisieren
train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Trainingsargumente
training_args = Seq2SeqTrainingArguments(
    output_dir="fine-tuned_model/results",
    evaluation_strategy="epoch",  # Evaluation nach jeder Epoche
    learning_rate=2e-5,
    per_device_train_batch_size=1,  # Kleinere Batch-Größe
    per_device_eval_batch_size=1,  # Kleinere Batch-Größe
    weight_decay=0.01,
    save_total_limit=1,  # Begrenze die Anzahl der gespeicherten Modelle
    num_train_epochs=3,  # Reduzierte Anzahl von Epochen
    predict_with_generate=True,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="epoch",  # Speichert nur am Ende jeder Epoche
    load_best_model_at_end=True,  # Aktiviert das Laden des besten Modells am Ende
    metric_for_best_model="eval_loss"
)

# Trainer initialisieren
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]  # Early Stopping nach 1 Epoche ohne Verbesserung
)

# Modell trainieren
trainer.train()

# Letztes Modell speichern
final_model_dir = os.path.join("fine-tuned_model/results", "final_model")
os.makedirs(final_model_dir, exist_ok=True)
model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)
print(f"Model and tokenizer saved to {final_model_dir}")