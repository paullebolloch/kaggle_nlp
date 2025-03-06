import time
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments, pipeline)
from utils import (load_data, split_data, prepare_datasets, tokenize_function,
                   encode_labels, compute_metrics, get_data_collator)


DATA_FILE = "data/train_submission.csv"
texts, labels = load_data(DATA_FILE)

# Séparation des ensembles d'entraînement, validation et test
train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = split_data(texts, labels)


dataset_train, dataset_val, dataset_test = prepare_datasets(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels)
MODEL_NAME = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Tokenization
tokenize_text = tokenize_function(tokenizer)
tokenized_train = dataset_train.map(tokenize_text, batched=True)
tokenized_val = dataset_val.map(tokenize_text, batched=True)
tokenized_test = dataset_test.map(tokenize_text, batched=True)

# Mapping des labels
unique_labels = list(set(labels))
id_to_label = {i: unique_labels[i] for i in range(len(unique_labels))}
label_to_id = {v: k for k, v in id_to_label.items()}

# Encodage des labels
tokenized_train = encode_labels(tokenized_train, label_to_id)
tokenized_val = encode_labels(tokenized_val, label_to_id)
tokenized_test = encode_labels(tokenized_test, label_to_id)

data_collator = get_data_collator(tokenizer)


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(unique_labels), id2label=id_to_label, label2id=label_to_id
)


training_args = TrainingArguments(
    output_dir="models/xlm-roberta-finetuned",
    num_train_epochs=10,
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=128,
    evaluation_strategy="epoch",
    logging_steps=len(tokenized_train) // 64,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    tokenizer=tokenizer,
)


trainer.train()

device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("text-classification", model="models/xlm-roberta-finetuned", device=device)

start_time = time.perf_counter()
predictions = [res["label"] for res in classifier(test_texts.tolist(), truncation=True, max_length=128)]
print(f"Temps d'inférence : {time.perf_counter() - start_time:.2f} secondes")

from sklearn.metrics import classification_report
print(classification_report(test_labels.tolist(), predictions, digits=3))
