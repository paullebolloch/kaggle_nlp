import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, DataCollatorWithPadding


def load_data(file_path):
    """Charge les données depuis un fichier CSV et supprime les valeurs nulles."""
    df = pd.read_csv(file_path, encoding="utf-8").dropna()
    return df["Text"], df["Label"]


def split_data(texts, labels, test_size=0.2, val_size=0.5, random_seed=42):
    """Divise les données en ensembles d'entraînement, validation et test."""
    from sklearn.model_selection import train_test_split

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=random_seed
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=val_size, random_state=random_seed
    )

    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


def prepare_datasets(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels):
    """Convertit les textes et labels en Dataset Hugging Face."""
    dataset_train = Dataset.from_dict({"text": train_texts, "labels": train_labels})
    dataset_val = Dataset.from_dict({"text": val_texts, "labels": val_labels})
    dataset_test = Dataset.from_dict({"text": test_texts, "labels": test_labels})

    return dataset_train, dataset_val, dataset_test


def tokenize_function(tokenizer):
    """Retourne une fonction pour tokenizer les textes."""
    def tokenize_text(sample):
        return tokenizer(sample["text"], truncation=True, max_length=128)
    return tokenize_text


def encode_labels(dataset, label_to_id):
    """Encode les labels sous forme d'entiers."""
    def label_encoding(entry):
        entry["labels"] = label_to_id[entry["labels"]]
        return entry

    return dataset.map(label_encoding, batched=False)


def compute_metrics(predictions):
    """Calcule les métriques d'évaluation."""
    true_labels = predictions.label_ids
    predicted_labels = predictions.predictions.argmax(-1)
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average="weighted")
    return {"accuracy": accuracy, "f1": f1}


def get_data_collator(tokenizer):
    """Retourne un data collator avec padding automatique."""
    return DataCollatorWithPadding(tokenizer=tokenizer)

