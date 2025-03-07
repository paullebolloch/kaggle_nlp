import pandas as pd
import torch
from transformers import pipeline

test_data = pd.read_csv("test_without_labels.csv", encoding="utf-8")
test_data = test_data.dropna()
sentences = test_data["Text"].tolist()

device = 0 if torch.cuda.is_available() else -1
model_checkpoint = "models/xlm-roberta-finetuned/checkpoint-2771"
model_pipe = pipeline(
    "text-classification", model=model_checkpoint, device=device, batch_size=64
)
labels = model_pipe(sentences, truncation=True, max_length=128)

y_predect = []
for i in range(len(labels)):
    label = labels[i]["label"]
    y_predect.append({"ID": i + 1, "Label": label})

y_predict_df = pd.DataFrame(y_predect)
y_predict_df.to_csv("predict_kaggle.csv", index=False)