import torch
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np # linear algebra


def predict(model, encoded_inputs):
    # Chuyển dữ liệu vào device (GPU nếu có)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    encoded_inputs = encoded_inputs.to(device)

    # Dự đoán với mô hình
    with torch.no_grad():
        outputs = model(**encoded_inputs)

    # Lấy kết quả dự đoán
    return outputs

