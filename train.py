from torch.utils.data import DataLoader
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from src.datasets.preprocess import prepairdata
from src.datasets.dataset import AlqacDataset
from src.model.model import BertModel
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import json
import numpy as np # linear algebra

with open('./data/ALQAC_2023_training_data/public_test.json', 'r') as file:
    # Load the contents of the file
    public_test = json.load(file)
with open('./data/ALQAC_2023_training_data/law.json', 'r') as file:

    # Load the contents of the file
    law = json.load(file)

with open('./data/savedata.json', 'r') as file:
    # Load the contents of the file
    public_test_data_preprocess = json.load(file)
data = pd.read_csv("./data/datasALQAC.csv")


train_df, eval_df = train_test_split(data, test_size= 0.2, random_state =42)

train_sen1, train_sen2, train_labels = prepairdata(train_df)
val_sen1, val_sen2, val_labels = prepairdata(eval_df)

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

train_inputs = tokenizer(train_sen1, train_sen2 , return_tensors = 'pt', max_length = 512, truncation='longest_first', padding = 'max_length')
val_inputs = tokenizer(val_sen1, val_sen2 , return_tensors = 'pt', max_length = 512, truncation='longest_first', padding = 'max_length')
train_inputs['labels'] = torch.LongTensor([train_labels]).T
val_inputs['labels'] = torch.LongTensor([val_labels]).T

train_data = AlqacDataset(train_inputs)
val_data = AlqacDataset(val_inputs)

model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')

def compute_metrics(p):    
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred,average='weighted')
    precision = precision_score(y_true=labels, y_pred=pred,average='weighted')
    f1 = f1_score(y_true=labels, y_pred=pred,average='weighted')
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1} 

training_args = TrainingArguments(
    output_dir='./results',  # Directory where the model checkpoints and evaluation results will be stored
    num_train_epochs=6,  # Number of training epochs
    per_device_train_batch_size=16,  # Batch size for training
    per_device_eval_batch_size=64,  # Batch size for evaluation
    logging_dir='./logs',  # Directory where training logs will be stored
    logging_steps=500,  # Log training metrics every X steps
    save_steps=1000,  # Save a checkpoint every X steps
)

trainer = Trainer(
    model=model,
    args = training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)