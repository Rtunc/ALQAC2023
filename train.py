from torch.utils.data import DataLoader
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from src.datasets.preprocess import repairdata
from src.datasets.dataset import AlqacDataset
from.src.utils import compute_metrics
from src.model.model import BertModel
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
import json

with open(', 'r') as file:
    # Load the contents of the file
    public_test = json.load(file)
with open('D:\Workspace\ALQAC2023\data\ALQAC_2023_training_data\law.json', 'r') as file:
    # Load the contents of the file
    law = json.load(file)
    
with open('D:\Workspace\ALQAC2023\data\savedata.json', 'r') as file:
    # Load the contents of the file
    public_test_data_preprocess = json.load(file)
data = pd.read_csv("/kaggle/input/dataalqac/datasALQAC.csv")


train_df, eval_df = train_test_split(data, test_size= 0.2, random_state =42)

train_sen1, train_sen2, train_labels = repairdata(train_df)
val_sen1, val_sen2, val_labels = repairdata(eval_df)

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

train_inputs = tokenizer(train_sen1, train_sen2 , return_tensors = 'pt', max_length = 512, truncation='longest_first', padding = 'max_length')
val_inputs = tokenizer(val_sen1, val_sen2 , return_tensors = 'pt', max_length = 512, truncation='longest_first', padding = 'max_length')
train_inputs['labels'] = torch.LongTensor([train_labels]).T
val_inputs['labels'] = torch.LongTensor([val_labels]).T

train_data = AlqacDataset(train_inputs)
val_data = AlqacDataset(val_inputs)

model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')


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