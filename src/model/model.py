import torch
from torch import nn
from transformers import BertForSequenceClassification

class BertModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert_encoder = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased").to(self.device)
    def forward(self, input_ids, attention_mask, token_type_ids):
        sco = self.bert_encoder(input_ids = input_ids, attention_mask= attention_mask, token_type_ids=token_type_ids)
        return sco.logits



