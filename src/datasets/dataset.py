import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class AlqacDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return self.encodings.input_ids.shape[0]
    def __getitem__(self, idx):
        return {key:tensor[idx] for key, tensor in self.encodings.items()}
        