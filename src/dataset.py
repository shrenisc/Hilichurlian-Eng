import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset
import pandas as pd

class TextDataset(Dataset):
    def __init__(self, engContextLength, hilliContextLength):
        self.data = pd.read_csv("dataset/Hilichurl_Eng -Sheet1.csv")
        self.engTokenizer = Tokenizer.from_file("models/englighTokenizer.json")
        self.hilliTokenizer = Tokenizer.from_file("models/hilliTokenizer.json")
        self.engContextLength = engContextLength
        self.hilliContextLength = hilliContextLength
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        hilliText = self.data.iloc[idx]["Hilichurl"]
        engText = self.data.iloc[idx]["English"]

        hilliText = self.hilliTokenizer.encode(hilliText).ids
        engText = self.engTokenizer.encode(engText).ids
        
        if len(hilliText) > self.hilliContextLength:
            hilliText = hilliText[:self.hilliContextLength]
        if len(engText) > self.engContextLength:
            engText = engText[:self.engContextLength]

        engText = engText + [4 for _ in range(self.engContextLength - len(engText))]
        hilliText = hilliText + [4 for _ in range(self.hilliContextLength - len(hilliText))]

        engText = torch.tensor(engText, dtype = torch.int64)

        x = engText[:-1]
        y = engText[1:]

        hilliText = torch.tensor(hilliText, dtype = torch.int64)
        return x, y, hilliText