from datasets import load_dataset
import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset
import pandas as pd
class TextDataset(Dataset):
    def __init__(self, max_context_length_abstract, max_context_lenght_article, split):
        path1="dataset/Hilichurl_Eng - Sheet1.csv"
        dataset=pd.read_csv(path1)
        #dataset = load_dataset("scientific_papers", 'arxiv', cache_dir=path1)[split]
        self.data = dataset
        json_path1="src/notebooks/tokenizer.json"
        self.tokenizer = Tokenizer.from_file(json_path1)
        self.max_context_length_abstract = max_context_length_abstract
        self.max_context_length_article = max_context_lenght_article
        self.length = len(self.data)
        self.split = split

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        #print(idx)
        abstract = self.data.iloc[idx]["Hilichurl"]
        article = self.data.iloc[idx]["English"]
        #tokenize
        abstract = self.tokenizer.encode(abstract).ids
        article = self.tokenizer.encode(article).ids
        #print(abstract,article)
        #pad
        if len(abstract) > self.max_context_length_abstract:
            abstract = abstract[:self.max_context_length_abstract]
        if len(article) > self.max_context_length_article:
            article = article[:self.max_context_length_article]
        abstract = abstract + [4 for _ in range(self.max_context_length_abstract - len(abstract))]
        if(self.split != 'test'):
            article = article + [4 for _ in range(self.max_context_length_article - len(article))]
        x = abstract[:-1]
        y = abstract[1:]
        article = torch.tensor(article, dtype = torch.int64)
        x = torch.tensor(x, dtype = torch.int64)
        y = torch.tensor(y, dtype = torch.int64)
        return x, y, article