import torch
from torch.utils.data import Dataset


class AutoDataset(Dataset):
    def __init__(self):
        super(AutoDataset, self).__init__()

    def encode_datas(self, tokenizer, sentences, targets):
        self.encodings = tokenizer(targets, sentences, padding=True, truncation='only_second', return_tensors='pt')
        self.encodings['classification_label'] = torch.tensor(self.labels)

    def __getitem__(self, item):
        return {k: v[item] for k, v in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])