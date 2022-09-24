import torch.nn as nn

from transformers import BertModel
from transformers import BertTokenizer


class FakeNewsClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.head = nn.Linear(768, 4)

    def forward(self, x):
        pass

    def get_embeddings(self, save: bool = False):

        pass

    def train_head(self):
        pass
