import torch
import torch.nn as nn

from torch import Tensor
from typing import Optional
from transformers import BertModel


class FakeNewsClassifier(nn.Module):

    def __init__(self, input_dim: int = 768, hidden_dim: Optional[int] = None):
        super().__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")

        if hidden_dim is not None:
            self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 4)
            )
        else:
            self.head = nn.Linear(input_dim,  4)

    def forward(self, *args) -> Tensor:
        assert len(args) in [1, 3], 'wrong number of parameters'

        if len(args) == 3:
            out = self.get_embeddings(*args)

            return self.head(out)

        return self.head(*args)

    def get_embeddings(self, input_ids, segments, attention_masks) -> Tensor:
        embeddings = self.bert(input_ids, token_type_ids=segments, attention_mask=attention_masks)
        embeddings = embeddings.last_hidden_state[:, 0, :]

        return embeddings

    def load_head(self, checkpoint_path: str):
        self.head = torch.load(checkpoint_path)
