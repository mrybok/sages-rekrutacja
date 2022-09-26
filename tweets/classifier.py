import torch
import torch.nn as nn

from torch import Tensor
from transformers import BertModel


class TweetsClassifier(nn.Module):

    def __init__(self, input_dim: int = 772):
        super().__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.head = nn.Sequential(nn.Linear(input_dim, 1), nn.Sigmoid())

    def forward(self, *args) -> Tensor:
        assert 1 <= len(args) <= 3, 'wrong number of parameters'

        if len(args) == 3:
            assert self.head[0].in_features == 772, 'wrong input dimensionality'

            out = self.get_embeddings(*args[:2])
            out = torch.cat((out, args[2]), dim=1)

            return self.head(out)
        elif len(args) == 2:
            assert self.head[0].in_features == 768, 'wrong input dimensionality'

            out = self.get_embeddings(*args)

            return self.head(out)

        return self.head(*args)

    def get_embeddings(self, input_ids, attention_masks) -> Tensor:
        embeddings = self.bert(input_ids, attention_mask=attention_masks)
        embeddings = embeddings.last_hidden_state[:, 0, :]

        return embeddings

    def load_head(self, checkpoint_path: str):
        self.head = torch.load(checkpoint_path)
