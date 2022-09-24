import torch.nn as nn

from torch import Tensor
from transformers import BertModel


class FakeNewsClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.head = nn.Linear(768, 4)

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
