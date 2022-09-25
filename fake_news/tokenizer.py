import torch
import pandas as pd

from tqdm.auto import tqdm
from torch.utils.data import TensorDataset
from torch.nn.utils.rnn import pad_sequence

from common.tokenizer import CommonTokenizer


class FakeNewsTokenizer(CommonTokenizer):

    def __init__(self,):
        super().__init__()
        self.text_dicts = {'head': {}, 'body': {}}

    def get_bert_input(self, pairs: pd.DataFrame, verbose: bool = False) -> TensorDataset:
        assert list(pairs.columns) == ['Headline ID', 'Body ID'], 'wrong input format'
        assert all([head_id in self.text_dicts['head'] for head_id in pairs['Headline ID']]), 'unknown Headline ID in pairs'
        assert all([body_id in self.text_dicts['body'] for body_id in pairs['Body ID']]), 'unknown Body ID in pairs'

        input_ids = []
        segments = []
        attention_masks = []

        iterator = list(pairs.itertuples(index=False))

        for row in tqdm(iterator, disable=not verbose, desc="BERT input preparation"):
            head_id, body_id = row

            head_input_ids = self.text_dicts['head'][head_id]['input_ids']
            body_input_ids = self.text_dicts['body'][body_id]['input_ids']

            body_len = min(len(body_input_ids), 512 - len(head_input_ids) - 3)
            head_len = len(head_input_ids)

            body_input_ids = body_input_ids[:body_len]

            input_id = [self.CLS, *head_input_ids, self.SEP, *body_input_ids, self.SEP]
            segment = [0] * (head_len + 2) + [1] * (body_len + 1)
            attention_mask = [1] * (body_len + head_len + 3)

            input_ids += [torch.tensor(input_id)]
            segments += [torch.tensor(segment)]
            attention_masks += [torch.tensor(attention_mask)]

        input_ids = pad_sequence(input_ids, batch_first=True)
        segments = pad_sequence(segments, batch_first=True)
        attention_masks = pad_sequence(attention_masks, batch_first=True)

        dataset = TensorDataset(input_ids, segments, attention_masks)

        return dataset
