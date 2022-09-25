import torch
import pandas as pd

from tqdm.auto import tqdm
from torch.utils.data import TensorDataset
from torch.nn.utils.rnn import pad_sequence

from common.tokenizer import CommonTokenizer


class TweetsTokenizer(CommonTokenizer):

    def __init__(self,):
        super().__init__()
        self.text_dicts = {'tweets': {}}

    def get_bert_input(self, tweets: pd.DataFrame, verbose: bool = False) -> TensorDataset:
        assert list(tweets.columns) == ['id', 'sin day', 'cos day' 'sin hour', 'cos hour'], 'wrong input format'
        assert all([tweet_id in self.text_dicts['tweets'] for tweet_id in tweets['id']]), 'unknown Tweet ID'

        input_ids = []
        attention_masks = []
        time_vectors = []

        iterator = list(tweets.itertuples(index=False))

        for row in tqdm(iterator, disable=not verbose, desc="BERT input preparation"):
            tweet_id = row[0]
            time_vector = row[1:]

            tweet_input_ids = self.text_dicts['tweets'][tweet_id]['input_ids']

            tweet_len = min(len(tweet_input_ids), 512 - 2)

            tweet_input_ids = tweet_input_ids[:tweet_len]

            input_id = [self.CLS, *tweet_input_ids, self.SEP]
            attention_mask = [1] * (tweet_len + 3)

            input_ids += [torch.tensor(input_id)]
            attention_masks += [torch.tensor(attention_mask)]
            time_vectors += [torch.tensor(time_vector)]

        input_ids = pad_sequence(input_ids, batch_first=True)
        attention_masks = pad_sequence(attention_masks, batch_first=True)
        time_vectors = torch.cat(time_vectors)

        dataset = TensorDataset(input_ids, attention_masks, time_vectors)

        return dataset
