import pickle

from tqdm.auto import tqdm
from transformers import BertTokenizer


class CommonTokenizer:

    def __init__(self,):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.text_dicts = {}

        self.CLS = self.tokenizer.cls_token_id
        self.SEP = self.tokenizer.sep_token_id

    def save_text_dicts(self, path: str):
        with open(path, 'wb') as file:
            pickle.dump(self.text_dicts, file)

    def load_text_dicts(self, path: str):
        with open(path, 'rb') as file:
            self.text_dicts = pickle.load(file)

    def tokenize_texts(self, verbose: bool = False):
        total = sum([len(text_dict) for text_dict in self.text_dicts.values()])
        pbar = tqdm(total=total, disable=not verbose, desc="tokenization")

        for text_dict in self.text_dicts.values():
            for text_id in text_dict:
                text = text_dict[text_id]['text']
                input_ids = self.tokenizer.encode(text, add_special_tokens=False)
                text_dict[text_id]['input_ids'] = input_ids
                pbar.update(1)

        pbar.close()
