import torch
import pickle
import transformers

from tqdm.auto import tqdm
from transformers import BertModel

transformers.logging.set_verbosity_error()

if __name__ == "__main__":
    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'

    model = BertModel.from_pretrained("bert-base-uncased")
    model = model.to(device)

    for split in ['train', 'valid', 'none', 'test']:
        with open(f'preprocessed/fake_news/bert_loaders/{split}.pkl', 'rb') as file:
            loader = pickle.load(file)

        X = []

        for input_ids, segments, mask, _ in tqdm(loader, desc=split):
            input_ids = input_ids.to(device)
            segments = segments.to(device)
            mask = mask.to(device)

            out = model(input_ids, token_type_ids=segments, attention_mask=mask)
            out = out.last_hidden_state[:, 0, :]
            out = out.detach().cpu()

            X += [out]

        with open(f'preprocessed/fake_news/embeddings/{split}.pkl', 'wb') as file:
            pickle.dump({'X': torch.cat(X)}, file)
