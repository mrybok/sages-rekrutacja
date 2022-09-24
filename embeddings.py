import torch
import pickle
import argparse
import transformers

import pandas as pd

from typing import Dict
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from fake_news.tokenizer import FakeNewsTokenizer
from fake_news.classifier import FakeNewsClassifier

from defaults import LABELS
from defaults import DEFAULT_TEXT_DICTS_PATH


def get_embeddings(
        stances: str,
        text_dicts: str,
        output_file: str,
        batch_size: int = 1,
        gpu: bool = False,
        verbose: bool = False,
        save: bool = True
) -> Dict[str, torch.Tensor]:

    tokenizer = FakeNewsTokenizer()

    tokenizer.load_text_dicts(text_dicts)

    stances = pd.read_csv(stances)
    dataset = tokenizer.get_bert_input(stances[['Headline ID', 'Body ID']], verbose)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = {'X': [], 'y': []}

    if 'Stances' in stances.columns:
        ys = [LABELS[stance] for stance in stances['Stances']]
        embeddings['y'] = torch.tensor(ys)

    device = 'cpu'

    if gpu and torch.cuda.is_available():
        device = 'cuda'
    elif gpu and not torch.cuda.is_available():
        print('CUDA not available')

    model = FakeNewsClassifier()

    model.bert.to(device)

    for input_ids, segments, mask in tqdm(loader, desc='Embedding', disable=not verbose):
        input_ids = input_ids.to(device)
        segments = segments.to(device)
        mask = mask.to(device)

        out = model.get_embeddings(input_ids, segments, mask)
        out = out.detach().cpu()

        embeddings['X'] += [out]

    embeddings['X'] = torch.cat(embeddings['X'])

    if save:
        with open(output_file, 'wb') as file:
            pickle.dump(embeddings, file)

    return embeddings


if __name__ == "__main__":
    transformers.logging.set_verbosity_error()

    parser = argparse.ArgumentParser()

    parser.add_argument("--stances", type=str, help="path to .csv stance file", required=True)
    parser.add_argument("--text_dicts", type=str, help="path to saved text_dicts .pkl file", default=DEFAULT_TEXT_DICTS_PATH)
    parser.add_argument("--output_file", type=str, help="where to save the embeddings")
    parser.add_argument("--batch_size", type=int, help="inputs for BERT in batches", default=1)
    parser.add_argument("--gpu",  help="whether to use GPU", action='store_true')
    parser.add_argument("--verbose", help="whether to log progress with tqdm", action='store_true')

    args = parser.parse_args()
    output_file = args.output_file

    if output_file is None:
        file_name = args.stances.split('/')[-1][:-4]
        output_file = f"preprocessed/fake_news/embeddings/{file_name}.pkl"

    get_embeddings(
        args.stances, args.text_dicts, output_file, args.batch_size, args.gpu, args.verbose
    )
