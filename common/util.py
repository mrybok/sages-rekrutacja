import torch
import pickle

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from common.tokenizer import CommonTokenizer


def tokenize_texts(in_file: str, out_file: str, verbose: bool):
    tokenizer = CommonTokenizer()

    tokenizer.load_text_dicts(in_file)
    tokenizer.tokenize_texts(verbose)
    tokenizer.save_text_dicts(out_file)


def check_for_gpu(gpu: bool) -> str:
    device = 'cpu'

    if gpu and torch.cuda.is_available():
        device = 'cuda'
    elif gpu and not torch.cuda.is_available():
        print('CUDA not available')

    return device


def get_data_loader(dataset: str, batch_size: int = 1, shuffle: bool = False) -> DataLoader:
    with open(dataset, 'rb') as file:
        dataset = pickle.load(file)

    dataset = TensorDataset(*dataset.values())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader
