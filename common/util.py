import torch

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
