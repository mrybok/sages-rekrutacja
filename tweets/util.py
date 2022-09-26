import os
import torch
import pickle
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import sklearn.metrics as metrics

from typing import List
from typing import Dict
from typing import Tuple
from tqdm.auto import tqdm
from typing import Optional
from tqdm.auto import trange
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from common.util import check_for_gpu
from common.util import get_data_loader
from tweets.tokenizer import TweetsTokenizer
from tweets.classifier import TweetsClassifier


def get_embeddings(
        tweets: str,
        text_dicts: str,
        output_file: str,
        batch_size: int = 1,
        gpu: bool = False,
        verbose: bool = False,
        save: bool = True
) -> Dict[str, torch.Tensor]:

    tokenizer = TweetsTokenizer()

    tokenizer.load_text_dicts(text_dicts)

    tweets = pd.read_csv(tweets)
    dataset = tokenizer.get_bert_input(tweets[tweets.columns[1:]], verbose)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = {'X': [], 'y': []}

    if 'target' in tweets.columns:
        ys = tweets['target']
        embeddings['y'] = torch.tensor(ys)

    device = check_for_gpu(gpu)
    model = TweetsClassifier()

    model.bert.to(device)
    model.eval()

    with torch.no_grad():
        for input_ids, mask, time_vector in tqdm(loader, desc='Embedding', disable=not verbose):
            input_ids = input_ids.to(device)
            mask = mask.to(device)

            out = model.get_embeddings(input_ids, mask)
            out = out.detach().cpu()
            out = torch.cat((out, time_vector), dim=1)

            embeddings['X'] += [out]

    embeddings['X'] = torch.cat(embeddings['X'])

    if save:
        with open(output_file, 'wb') as file:
            pickle.dump(embeddings, file)

    return embeddings


def train_classifier_head(
        train_set: str,
        experiment_dir: str,
        valid_set: Optional[str] = None,
        batch_size: int = 1,
        epochs: int = 100,
        gpu: bool = False,
        seed: int = 0,
        tensorboard: bool = False,
        use_time_rep: bool = True
) -> Tuple[nn.Module, Dict[str, Dict[str, List[float]]]]:
    try:
        os.mkdir(experiment_dir)
    except OSError:
        pass

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    loaders = {}
    min_valid_loss = np.inf
    writer = SummaryWriter(experiment_dir + '/tensorboard') if tensorboard else None
    history = {split: {'loss': [], 'acc': [], 'f1': []} for split in ['train', 'valid']}

    loaders['train'] = get_data_loader(train_set, batch_size, True)

    if valid_set is not None:
        loaders['valid'] = get_data_loader(valid_set, batch_size, False)

    model = TweetsClassifier(772 if use_time_rep else 768)
    criterion = nn.BCELoss()
    optim = Adam(model.head.parameters())
    device = check_for_gpu(gpu)

    # Do not waste gpu space on BERT or its computational graph
    model.eval()
    model.head.to(device)

    for epoch in trange(epochs):
        model.head.train()

        train_loss = 0
        train_true = []
        train_pred = []

        for data, target in loaders['train']:
            train_true += [target.numpy()]

            if not use_time_rep:
                data = data[:, :-4]

            data = data.to(device)
            target = target.to(device)

            out = model(data)
            loss = criterion(out, target.unsqueeze(dim=1).float())
            train_loss += loss.item()
            train_pred += [torch.round(out).detach().cpu().numpy()]

            optim.zero_grad()
            loss.backward()
            optim.step()

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)

        train_loss /= len(loaders['train'])
        train_acc = metrics.accuracy_score(train_true, train_pred) * 100
        train_f1 = metrics.f1_score(train_true, train_pred, average='macro') * 100

        history['train']['loss'] += [train_loss]
        history['train']['acc'] += [train_acc]
        history['train']['f1'] += [train_f1]

        if valid_set is not None:
            valid_loss, valid_true, valid_pred = evaluate(model, loaders['valid'], device, use_time_rep)

            # Save best model
            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                torch.save(model.head, f'{experiment_dir}/classifier_best.pth')

            valid_acc = metrics.accuracy_score(valid_true, valid_pred) * 100
            valid_f1 = metrics.f1_score(valid_true, valid_pred, average='macro') * 100

            history['valid']['loss'] += [valid_loss]
            history['valid']['acc'] += [valid_acc]
            history['valid']['f1'] += [valid_f1]

            if tensorboard:
                writer.add_scalars('loss', {'train': train_loss, 'valid': valid_loss}, epoch)
                writer.add_scalars('acc', {'train': train_acc, 'valid': valid_acc}, epoch)
                writer.add_scalars('f1', {'train': train_f1, 'valid': valid_f1}, epoch)

            print(
                f'Epoch: {epoch+1:>3d} | '
                f'train loss: {train_loss:>.3f} | '
                f'valid loss: {valid_loss:>.3f} | '
                f'train acc: {train_acc:4.1f}% | '
                f'valid acc: {valid_acc:4.1f}% | '
                f'train f1: {train_f1:4.1f}% | '
                f'valid f1: {valid_f1:4.1f}%'
            )
        else:
            print(
                f'Epoch: {epoch+1:>3d} | '
                f'train loss: {train_loss:>.3f} | '
                f'train acc: {train_acc:4.1f}% | '
                f'train f1: {train_f1:4.1f}% | '
            )

            if tensorboard:
                writer.add_scalars('loss', {'train': train_loss}, epoch)
                writer.add_scalars('acc', {'train': train_acc}, epoch)
                writer.add_scalars('f1', {'train': train_f1}, epoch)

    # Save performance history & last model checkpoint
    with open(f'{experiment_dir}/history.pkl', 'wb') as file:
        pickle.dump(history, file)

    torch.save(model.head, f'{experiment_dir}/classifier_last.pth')

    return model, history


def evaluate(
        model: nn.Module,
        loader: DataLoader,
        device: str = 'cpu',
        use_time_rep: bool = True
) -> Tuple[int, np.ndarray, np.ndarray]:
    criterion = nn.BCELoss()

    y_true = []
    y_pred = []
    loss = 0

    model.eval()

    with torch.no_grad():
        for data, target in loader:
            y_true += [target.numpy()]

            if not use_time_rep:
                data = data[:, :-4]

            data = data.to(device)
            target = target.to(device)

            out = model(data)
            loss += criterion(out, target.unsqueeze(dim=1).float()).item()
            y_pred += [torch.round(out).cpu().numpy()]

    loss /= len(loader)
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    return loss, y_true, y_pred


def evaluate_wrapper(
        checkpoint_path: str,
        dataset: str,
        batch_size: int = 1,
        gpu: bool = False,
        use_time_rep: bool = True
) -> Tuple[int, np.ndarray, np.ndarray]:
    device = check_for_gpu(gpu)
    model = TweetsClassifier(772 if use_time_rep else 768)

    model.load_head(checkpoint_path)
    model.head.to(device)

    loader = get_data_loader(dataset, batch_size, False)

    return evaluate(model, loader, device, use_time_rep)
