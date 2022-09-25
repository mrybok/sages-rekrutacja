import os
import torch
import pickle

import numpy as np
import pandas as pd
import networkx as nx
import torch.nn as nn
import sklearn.metrics as metrics

from typing import Dict
from tqdm.auto import tqdm
from tqdm.auto import trange
from torch.optim import Adam
from collections import Counter
from fake_news.defaults import LABELS
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from fake_news.tokenizer import FakeNewsTokenizer
from fake_news.classifier import FakeNewsClassifier

from torch.utils.tensorboard import SummaryWriter

def score_to_prob(candidates: dict) -> np.array:
    """
    Turns the scores to probability distribution for sampling.

    :param candidates: dictionary mapping candidate node to its score
    :return: probability distribution,
    """
    prob = np.array(list(candidates.values()))
    prob = prob + 1 - min(prob)
    prob = prob / sum(prob)

    return prob


def split_data(stances: pd.DataFrame, valid_size: float = 0.1, seed: int = 0) -> pd.DataFrame:
    """
    Use BFS to split the dataset into train and validation sets, so that every body and headline
    belongs only to one of the sets and there are no inter-set pairings.
    The returned stances DataFrame has a new 'split'
    column indicating body-headline pair split assignment.

    :param stances: DataFrame with Headline IDs, Body IDs and Stances
    :param valid_size: ratio of connections in the validation set
    :param seed: Random number generator seed
    :return: new_stances DataFrame with new 'Split' column.
    """

    assert 1 > valid_size > 0, 'Invalid valid_size value'

    stances_iter = stances.itertuples(index=False)

    # Build graph where headlines and bodies are nodes, while their pairings are edges
    edges = [(head, body, {'Stance': stance}) for head, body, stance in stances_iter]
    graph = nx.Graph(edges)

    # scores guide the BFS expansion
    # node's score is +1 for every edge with node in validation set
    #                 -1 for every edge with node out validation set
    # Heuristic for building splits with high intra- and low inter-split connectivity.
    scores = {node: -len(list(graph.neighbors(node))) for node in graph.nodes}

    rng = np.random.RandomState(seed)

    # Sample first_node - the least connected nodes are most likely selected.
    first_node = rng.choice(list(scores), p=score_to_prob(scores))

    # Valid nodes - nodes assigned to validation set
    valid_nodes = [first_node]

    # Update neighbor scores.
    for neighbor in graph.neighbors(first_node):
        scores[neighbor] += 2

    # Frontier - edges between validation and train components
    frontier = list(graph.edges(valid_nodes[0], data=True))

    # Original distribution of class labels
    dist = list(Counter(stances['Stance']).items())

    # Bookkeeping - counts how many edges of given stance must still be added to build a validation
    # set of required size
    valid_dist = Counter({stance: int(count * valid_size) for stance, count in dist})

    pbar = tqdm(total=int(sum(valid_dist.values())))

    while any([count > 0 for count in valid_dist.values()]):

        # Consider only edges of which stance is still needed in the validation set
        candidates = [edge for edge in frontier if edge[2]['Stance'] in valid_dist]

        if len(candidates) == 0:
            candidates = frontier

        candidates = [edge[0] if edge[1] in valid_nodes else edge[1] for edge in candidates]
        candidates = {node: scores[node] for node in candidates}

        # Sample a node for expansion
        new_node = rng.choice(list(candidates), p=score_to_prob(candidates))
        valid_nodes += [new_node]

        # Remove intra-validation-set edges resulting from adding new node
        frontier = [edge for edge in frontier if new_node not in edge]

        for neighbor in graph.neighbors(new_node):
            edge_data = graph.get_edge_data(new_node, neighbor)

            if neighbor in valid_nodes:
                stance = edge_data['Stance']
                valid_dist[stance] -= 1

                if valid_dist[stance] >= 0:
                    pbar.update(1)
            else:
                frontier += [(new_node, neighbor, edge_data)]
                scores[neighbor] += 2

    pbar.close()

    def train_or_valid(row: pd.core.series.Series) -> str:
        """
        Assign a flag to a body-heading pair of stances table, indicating assigned split.

        :param row: row of stances pandas DataFrame
        :return: row split assignment
        """
        if row['Headline ID'] in valid_nodes and row['Body ID'] in valid_nodes:
            return 'valid'
        elif row['Headline ID'] not in valid_nodes and row['Body ID'] not in valid_nodes:
            return 'train'

        return 'none'

    new_stances = stances.copy()
    new_stances['Split'] = stances.apply(lambda row: train_or_valid(row), axis=1)

    return new_stances


def tokenize_texts(in_file: str, out_file: str, verbose: bool):
    tokenizer = FakeNewsTokenizer()

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

    if 'Stance' in stances.columns:
        ys = [LABELS[stance] for stance in stances['Stance']]
        embeddings['y'] = torch.tensor(ys)

    device = check_for_gpu(gpu)
    model = FakeNewsClassifier()

    model.bert.to(device)
    model.eval()

    with torch.no_grad():
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


def train_classifier_head(
        train_set: str,
        experiment_dir: str,
        valid_set: str = None,
        batch_size: int = 1,
        epochs: int = 100,
        gpu: bool = False,
        tensorboard: bool = False,
):
    try:
        os.mkdir(experiment_dir)
    except OSError:
        pass

    loaders = {}
    min_valid_loss = np.inf
    writer = SummaryWriter(experiment_dir + '/tensorboard') if tensorboard else None
    history = {split: {'loss': [], 'acc': [], 'f1': []} for split in ['train', 'valid']}

    with open(train_set, 'rb') as file:
        train_dataset = pickle.load(file)

    train_dataset = TensorDataset(*train_dataset.values())
    loaders['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if valid_set is not None:
        with open(valid_set, 'rb') as file:
            valid_dataset = pickle.load(file)

        valid_dataset = TensorDataset(*valid_dataset.values())
        loaders['valid'] = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    model = FakeNewsClassifier()
    criterion = nn.CrossEntropyLoss()
    optim = Adam(model.head.parameters())
    device = check_for_gpu(gpu)

    model.eval()
    model.head.to(device)

    for epoch in trange(epochs):
        model.head.train()

        train_loss = 0
        train_true = []
        train_pred = []

        for data, target in loaders['train']:
            train_true += [target.numpy()]

            data = data.to(device)
            target = target.to(device)

            out = model(data)
            loss = criterion(out, target)
            train_loss += loss.item()
            train_pred += [torch.argmax(out, dim=1).cpu().numpy()]

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
            valid_loss = 0
            valid_true = []
            valid_pred = []

            model.head.eval()

            with torch.no_grad():
                for data, target in loaders['valid']:
                    valid_true += [target.numpy()]

                    data = data.to(device)
                    target = target.to(device)

                    out = model(data)
                    loss = criterion(out, target)
                    valid_loss += loss.item()
                    valid_pred += [torch.argmax(out, dim=1).cpu().numpy()]

            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                torch.save(model.head, f'{experiment_dir}/classifier_best.pth')

            valid_true = np.concatenate(valid_true)
            valid_pred = np.concatenate(valid_pred)

            valid_loss /= len(loaders['valid'])
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

    with open(f'{experiment_dir}/history.pkl', 'wb') as file:
        pickle.dump(history, file)

    torch.save(model.head, f'{experiment_dir}/classifier_last.pth')

def evaluate_model():
    pass
