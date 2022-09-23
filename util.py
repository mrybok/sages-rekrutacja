import numpy as np
import pandas as pd
import networkx as nx

from tqdm.auto import tqdm
from collections import Counter


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
    :return: new_stances DataFrame with new 'split' column.
    """

    assert 1 > valid_size > 0, 'Invalid valid_size value'

    stances_iter = stances.itertuples(index=False)

    # Build graph where headlines and bodies are nodes, while their pairings are edges
    edges = [(head, body, {'stance': stance}) for head, body, stance in stances_iter]
    graph = nx.Graph(edges)

    # scores guide the BFS expansion
    # node's score is +1 for every edge with node  in validation set
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
        candidates = [edge for edge in frontier if edge[2]['stance'] in valid_dist]

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
                stance = edge_data['stance']
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
        if row['Headline'] in valid_nodes and row['Body ID'] in valid_nodes:
            return 'valid'
        elif row['Headline'] not in valid_nodes and row['Body ID'] not in valid_nodes:
            return 'train'

        return 'none'

    new_stances = stances.copy()
    new_stances['split'] = stances.apply(lambda row: train_or_valid(row), axis=1)

    return new_stances
