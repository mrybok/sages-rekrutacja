import pandas as pd
import networkx as nx

from collections import Counter


def split_data(stances: pd.DataFrame, valid_size: float = 0.05) -> pd.DataFrame:
    """
    Use BFS to split the dataset into train and validation sets, so that every body and headline
    belongs only to one of the sets and there are no inter-set pairings.
    The returned stances DataFrame has a new 'split'
    column indicating body-headline pair split assignment:
    1 - valid,
    0 - train,
    -1 - inter-set pair.

    :param stances: DataFrame with Headline IDs, Body IDs and Stances
    :param valid_size: ratio of connections in the validation set
    :return: stances DataFrame with new 'split' column.
    """

    assert 1 > valid_size > 0, 'Invalid valid_size value'

    stances_iter = stances.itertuples(index=False)

    # Build graph where headlines and bodies are nodes, while their pairings are edges
    edges = [(head, body, {'stance': stance}) for head, body, stance in stances_iter]
    graph = nx.Graph(edges)

    # scores guide the BFS expansion
    # expand connected component by adding node with the least score
    # score is +1 for every edge with node out validation set
    #          -1 for evert edge with node  in validation set
    # Heuristic for building splits with high intra- and low inter-split connectivity.
    scores = {node: len(list(graph.neighbors(node))) for node in graph.nodes}

    # Start search from the least connected node.
    first_node = sorted(scores.items(), key=lambda x: x[1])[0][0]

    # Valid nodes - nodes assigned to validation set
    valid_nodes = [first_node]

    # Update neighbor scores.
    for neighbor in graph.neighbors(first_node):
        scores[neighbor] -= 2

    # Frontier - edges between validation and train components
    frontier = list(graph.edges(valid_nodes[0], data=True))

    # Original distribution of class labels
    dist = list(Counter(stances['Stance']).items())

    # Bookkeeping - counts how many edges of given stance must still be added to build a validation
    # set of required size
    valid_dist = Counter({stance: int(count * valid_size) for stance, count in dist})

    while any([count > 0 for count in valid_dist.values()]):

        # Consider only edges of which stance is still needed in the validation set
        candidates = [edge for edge in frontier if edge[2]['stance'] in valid_dist]

        if len(candidates) == 0:
            candidates = frontier

        candidates = [edge[0] if edge[1] in valid_nodes else edge[1] for edge in candidates]
        candidates = [(node, scores[node]) for node in candidates]

        # Expand with node of the least score
        candidates = sorted(candidates, key=lambda x: x[1])
        new_node = candidates[0][0]
        valid_nodes += [new_node]

        # Remove intra-validation-set edges resulting from adding new node
        frontier = [edge for edge in frontier if new_node not in edge]

        for neighbor in graph.neighbors(new_node):
            edge_data = graph.get_edge_data(new_node, neighbor)

            if neighbor in valid_nodes:
                valid_dist[edge_data['stance']] -= 1
            else:
                frontier += [(new_node, neighbor, edge_data)]
                scores[neighbor] -= 2

    def train_or_valid(row: pd.core.series.Series) -> int:
        """
        Assign a flag to a body-heading pair of stances table, indicating assigned split:
        1 - valid, 0 - train, -1 - none.

        :param row: row of stances pandas DataFrame
        :return: row split assignment
        """
        if row['Headline'] in valid_nodes and row['Body ID'] in valid_nodes:
            return 1
        elif row['Headline'] not in valid_nodes and row['Body ID'] not in valid_nodes:
            return 0

        return -1

    stances['split'] = stances.apply(lambda row: train_or_valid(row), axis=1)

    return stances
