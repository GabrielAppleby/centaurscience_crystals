import pathlib
from typing import List, Dict, Tuple

import networkx as nx
import numpy as np
from karateclub import GL2Vec, FGSD, Graph2Vec
from tqdm import tqdm

ROOT_FOLDER_PATH: pathlib.Path = pathlib.Path(__file__).parent
NX_DATA_FOLDER_PATH: pathlib.Path = pathlib.Path(ROOT_FOLDER_PATH, 'nx_data')
EMBEDDINGS_FOLDER_PATH: pathlib.Path = pathlib.Path(ROOT_FOLDER_PATH, 'embeddings')
NPZ_NAME_TEMPLATE: str = '{method}_{feature}.npz'

FEATURE_NAMES = ['degree', 'atomic_number', 'atomic_group']


def load_graphs() -> Dict[int, nx.Graph]:
    """
    CAUTION: The methods for graph embeddings in the karateclub library seem to require fully
    connected graphs. So graphs that don't meet this requirement are filtered out at the moment.

    Load your graphs nx.Graphs saved in pickle format from a zip.
    :return: The nx.Graphs by cid.
    """
    graphs: Dict[int, nx.Graph] = {}
    print("Loading all of the graphs.")
    for file_name in tqdm(list(NX_DATA_FOLDER_PATH.glob('*.pkl')), desc='Graphs'):
        cid = file_name.name.split('.')[0]
        g: nx.Graph = nx.read_gpickle(file_name)
        if nx.is_connected(g):
            graphs[int(cid)] = g
    return graphs


def set_node_attributes(graph, feature_name):
    """
    Sets the feature node attribute of the graph based on the feature_name. Assumes the atomic
    number is stored in the graph under the node attributes 'label' and that the atomic group is
    stored in the graph under the node attributes 'atomic_group'.
    :param graph: The graph whose node attributes must be set.
    :param feature_name: The feature to set as the node attributes. Must be 'atomic_number' or
    'atomic_group'.
    :return: The graph with the correct node attributes set.
    """
    if feature_name == 'atomic_number':
        atomic_numbers = nx.get_node_attributes(graph, 'label')
        nx.set_node_attributes(graph, atomic_numbers, 'feature')
    elif feature_name == 'atomic_group':
        group_numbers = nx.get_node_attributes(graph, 'atomic_group')
        nx.set_node_attributes(graph, group_numbers, 'feature')
    return graph


def get_embeddings(graphs_by_cid: Dict[int, nx.Graph],
                   feature_name: str) -> Tuple[Dict[str, Tuple[np.ndarray, List[int]]], List[int]]:
    """
    Turn your graphs into embeddings. Currently tries everything in karateclub, that didn't throw an
    error.
    :param graphs_by_cid: Dict of cid to fully connected graphs to turn into embeddings.
    :param feature_name: The name of the atom features to use.
    :return: A tuple containing the different embeddings by embedding method + feature_name
    combination, and cids corresponding to each graph contained in each embedding.

    Remember, returned embeddings will be a dict containing embedding lists for each method +
    feature_name combination.
    [
    method/feature one: embedding,
    method/feature two: embedding,
    method/feature three: embedding
    ]

    And each embedding list will contain a vector embedding for each graph
    [
    graph one embedding,
    graph two embedding,
    graph three embedding
    ]

    The list of cids will be the same length and order as the embedding list.
    """
    if feature_name == 'degree':
        embedding_methods = [('gl2vec', GL2Vec()),
                             ('fgsd', FGSD()),
                             ('g2vec', Graph2Vec())]
    else:
        embedding_methods = [('g2vec', Graph2Vec(attributed=True))]
    embeddings = {}
    graphs = [set_node_attributes(graph, feature_name) for graph in graphs_by_cid.values()]
    cids = list(graphs_by_cid.keys())
    print("Creating all of the embeddings for feature: {}.".format(feature_name))
    for name, model in tqdm(embedding_methods, desc="Embedding Methods"):
        model.fit(graphs)
        embeddings[name] = model.get_embedding()
    return embeddings, cids


def save_embeddings(embeddings: Dict[str, Tuple[np.ndarray, List[int]]],
                    cids: List[int],
                    feature_name: str) -> None:
    """
    Saves all of the embeddings to disk in separate npzs. Why separate NPZs? So we can add
    embeddings from other places easily.
    :param embeddings: The embeddings by embedding method to save.
    :param cids: The cids corresponding to the embeddings.
    :param feature_name: The cids corresponding to the embeddings.
    :return: None. Embeddings and cids saved to disk. Even though in most cases the cids will be the
    same for each embedding, they are saved along each npz so that we can break that format if
    needed.
    """
    print("Saving embeddings to disk for feature: {}.".format(feature_name))
    for name, embedding in embeddings.items():
        npz_path = pathlib.Path(EMBEDDINGS_FOLDER_PATH,
                                NPZ_NAME_TEMPLATE.format(method=name, feature=feature_name))
        np.savez(npz_path, embedding=embedding, cids=cids)


def main():
    graphs_by_cid = load_graphs()
    for feature_name in tqdm(FEATURE_NAMES, desc='Node Features'):
        embeddings, cids = get_embeddings(graphs_by_cid, feature_name)
        save_embeddings(embeddings, cids, feature_name)


if __name__ == '__main__':
    main()
