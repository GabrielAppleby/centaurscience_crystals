import pathlib
import numpy as np
from typing import List, Dict, Tuple
from zipfile import ZipFile

import networkx as nx
from karateclub import GL2Vec, FGSD, Graph2Vec
from tqdm import tqdm

ROOT_FOLDER_PATH: pathlib.Path = pathlib.Path(__file__).parent
NX_DATA_FOLDER_PATH: pathlib.Path = pathlib.Path(ROOT_FOLDER_PATH, 'nx_data')
EMBEDDINGS_FOLDER_PATH: pathlib.Path = pathlib.Path(ROOT_FOLDER_PATH, 'embeddings')
NPZ_NAME_TEMPLATE: str = '{method}.npz'


def load_graphs() -> Dict[int, nx.Graph]:
    """
    CAUTION: The methods for graph embeddings in the karateclub library seem to require fully
    connected graphs. So graphs that don't meet this requirement are filtered out at the moment.

    Load your graphs nx.Graphs saved in pickle format from a zip.
    :return: The nx.Graphs by cid.
    """
    graphs: Dict[int, nx.Graph] = {}
    with ZipFile(pathlib.Path(NX_DATA_FOLDER_PATH, 'graphs.zip'), 'r') as read_zip:
        print("Loading all of the graphs. Progress bar (1/2).")
        for file_name in tqdm(read_zip.namelist()):
            cid = file_name.split('.')[0]
            with read_zip.open(file_name, 'r') as file:
                g: nx.Graph = nx.read_gpickle(file)
                if nx.is_connected(g):
                    graphs[int(cid)] = g
    return graphs


def get_embeddings(graphs_by_cid: Dict[int, nx.Graph]) -> Tuple[Dict[str, Tuple[np.ndarray, List[int]]], List[int]]:
    """
    Turn your graphs into embeddings. Currently tries everything in karateclub, that didn't throw an
    error.
    :param graphs_by_cid: Dict of cid to fully connected graphs to turn into embeddings.
    :return: A tuple containing the different embeddings by embedding method, and cids corresponding
    to each graph contained in each embedding.

    Remember, returned embeddings will be a list containing embedding lists for each method
    [
    method one embedding,
    method two embedding,
    method three embedding
    ]

    And each embedding list will contain a vector embedding for each graph
    [
    graph one embedding,
    graph two embedding,
    graph three embedding
    ]

    The list of cids will be the same length and order as the embedding list.

    """
    embedding_methods = [('gl2vec', GL2Vec()),
                         ('fgsd', FGSD()),
                         ('g2vec', Graph2Vec())]
    embeddings = {}
    graphs = graphs_by_cid.values()
    cids = list(graphs_by_cid.keys())
    print("Creating all of the embeddings. This part takes the longest by far. Progress bar (2/2).")
    for name, model in tqdm(embedding_methods):
        model.fit(graphs)
        embeddings[name] = model.get_embedding()
    return embeddings, cids


def save_embeddings(embeddings: Dict[str, Tuple[np.ndarray, List[int]]], cids: List[int]) -> None:
    """
    Saves all of the embeddings to disk in separate npzs. Why separate NPZs? So we can add
    embeddings from other places easily.
    :param embeddings: The embeddings by embedding method to save.
    :param cids: The cids corresponding to the embeddings.
    :return: None. Embeddings and cids saved to disk. Even though in most cases the cids will be the
    same for each embedding, they are saved along each npz so that we can break that format if
    needed.
    """
    print("Saving embeddings to disk. You made it!")
    for name, embedding in embeddings.items():
        npz_path = pathlib.Path(EMBEDDINGS_FOLDER_PATH, NPZ_NAME_TEMPLATE.format(method=name))
        np.savez(npz_path, embedding=embedding, cids=cids)


def main():
    graphs_by_cid = load_graphs()
    embeddings, cids = get_embeddings(graphs_by_cid)
    save_embeddings(embeddings, cids)


if __name__ == '__main__':
    main()
