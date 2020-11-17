import json
import pathlib
from typing import Dict, List

import networkx as nx
from pymatgen.core.structure import Structure, PeriodicNeighbor

ROOT_FOLDER_PATH: pathlib.Path = pathlib.Path(__file__).parent
RAW_DATA_FOLDER_PATH: pathlib.Path = pathlib.Path(ROOT_FOLDER_PATH, 'raw_data')
ATOM_EMBEDDING_FILE_PATH: pathlib.Path = pathlib.Path(RAW_DATA_FOLDER_PATH, 'atom_init.json')
NX_DATA_FOLDER_PATH: pathlib.Path = pathlib.Path(ROOT_FOLDER_PATH, 'nx_data')

PICKLE_FILE_NAME_TEMPLATE: str = '{mpid}.pkl'

RADIUS = 8
MAX_NUM_NEIGHBORS = 12


def get_atom_embeddings(embedding_file_path: pathlib.Path) -> Dict[int, List[int]]:
    """
    Given the path to a json file of key to embedding format, return a dictionary of the same form.
    :param embedding_file_path: The path to the embedding file.
    :return: The dictionary of key to embedding.
    """
    with open(embedding_file_path, 'r') as file:
        data = json.load(file)
    return {int(key): value for key, value in data.items()}


def get_neighbors_by_atom(
        crystal: Structure,
        radius: int,
        max_num_neighbors: int) -> List[List[PeriodicNeighbor]]:
    """
    CAUTION: CAN RETURN FEWER THAN MAX_NUM_NEIGHBORS.

    Get the neighbors of each site within the structure. Only look within radius units, and only
    include max_num_neighbors. If greater than max_num_neighbors is found return the closest
    neighbors.
    :param crystal: The crystal get the neighbors of.
    :param radius: The radius to search each atom within.
    :param max_num_neighbors: The maximum number of neighbors to include.
    :return: max_num_neighbors neighbors within radius units of each site within the structure.
    """
    all_neighbors = crystal.get_all_neighbors(radius)
    return [sorted(nbrs, key=lambda nbr: nbr.nn_distance)[:max_num_neighbors] for nbrs in
            all_neighbors]


def create_graph(crystal: Structure,
                 atom_embeddings: Dict[int, List[int]],
                 neighbors_by_atom: List[List[PeriodicNeighbor]]):
    """
    Create a networkx graph.
    :param crystal: The structure being converted into a graph.
    :param atom_embeddings: The embeddings for each atom.
    :param neighbors_by_atom: The closest neighbors of each atom.
    :return: The networkx graph.
    """
    graph = nx.Graph()
    for idx, site in enumerate(crystal.sites):
        graph.add_nodes_from([(idx, {'embedding': atom_embeddings[site.specie.number],
                                     'label': site.specie.number})])
        for neighbor in neighbors_by_atom[idx]:
            graph.add_edges_from([(idx, neighbor.index, {'weight': neighbor.nn_distance})])
    return graph


def write_graph(graph: nx.Graph, path: pathlib.Path) -> None:
    """
    Write the graph to file.
    :param graph: The graph to write.
    :param path: The path to write it to.
    :return: None.
    """
    with open(path, 'wb+') as file:
        nx.write_gpickle(graph, file)


def main():
    atom_embeddings = get_atom_embeddings(ATOM_EMBEDDING_FILE_PATH)
    for file_path in RAW_DATA_FOLDER_PATH.iterdir():
        if file_path.name.endswith('.cif'):
            crystal = Structure.from_file(str(file_path))
            # This will not warn if there are not enough neighbors
            neighbors_by_atom = get_neighbors_by_atom(crystal, RADIUS, MAX_NUM_NEIGHBORS)
            graph = create_graph(crystal, atom_embeddings, neighbors_by_atom)
            write_graph(graph, pathlib.Path(NX_DATA_FOLDER_PATH,
                                            PICKLE_FILE_NAME_TEMPLATE.format(mpid=file_path.stem)))


if __name__ == '__main__':
    main()
