import json
import pathlib
from typing import Dict, List, NamedTuple

import matplotlib.pyplot as plt
import networkx as nx
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.structure import Structure

ROOT_FOLDER_PATH: pathlib.Path = pathlib.Path(__file__).parent
RAW_DATA_FOLDER_PATH: pathlib.Path = pathlib.Path(ROOT_FOLDER_PATH, 'raw_data')
ATOM_EMBEDDING_FILE_PATH: pathlib.Path = pathlib.Path(RAW_DATA_FOLDER_PATH, 'atom_init.json')
NX_DATA_FOLDER_PATH: pathlib.Path = pathlib.Path(ROOT_FOLDER_PATH, 'nx_data')

PICKLE_FILE_NAME_TEMPLATE: str = '{mpid}.pkl'

RADIUS = 8
MAX_NUM_NEIGHBORS = 12


class Neighbor(NamedTuple):
    index: int
    weight: float


def get_atom_embeddings(embedding_file_path: pathlib.Path) -> Dict[int, List[int]]:
    """
    Given the path to a json file of key to embedding format, return a dictionary of the same form.
    :param embedding_file_path: The path to the embedding file.
    :return: The dictionary of key to embedding.
    """
    with open(embedding_file_path, 'r') as file:
        data = json.load(file)
    return {int(key): value for key, value in data.items()}


def get_neighbors_cgcnn(
        crystal: Structure,
        radius: int,
        max_num_neighbors: int) -> List[List[Neighbor]]:
    """
    CAUTION: CAN RETURN FEWER THAN MAX_NUM_NEIGHBORS.

    Gets neighbors in the style of the cgcnn paper.

    Get the neighbors of each site within the structure. Only look within radius units, and only
    include max_num_neighbors. If greater than max_num_neighbors is found return the closest
    neighbors.
    :param crystal: The crystal get the neighbors of.
    :param radius: The radius to search each atom within.
    :param max_num_neighbors: The maximum number of neighbors to include.
    :return: max_num_neighbors neighbors within radius units of each site within the structure.
    """
    all_neighbors = []
    for sites_neighbors in crystal.get_all_neighbors(radius):
        all_neighbors.append(list(
            map(lambda nbr: Neighbor(nbr.index, nbr.nn_distance),
                sorted(sites_neighbors, key=lambda nbr: nbr.nn_distance)[:max_num_neighbors])))
    return all_neighbors


def get_neighbors_crystalnn(crystal: Structure) -> List[List[Neighbor]]:
    """
    Gets the neighbors of each site using CrystalNN. As described in:
    "Benchmarking_Coordination_Number_Prediction_Algorithms_on_Inorganic_Crystal_Structures"
    :param crystal: The crystal to get the neighbors per site for.
    :return: The neighbors per site.
    """
    all_nn_info = CrystalNN().get_all_nn_info(crystal)
    all_neighbors = []
    for idx, neighbors in enumerate(all_nn_info):
        all_neighbors.append(
            [Neighbor(neighbor['site_index'], neighbor['weight']) for neighbor in neighbors])
    return all_neighbors


def create_graph(crystal: Structure,
                 atom_embeddings: Dict[int, List[int]],
                 neighbors_by_atom: List[List[Neighbor]]):
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
            if not graph.has_edge(idx, neighbor.index) and not graph.has_edge(neighbor.index, idx):
                graph.add_edges_from([(idx, neighbor.index, {'weight': neighbor.weight})])
    return graph


def save_graph(graph: nx.Graph, path: pathlib.Path) -> None:
    """
    Write the graph to file.
    :param graph: The graph to write.
    :param path: The path to write it to.
    :return: None.
    """
    with open(path, 'wb+') as file:
        nx.write_gpickle(graph, file)


def load_graph(path: pathlib.Path) -> None:
    """
    Read the file into a graph
    :param path: The path to read from.
    :return: The graph.
    """
    with open(path, 'rb') as file:
        return nx.read_gpickle(file)


def draw_graph(graph: nx.Graph) -> None:
    """
    Draws a networkx graph using matplotlib. Consider using Graphviz in the future.
    :param graph: The graph to draw.
    :return: None.
    """
    nx.draw(graph)
    plt.show()
    plt.clf()


def main():
    atom_embeddings = get_atom_embeddings(ATOM_EMBEDDING_FILE_PATH)
    for file_path in RAW_DATA_FOLDER_PATH.iterdir():
        if file_path.name.endswith('.cif'):
            save_file_name = pathlib.Path(NX_DATA_FOLDER_PATH,
                                          PICKLE_FILE_NAME_TEMPLATE.format(mpid=file_path.stem))
            # graph = read_graph(save_file_name)
            crystal = Structure.from_file(str(file_path))
            # This will not warn if there are not enough neighbors
            neighbors = get_neighbors_cgcnn(crystal, RADIUS, MAX_NUM_NEIGHBORS)
            graph = create_graph(crystal, atom_embeddings, neighbors)
            # draw_graph(graph)
            save_graph(graph, pathlib.Path(NX_DATA_FOLDER_PATH,
                                           PICKLE_FILE_NAME_TEMPLATE.format(mpid=file_path.stem)))


if __name__ == '__main__':
    main()
