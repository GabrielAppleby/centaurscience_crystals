import bisect
import csv
import json
import pathlib
from itertools import product
from typing import Dict, NamedTuple, List

import numpy as np
import matplotlib.pyplot as plt
import umap
import umap.plot
from tqdm import tqdm

ROOT_FOLDER_PATH: pathlib.Path = pathlib.Path(__file__).parent
PLOT_FOLDER_PATH: pathlib.Path = pathlib.Path(ROOT_FOLDER_PATH, 'plots')
EMBEDDINGS_FOLDER_PATH: pathlib.Path = pathlib.Path(ROOT_FOLDER_PATH, 'embeddings')
RAW_DATA_FOLDER_PATH: pathlib.Path = pathlib.Path(ROOT_FOLDER_PATH, 'raw_data')
PLOT_NAME_TEMPLATE: str = '{method}_{metric}_{n_neighbors}_{coloring}.png'
NPZ_NAME_TEMPLATE: str = '{method}.npz'

SPACE_GROUP_CUTOFFS: List[int] = [2, 15, 74, 142, 167, 194, 230]


class GroupedEmbedding(NamedTuple):
    cid: List[int]
    embedding: np.ndarray
    group: np.ndarray
    formation_energy: np.ndarray


def get_space_groups_by_cids() -> Dict[int, int]:
    """
    Given the path to a json file of cid to space group, return a dictionary of the same form.
    :return: The dictionary of cid to space group.
    """
    with open(pathlib.Path(RAW_DATA_FOLDER_PATH, 'space_groups_by_cid.json'), 'r') as file:
        data = json.load(file)
    return {int(key): bisect.bisect_left(SPACE_GROUP_CUTOFFS, int(value)) for key, value in data.items()}


def get_formation_energies_by_cids() -> Dict[int, float]:
    """
    Gets the target variables as defined in id_prop.csv.
    :return: The target variables by cif file id.
    """
    with open(pathlib.Path(RAW_DATA_FOLDER_PATH, 'id_prop.csv'), 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        formation_energies_by_id = {}
        for row in reader:
            formation_energies_by_id[int(row[0])] = float(row[1])
    return formation_energies_by_id


def load_embeddings(space_groups_by_cid: Dict[int, int],
                    formation_energies_by_cid: Dict[int, float]) -> Dict[str, GroupedEmbedding]:
    """
    Loads the embeddings from npzs on disk. Uses the space groups by id dictionary to add space
    groups to the embedding.
    :param space_groups_by_cid: Dictionary of space groups by cid.
    :param formation_energies_by_cid: Dictionary of formation energies by cid.
    :return: A dictionary with the name of the embedding method used to generate as the key, and
    a grouped embedding as the value.
    """
    embeddings = {}
    for file_path in EMBEDDINGS_FOLDER_PATH.iterdir():
        if file_path.name.endswith('.npz'):
            data = np.load(str(file_path))
            embedding = data['embedding']
            cids = data['cids']
            formation_energy = np.array([formation_energies_by_cid[cid] for cid in cids])
            group = np.array([space_groups_by_cid[cid] for cid in cids])
            method = file_path.stem
            embeddings[method] = GroupedEmbedding(cids, embedding, group, formation_energy)
    return embeddings


def plot_embeddings(embeddings: Dict[str, GroupedEmbedding]) -> None:
    """
    Plot the embeddings using umap in a variety of ways. Currently only exploring relevant metrics
    and a small number of nearest neighbor variations.
    :param embeddings: The embeddings to plot as a dictionary from method name to grouped embedding.
    :return: None. Graphs saved to disk.
    """
    print("Plotting each embedding with a variety of settings.")
    plot_params = list(product((2, 4, 5, 10, 20, 40, 80), ('euclidean', 'cosine')))
    for name, embedding in tqdm(embeddings.items(), desc='Embedding types'):  # type: str, GroupedEmbedding
        for n_neighbors, metric in tqdm(plot_params, desc='Plot params'):
            plot_file_name = PLOT_NAME_TEMPLATE.format(
                method=name, metric=metric, n_neighbors=n_neighbors, coloring='fe')
            mapper = umap.UMAP(
                n_neighbors=n_neighbors, metric=metric).fit(embedding.embedding)
            umap.plot.points(mapper, values=embedding.formation_energy)
            plt.savefig(pathlib.Path(PLOT_FOLDER_PATH, plot_file_name))
            plt.clf()


def main():
    space_groups_by_cids = get_space_groups_by_cids()
    formation_energies_by_cids = get_formation_energies_by_cids()
    embeddings = load_embeddings(space_groups_by_cids, formation_energies_by_cids)
    plot_embeddings(embeddings)


if __name__ == '__main__':
    main()
