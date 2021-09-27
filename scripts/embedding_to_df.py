import bisect
import csv
import json
from itertools import product
from pathlib import Path
from typing import Dict, NamedTuple, List, Tuple, Iterable, Union

import numpy as np
import pandas as pd
import umap
import umap.plot
from joblib import Parallel, delayed
from tqdm import tqdm

RANDOM_SEED: int = 42
ROOT_FOLDER_PATH: Path = Path(__file__).parent
PLOT_FOLDER_PATH: Path = Path(ROOT_FOLDER_PATH, 'plots')
EMBEDDINGS_FOLDER_PATH: Path = Path(ROOT_FOLDER_PATH, 'embeddings')
PROJECTIONS_FOLDER_PATH: Path = Path(ROOT_FOLDER_PATH, 'projections')
RAW_DATA_FOLDER_PATH: Path = Path(ROOT_FOLDER_PATH, 'raw_data')
DF_PROJECTION_NAME_TEMPLATE: str = '{method}_{metric}_{n_neighbors}'
NPZ_NAME_TEMPLATE: str = '{method}.npz'

SPACE_GROUP_CUTOFFS: List[int] = [2, 15, 74, 142, 167, 194, 230]


class EmbeddingDF(NamedTuple):
    df: pd.DataFrame
    embedding_names: List[str]


def get_attributes_by_cids() -> Dict[int, Dict[str, Union[int, str]]]:
    """
    Given the path to a json file of cid to attributes file, return a dictionary of the same form.
    :return: The dictionary of cid to attributes.
    """
    with open(Path(RAW_DATA_FOLDER_PATH, 'attributes_by_cid.json'), 'r') as file:
        data = json.load(file)
        # The bisect reorganizes the space groups into 6 families so we don't need 200 and something
        # unique colors
        for value in data.values():
            value['space_group'] = bisect.bisect_left(SPACE_GROUP_CUTOFFS, value['space_group'])

        return {int(key): value for key, value in data.items()}


def get_formation_energies_by_cids() -> Dict[int, float]:
    """
    Gets the target variables as defined in id_prop.csv.
    :return: The target variables by cif file id.
    """
    with open(Path(RAW_DATA_FOLDER_PATH, 'id_prop.csv'), 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        formation_energies_by_id = {}
        for row in reader:
            formation_energies_by_id[int(row[0])] = float(row[1])
    return formation_energies_by_id


def load_embeddings(attributes_by_cids: Dict[int, Dict[str, Union[int, str]]],
                    formation_energies_by_cid: Dict[int, float]) -> EmbeddingDF:
    """
    Loads the embeddings from npzs on disk. Uses the attributes_by_cids by id dictionary to add
    space groups and composition formulas to the embedding.
    :param attributes_by_cids: Dictionary of attributes by cid.
    :param formation_energies_by_cid: Dictionary of formation energies by cid.
    :return: An EmbeddingDF containing a dataframe with all of the embeddings and additional
    information about the structures they represent, as well as a list of the different methods
    used to produce an embedding.
    """
    embeddings_df = pd.DataFrame()
    embedding_names = []
    index = 0
    for file_path in EMBEDDINGS_FOLDER_PATH.iterdir():
        if file_path.name.endswith('.npz'):
            embedding_name = file_path.stem
            embedding_names.append(embedding_name)
            data = np.load(str(file_path))
            embeddings_df[embedding_name] = data['embedding'].tolist()
            if index == 0:
                cids = data['cids']
                embeddings_df['cids'] = cids
                embeddings_df['formation_energy'] = np.array(
                    [formation_energies_by_cid[cid] for cid in cids])
                embeddings_df['space_group'] = np.array(
                    [attributes_by_cids[cid]['space_group'] for cid in cids])
                embeddings_df['formula'] = np.array(
                    [attributes_by_cids[cid]['formula'] for cid in cids])
                embeddings_df['most_prevalent_atomic_group'] = np.array(
                    [attributes_by_cids[cid]['most_prevalent_atomic_group'] for cid in cids])
            index += 1
    embeddings_df = embeddings_df.sample(10000)
    return EmbeddingDF(embeddings_df, embedding_names)


def project_single(
        embedding: np.ndarray,
        name: str,
        metric: str,
        n: int) -> Tuple[np.ndarray, str]:
    """
    Project a single configuration
    :param embedding: The embedding to project.
    :param name: The name of the embedding.
    :param metric: The metric for UMAP to use to calculate distances between embeddings.
    :param n: The number of neighbors for UMAP to use.
    :return: The projection, and the projection name.
    """
    projection = umap.UMAP(n_neighbors=n, metric=metric, random_state=RANDOM_SEED,
                           n_jobs=2).fit_transform(embedding)
    projection_name = DF_PROJECTION_NAME_TEMPLATE.format(method=name, metric=metric, n_neighbors=n)
    return projection, projection_name


def project_embeddings(embedding_df: EmbeddingDF) -> pd.DataFrame:
    """
    Plot the embeddings using umap in a variety of ways. Currently only exploring relevant metrics
    and a small number of nearest neighbor variations.
    :param embedding_df: The embeddings to plot, as well as the names of the methods used.
    :return: The dataframe of embeddings, structure info, and projections.
    """
    print("Projecting each embedding with a variety of settings.")
    neighbors = [2, 4, 5, 10, 20, 40, 80]
    distance_metrics = ['euclidean', 'cosine']
    df = embedding_df.df
    params = list(product(embedding_df.embedding_names, distance_metrics, neighbors))
    projections: Iterable[Tuple[np.ndarray, str]] = Parallel(n_jobs=4)(
        delayed(project_single)(np.vstack(df[name].values.tolist()), name, metric, n) for
        name, metric, n
        in tqdm(params))
    for projection, name in projections:
        df[name + '_x'] = projection[:, 0]
        df[name + '_y'] = projection[:, 1]
    return df


def main():
    attributes_by_cids = get_attributes_by_cids()
    formation_energies_by_cids = get_formation_energies_by_cids()
    embedding_df = load_embeddings(attributes_by_cids, formation_energies_by_cids)
    df = project_embeddings(embedding_df)
    df.to_csv(Path(PROJECTIONS_FOLDER_PATH, 'crystal_graph_embedding_projections.csv'), index=False)


if __name__ == '__main__':
    main()
