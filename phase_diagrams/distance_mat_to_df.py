from pathlib import Path
from typing import NamedTuple, List, Tuple, Iterable

import hdbscan
import numpy as np
import pandas as pd
import umap
import umap.plot
from joblib import Parallel, delayed
from sklearn.cluster import DBSCAN
from tqdm import tqdm

RANDOM_SEED: int = 42
ROOT_FOLDER_PATH: Path = Path(__file__).parent
PLOT_FOLDER_PATH: Path = Path(ROOT_FOLDER_PATH, 'plots')
EMBEDDINGS_FOLDER_PATH: Path = Path(ROOT_FOLDER_PATH, 'embeddings')
RAW_DATA_FOLDER_PATH: Path = Path(ROOT_FOLDER_PATH, 'raw_data')
NPZ_NAME_TEMPLATE: str = '{method}.npz'

DISTANCE_MAT_FILE_PATH: Path = Path(ROOT_FOLDER_PATH, 'binary_distances.npz')

SPACE_GROUP_CUTOFFS: List[int] = [2, 15, 74, 142, 167, 194, 230]


class PhaseDistances(NamedTuple):
    mat: np.ndarray
    names: List[str]
    atomic_group: List[int]


def load_distances() -> PhaseDistances:
    """
    Load the phase diagram distance matrix, and accompanying data.
    """
    data = np.load(str(DISTANCE_MAT_FILE_PATH))
    return PhaseDistances(data['mat'], data['names'], data['highest_groups'])


def project_single(distances: np.ndarray, n: int) -> Tuple[np.ndarray, str]:
    """
    UMAP Project using precomputed distances at a specific nearest neighbors value.
    :param distances: The distances to use for the projection.
    :param n: The number of nearest neighbors to consider.
    :return: A Tuple containing the projection and the number of nearest neighbors used.
    """
    projection = umap.UMAP(n_neighbors=n,
                           metric='precomputed',
                           random_state=RANDOM_SEED,
                           n_jobs=2).fit_transform(distances)
    return projection, str(n)


def project_distance_mat(distance_mat: PhaseDistances) -> pd.DataFrame:
    """
    Plot the phase diagram distances using umap in a variety of ways. Currently only exploring
    relevant metrics and a small number of nearest neighbor variations.
    :param distance_mat: The phase diagram distances to plot, as well as the names of the methods
    used.
    :return: The dataframe of structure info, and projections.
    """
    print("Projecting with a variety of settings.")
    neighbors = list(range(2, 20))
    df = pd.DataFrame({'name': distance_mat.names, 'atomic_group': distance_mat.atomic_group})
    projections: Iterable[Tuple[np.ndarray, str]] = Parallel(n_jobs=4)(
        delayed(project_single)(distance_mat.mat, n) for n in tqdm(neighbors))
    for projection, name in projections:
        df[name + '_x'] = projection[:, 0]
        df[name + '_y'] = projection[:, 1]
    return df


def cluster_distance_mat(distance_mat: PhaseDistances) -> pd.DataFrame:
    """
    Cluster the phase diagram distances using a variety of clustering algs.
    :param distance_mat: The phase diagram distances to cluster.
    :return: The dataframe of cluster labels.
    """
    clustering_methods = [('dbscan', DBSCAN(metric='precomputed')),
                          ('hdbscan', hdbscan.HDBSCAN(metric='precomputed'))]
    df = pd.DataFrame()
    print("Clustering")
    for method_name, method in tqdm(clustering_methods):
        df[method_name] = method.fit_predict(distance_mat.mat)
    return df


def main():
    dist_mat = load_distances()
    projection_df = project_distance_mat(dist_mat)
    cluster_df = cluster_distance_mat(dist_mat)
    df = pd.concat([projection_df, cluster_df], axis=1)
    df.to_csv(Path(ROOT_FOLDER_PATH, 'crystal_phase_projections.csv'), index=False)


if __name__ == '__main__':
    main()
