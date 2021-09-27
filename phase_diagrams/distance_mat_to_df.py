from pathlib import Path
from typing import NamedTuple, List, Tuple, Iterable

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


def project_embeddings(distance_mat: PhaseDistances) -> pd.DataFrame:
    """
    Plot the embeddings using umap in a variety of ways. Currently only exploring relevant metrics
    and a small number of nearest neighbor variations.
    :param embedding_df: The embeddings to plot, as well as the names of the methods used.
    :return: The dataframe of embeddings, structure info, and projections.
    """
    print("Projecting each embedding with a variety of settings.")
    neighbors = list(range(2, 20))
    df = pd.DataFrame({'name': distance_mat.names, 'atomic_group': distance_mat.atomic_group})
    projections: Iterable[Tuple[np.ndarray, str]] = Parallel(n_jobs=4)(
        delayed(project_single)(distance_mat.mat, n) for n in tqdm(neighbors))
    for projection, name in projections:
        df[name + '_x'] = projection[:, 0]
        df[name + '_y'] = projection[:, 1]
    return df


def main():
    sim_mat = load_distances()
    df = project_embeddings(sim_mat)
    df.to_csv(Path(ROOT_FOLDER_PATH, 'crystal_phase_projections.csv'), index=False)


if __name__ == '__main__':
    main()
