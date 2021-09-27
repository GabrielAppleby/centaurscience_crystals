from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT_FOLDER_PATH: Path = Path(__file__).parent
PROJECTIONS_FOLDER_PATH: Path = Path(ROOT_FOLDER_PATH, 'projections')
PLOT_FOLDER_PATH: Path = Path(ROOT_FOLDER_PATH, 'plots')

SPACE_GROUP_CUTOFFS: List[int] = [2, 15, 74, 142, 167, 194, 230]


def load_projections() -> pd.DataFrame:
    """
    Loads the projections and supporting embedding information.
    :return: A pd.DataFrame containing all of the projections, embeddings, and structure info.
    """
    return pd.read_csv(Path(PROJECTIONS_FOLDER_PATH, 'crystal_graph_embedding_projections.csv'),
                       index_col=False)


def plot_embeddings(df: pd.DataFrame) -> None:
    """
    Plot the projections and save to disk.
    :param df: The pd.DataFrame containing all of the projections and coloring values.
    :return: None. Graphs saved to disk.
    """
    projection_columns = [c for c in df.columns if c.endswith('_x')]
    for proj_col in projection_columns:
        for c_column, cmap, legend_title in [('space_group', 'tab10', 'Space Group'),
                                             ('most_prevalent_atomic_group', 'tab20', 'Atomic Group')]:
            method_metric_neighbor = proj_col[:-1]
            ax = sns.scatterplot(x=df[proj_col],
                                 y=df[method_metric_neighbor + 'y'],
                                 s=8.0,
                                 linewidth=.1,
                                 hue=df[c_column],
                                 palette=cmap,
                                 edgecolor='white')
            ax.set_axis_off()
            ax.legend(loc='center left', title=legend_title, bbox_to_anchor=(1, 0.5))
            plt.savefig(Path(PLOT_FOLDER_PATH, method_metric_neighbor + '_{}.pdf'.format(c_column)),
                        bbox_inches='tight')
            plt.clf()


def main():
    df = load_projections()
    plot_embeddings(df)


if __name__ == '__main__':
    main()
