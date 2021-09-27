import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pymatgen.core as mg

ROOT_FOLDER_PATH: Path = Path(__file__).parent
SIMILARITY_JSON_FILE_PATH: Path = Path(ROOT_FOLDER_PATH, 'binary_similarities_with_sg.json')
DISTANCE_MAT_FILE_PATH: Path = Path(ROOT_FOLDER_PATH, 'binary_distances.npz')
IMPORTANT_GROUPS: List[int] = [15, 16, 17]


def get_similarity_dict(sim_dict_file_path: Path) -> Dict[str, Dict[str, int]]:
    """
    Gets the similarity dictionary.
    :param sim_dict_file_path: The path to the similarity dictionary file.
    :return: The similarity dictionary.
    """
    with open(sim_dict_file_path, 'r') as file:
        return json.load(file)


def get_group(formula: str):
    """
    :param formula: The formula of the composition.
    :return: The highest atomic group within IMPORTANT_GROUPS or -1 if none of those atomic groups
    are present.
    """
    groups = [e.group for e in mg.Composition(formula).elements]
    final_group = -1
    for current_group in groups:
        if current_group in IMPORTANT_GROUPS and current_group > final_group:
            final_group = current_group
    return final_group


def main():
    sim_dict = get_similarity_dict(SIMILARITY_JSON_FILE_PATH)
    num_entries = len(sim_dict)
    sim_matrix = np.zeros((num_entries, num_entries))
    entity_names = [key for key in sorted(sim_dict.keys())]
    highest_groups = [get_group(key) for key in sorted(sim_dict.keys())]

    for row_index, row_name in enumerate(entity_names):
        for col_index, col_name in enumerate(entity_names):
            if row_index != col_index:
                sim = 0.0
                try:
                    sim = 1 - sim_dict[row_name][col_name]
                except KeyError:
                    sim = 1 - sim_dict[col_name][row_name]
                finally:
                    sim_matrix[row_index][col_index] = sim
    np.savez(DISTANCE_MAT_FILE_PATH,
             mat=sim_matrix,
             names=entity_names,
             highest_groups=highest_groups)


if __name__ == '__main__':
    main()
