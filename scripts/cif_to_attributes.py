import json
import pathlib
from collections import Counter
from json import JSONEncoder
from typing import Dict
from zipfile import ZipFile

import numpy as np
from pymatgen import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm

ROOT_FOLDER_PATH: pathlib.Path = pathlib.Path(__file__).parent
RAW_DATA_FOLDER_PATH: pathlib.Path = pathlib.Path(ROOT_FOLDER_PATH, 'raw_data')


class AttributesEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


class Attributes:
    def __init__(self,
                 space_group: int,
                 formula: str,
                 most_prevalent_atomic_group: int,
                 average_bond_distance: float):
        super().__init__()
        self.space_group = space_group
        self.formula = formula
        self.most_prevalent_atomic_group = most_prevalent_atomic_group
        self.average_bond_distance = average_bond_distance


def get_space_group(crystal: Structure) -> int:
    """
    Gets the space group of a crystal.
    :param crystal: The crystal to get the space group of.
    :return: The space group number.
    """
    sp_analyzer = SpacegroupAnalyzer(crystal)
    return sp_analyzer.get_space_group_number()


def get_composition_formula(crystal: Structure) -> str:
    """
    Gets the formula of the composition of the crystal.
    :param crystal: The crystal to get the composition formula of.
    :return: The composition formula.
    """
    return crystal.composition.formula


def get_most_prevalent_atomic_group(crystal: Structure):
    """
    Gets the most prevalent atomic group. In this case we look at the table group
    of each element (including repeats), and then find the most common group.
    :param crystal: The crystal to get the most prevalent atomic group of.
    :return: The prevalent atomic group.
    """
    return Counter([g.group for g in crystal.species]).most_common(1)[0][0]


def get_avg_bond_distance(structure, bond_tolerance=0.03):
    """
    Method to compute average bond distance of all first nearest neighbors a
    Structure
    :param structure: pymatgen structure object, typically read from cif
    :param bond_tolerance: tolerance for deviations in min bond distance in A
    """
    avg_bond_distance = []  # start with avg bond distance of 0
    distance_matrix = structure.distance_matrix  # get matrix of all distances
    # go over every atom and figure out the minimum bond distance
    if len(distance_matrix) < 2:
        return 0
    for bonds in distance_matrix:
        # get minimum bond distances that aren't 0
        min_bond = np.min(bonds[np.nonzero(bonds)])
        # Check to make sure there isn't more than one bond at this min length!
        # include a tolerance factor because crystals aren't perfect
        for b in bonds:
            if b <= min_bond + bond_tolerance and b > 0.0:
                avg_bond_distance.append(b)
    return np.mean(avg_bond_distance)


def save_attributes(attributes_by_id: Dict[int, Attributes]) -> None:
    """
    Saves the attributes by id to a json file.
    :param attributes_by_id: The attributes dictionary, cid to attributes.
    :return: None. Dict saved to disk as json.
    """
    with open(pathlib.Path(RAW_DATA_FOLDER_PATH, 'attributes_by_cid.json'), 'w+') as json_file:
        json.dump(attributes_by_id, json_file, cls=AttributesEncoder)


def main():
    attributes_by_id: Dict[int, Attributes] = {}
    with ZipFile(pathlib.Path(RAW_DATA_FOLDER_PATH, 'cifs.zip'), 'r') as read_zip:
        print("Grabbing the properties for each cif. This will take some time.")
        print("The process will be done when the progress bar finishes.")
        for file_name in tqdm(read_zip.namelist()):
            with read_zip.open(file_name, 'r') as file:
                crystal = Structure.from_str(file.read().decode("utf-8"), fmt='cif')
                cid = file_name.split('.')[0]
                space_group = get_space_group(crystal)
                formula = get_composition_formula(crystal)
                most_prevalent_atomic_group = get_most_prevalent_atomic_group(crystal)
                average_bond_distance = get_avg_bond_distance(crystal)
                attributes_by_id[cid] = Attributes(space_group,
                                                   formula,
                                                   most_prevalent_atomic_group,
                                                   average_bond_distance)
    save_attributes(attributes_by_id)


if __name__ == '__main__':
    main()
