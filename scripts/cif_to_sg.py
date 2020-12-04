import json
import pathlib
from typing import Dict
from zipfile import ZipFile

from pymatgen import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm

ROOT_FOLDER_PATH: pathlib.Path = pathlib.Path(__file__).parent
RAW_DATA_FOLDER_PATH: pathlib.Path = pathlib.Path(ROOT_FOLDER_PATH, 'raw_data')


def get_space_group(crystal: Structure) -> int:
    """
    Gets the space group of a crystal.
    :param crystal: The crystal to get the space group of.
    :return: The space group number
    """
    sp_analyzer = SpacegroupAnalyzer(crystal)
    return sp_analyzer.get_space_group_number()


def save_space_groups(space_groups: Dict[int, int]) -> None:
    """
    Saves the space groups to a json file.
    :param space_groups: The space groups dictionary, cid to space group.
    :return: None. Dict saved to disk as json.
    """
    with open(pathlib.Path(RAW_DATA_FOLDER_PATH, 'space_groups_by_cid.json'), 'w+') as json_file:
        json.dump(space_groups, json_file)


def main():
    space_groups: Dict[int, int] = {}
    with ZipFile(pathlib.Path(RAW_DATA_FOLDER_PATH, 'cifs.zip'), 'r') as read_zip:
        print("Grabbing the space group for each cif. This will take some time.")
        print("The process will be done when the tdqm bar finishes.")
        for file_name in tqdm(read_zip.namelist()):
            with read_zip.open(file_name, 'r') as file:
                crystal = Structure.from_str(file.read().decode("utf-8"), fmt='cif')
                cid = file_name.split('.')[0]
                space_groups[cid] = get_space_group(crystal)
    save_space_groups(space_groups)


if __name__ == '__main__':
    main()
