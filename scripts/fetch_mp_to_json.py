import json
import pathlib
from math import ceil
from typing import List, Sized, Generator

import pandas as pd
from pymatgen import MPRester
from tqdm import tqdm

API_KEY: str = 'INSERT_YOUR_API_KEY'
ROOT_FOLDER_PATH: pathlib.Path = pathlib.Path(__file__).parent
RAW_DATA_FOLDER_PATH: pathlib.Path = pathlib.Path(ROOT_FOLDER_PATH, 'raw_data')
DOWNLOADED_DATA_FOLDER_PATH: pathlib.Path = pathlib.Path(ROOT_FOLDER_PATH, 'downloaded_data')

MPID_CSV_PATH: pathlib.Path = pathlib.Path(RAW_DATA_FOLDER_PATH, 'material_id_hash.csv')
DOWNLOADED_DATA_FILE_PATH: pathlib.Path = pathlib.Path(DOWNLOADED_DATA_FOLDER_PATH,
                                                       'mp_dataset.json')

PROPERTIES_TO_FETCH: List[str] = ['material_id', 'band_gap', 'cif']
BATCH_SIZE: int = 500


def get_mpids(mpids_csv_path: pathlib.Path) -> List[str]:
    """
    Gets the mpids from a weirdly formatted csv. Make sure this is returning what you
    expect.
    :param mpids_csv_path: THe path to the mpids csv file.
    :return: The mpids.
    """
    df = pd.read_csv(mpids_csv_path,
                     delimiter=',',
                     header=None,
                     usecols=[1])
    return df.iloc[:, 0].values[0:1012].tolist()


def batch(iterable: Sized, batch_size: int) -> Generator[List, None, None]:
    """
    Function to batch a sized iterable. Why call it an iterable when sized inherits from iterable?
    Because people are much more familiar with the iterable type and what it means than the sized
    type.
    :param iterable: The iterable to batch.
    :param batch_size: The batch size.
    :return: a batch.
    """
    length = len(iterable)
    for ndx in range(0, length, batch_size):
        yield iterable[ndx:min(ndx + batch_size, length)]


def fetch_data(mpids: List[str], properties: List[str], batch_size: int) -> List[str]:
    """
    Fetchs the data from material project.
    :param mpids: The mpids to fetch.
    :param properties: The properties of the mpids to fetch.
    :param batch_size: The size of batchs to use for the requests.
    :return: None.
    """
    results = []
    num_batchs = ceil(len(mpids) / batch_size)
    with MPRester(API_KEY) as m:
        for batch_of_ids in tqdm(batch(mpids, batch_size), total=num_batchs):
            query = {"material_id": {"$in": batch_of_ids}}
            results.extend(m.query(query, properties))
    return results


def write_data(data: List[str], path: pathlib.Path) -> None:
    """
    Writes the data to file as json.
    :param data: The data to write.
    :param path: The path to the file to create / overwrite with data.
    :return: None.
    """
    with open(path, 'w') as f:
        json.dump(data, f)


def main():
    """
    Fetchs material project materials and saves them in a json file.
    :return: None.
    """
    mpids = get_mpids(MPID_CSV_PATH)
    data = fetch_data(mpids, PROPERTIES_TO_FETCH, BATCH_SIZE)
    write_data(data, DOWNLOADED_DATA_FILE_PATH)


if __name__ == '__main__':
    main()
