import csv
import json
import os
import pathlib
from typing import Tuple, List
from zipfile import ZipFile

import torch
from pymatgen import Structure
from pymatgen.analysis.local_env import CrystalNN
from torch_geometric.data import InMemoryDataset, download_url, extract_zip, Data
from tqdm import tqdm


class CrystalBenchmark(InMemoryDataset):
    """
    The cifs and targets from the multitask crystal graph neural network paper:
    https://arxiv.org/abs/1811.05660.
    """

    def __init__(self):
        """
        I can host the raw url until Tufts yells at me. Much faster than redownloading from
        materials project..
        """
        self.raw_url = ('https://www.eecs.tufts.edu/~gappleby/benchmark.zip')
        self.data_folder_path: pathlib.Path = pathlib.Path(pathlib.Path(__file__).parent,
                                                           'crystal_benchmark')
        super().__init__(root=self.data_folder_path)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """
        Raw file names that need to exist to prevent redownload.
        :return: The list of file names.
        """
        return ['atom_init.json', 'cifs.zip', 'id_prop.csv', 'material_id_hash.csv']

    @property
    def processed_file_names(self):
        """
        Processed file names that need to exist to prevent reprocess.
        :return: The list of file names.
        """
        return ['data.pt']

    def download(self):
        """
        Download the files from the raw_url.
        :return: None
        """
        file_path = download_url(self.raw_url, self.raw_dir)
        extract_zip(file_path, self.raw_dir)
        os.unlink(file_path)

    def process(self):
        """
        Process the files. Currently used the crystalnn approach to generating edges, this seems
        to take a fill 12 hours or so to run on a decent computer. If we need to do this often it
        shouldn't be too much work to parallelize it.

        Constructing adjacency matrix:

            General idea is to run through the all of the cifs.
                Get the neighbors of each 'site' of the crystal
                Then run through each site of the crystal
                    Add an edge between that site and each of its neighbors if that edge does not
                    already exist.

        Construct node features:

            Currently, this is actually just a one hot encoding of each atom stacked together.
            In the past I have also tried using the atom features given from the mt-cgnn paper, but
            did not notice a real difference in performance. Need to revisit this.

        Constructing Targets:
            Taken as given in the mt-cgnn paper.


        :return: None. Saves the processed files.
        """
        raw_atom_embedding = self._get_raw_atom_embedding()
        target_vars_by_id = self._get_target_variables()
        data_list = []
        with ZipFile(pathlib.Path(self.raw_dir, 'cifs.zip'), 'r') as zip:
            for file_name in tqdm(zip.namelist()):
                nodes = []
                edges = []
                with zip.open(file_name, 'r') as file:
                    crystal = Structure.from_str(file.read().decode("utf-8"), fmt='cif')
                    neighbors_by_atom = self._get_neighbors_crystalnn(crystal)
                    cid = file_name.split('.')[0]
                    for idx, site in enumerate(crystal.sites):
                        nodes.append(raw_atom_embedding[site.specie.number])
                        for neighbor in neighbors_by_atom[idx]:
                            edge = [idx, neighbor[0]]
                            if edge not in edges and list(reversed(edge)) not in edges:
                                edges.append(edge)
                    data = Data(x=torch.tensor(nodes, dtype=torch.float),
                                y=torch.tensor([target_vars_by_id[cid]], dtype=torch.float),
                                edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous())
                    data.id = cid
                    data_list.append(data)
        data, slice = self.collate(data_list)
        torch.save((data, slice), self.processed_paths[0])

    def _get_raw_atom_embedding(self):
        """
        Gets the raw atom embeddings however they are defined in atom_init.
        :return: The embedding by atomic number.
        """
        with open(pathlib.Path(self.raw_dir, 'atom_init.json'), 'r') as file:
            data = json.load(file)
        return {int(key): value for key, value in data.items()}

    def _get_target_variables(self):
        """
        Gets the target variables as defined in id_prop.csv.
        :return: The target variables by cif file id.
        """
        with open(pathlib.Path(self.raw_dir, 'id_prop.csv'), 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            target_vars_by_id = {}
            for row in reader:
                target_vars_by_id[row[0]] = list(map(float, row[1:]))
        return target_vars_by_id

    def _get_neighbors_cgcnn(
            self,
            crystal: Structure,
            radius: int,
            max_num_neighbors: int) -> List[List[Tuple]]:
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
                map(lambda nbr: (nbr.index, nbr.nn_distance),
                    sorted(sites_neighbors, key=lambda nbr: nbr.nn_distance)[:max_num_neighbors])))
        return all_neighbors

    def _get_neighbors_crystalnn(self, crystal: Structure) -> List[List[Tuple]]:
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
                [(neighbor['site_index'], neighbor['weight']) for neighbor in neighbors])
        return all_neighbors
