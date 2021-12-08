import json
from collections import defaultdict
from datetime import datetime
from itertools import combinations

from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error
from statistics import mean

from tqdm import tqdm


def calculate_distance(name, compounds1, compounds2):
    mses = []

    # convert the stoichiometries to floats
    for i in range(len(compounds1)):
        for j in range(len(compounds1[i])):
            compounds1[i][j] = float(compounds1[i][j])

    for i in range(len(compounds2)):
        for j in range(len(compounds2[i])):
            compounds2[i][j] = float(compounds2[i][j])

    for i in range(len(compounds1)):
        for j in range(len(compounds2)):
            stoichs1 = compounds1[i]
            stoichs2 = compounds2[j]
            true = [[stoichs1[0]], [stoichs1[0]], [stoichs1[1]], [stoichs1[1]]]
            test = [[stoichs2[0]], [stoichs2[1]], [stoichs2[0]], [stoichs2[1]]]

            # check MSE of all pairwise combos of stoichiometries
            mses.append(mean_squared_error(true, test))

    return {name[0]: {name[1]: mean(mses)}}


def main():
    start_time = datetime.now()  # starts timer
    binary_data = json.load(open('binaries_fulllist.json', 'r'))  # load in MP data

    # get all unique pairs of elements that appear in dataset
    elem_combos = defaultdict(lambda: [])
    for comp_name in binary_data.keys():
        comp = binary_data[comp_name]
        elem_combos[''.join(comp['Elements'])].append(comp['Stoichiometry'])

    # create all unique combinations of phase diagram pairs and loop through
    phase_diagram_combos = list(combinations(elem_combos.keys(), 2))

    dist_dicts = Parallel(n_jobs=-1)(
        delayed(calculate_distance)(c, elem_combos[c[0]], elem_combos[c[1]]) for c in tqdm(phase_diagram_combos))

    write_data = defaultdict(lambda: {})
    for dist_dict in dist_dicts:
        for key in dist_dict.keys():
            write_data[key].update(dist_dict[key])

    json.dump(write_data, open('binary_distances.json', 'w'))
    endtime = datetime.now() - start_time
    print('Run time: %s' % endtime)


if __name__ == "__main__":
    main()
