import json
from itertools import combinations
from pymatgen.core import Composition
from datetime import datetime
from sklearn.metrics import mean_squared_error

def calculate_similarity(compounds1, compounds2):
    similarity_index = 0.0 # start similarity index at 0 - totally dissimilar

    # convert the stoichiometries to floats
    for i in range(len(compounds1)):
        for j in range(len(compounds1[i]['Stoichiometry'])):
            compounds1[i]['Stoichiometry'][j] = float(compounds1[i]['Stoichiometry'][j])

    for i in range(len(compounds2)):
        for j in range(len(compounds2[i]['Stoichiometry'])):
            compounds2[i]['Stoichiometry'][j] = float(compounds2[i]['Stoichiometry'][j])

    for i in range(len(compounds1)):
        for j in range(len(compounds2)):
            stoichs1 = compounds1[i]['Stoichiometry']
            stoichs2 = compounds2[j]['Stoichiometry']
            true = [[stoichs1[0]], [stoichs1[0]], [stoichs1[1]], [stoichs1[1]]]
            test = [[stoichs2[0]], [stoichs2[1]], [stoichs2[0]], [stoichs2[1]]]
            # check MSE of all pairwise combos of stoichiometries
            mse = mean_squared_error(true, test)

            similarity_index += mse

    return 1 - similarity_index/(len(compounds1) + len(compounds2)) # normalize similarity index and return

def main():
    start_time = datetime.now() #starts timer
    binary_data = json.load(open('binaries_fulllist.json', 'r')) # load in MP data

    write_data = {}

    counter = 0

    # get all unique pairs of elements that appear in dataset
    elem_combos = []
    for comp in binary_data.keys():
        if set(binary_data[comp]['Elements']) not in elem_combos:
            elem_combos.append(set(binary_data[comp]['Elements']))

    # create all unique combinations of phase diagram pairs and loop through
    phase_diagram_combos = combinations(elem_combos,2)

    for p in phase_diagram_combos:
        phase_diagram1 = p[0] # first half of phase diagrams to compare
        phase_diagram2 = p[1] # second half of phase diagrams to compare
        compounds1 = [] # list holding compounds that exist in phase diagram 1
        compounds2 = [] # list holding compounds that exist in phase diagram 2
        # now, go through all of the compounds and add to appropriate list
        for compound in binary_data.keys():
            elem_list = set(binary_data[compound]['Elements'])
            if elem_list == phase_diagram1:
                compounds1.append(binary_data[compound])
            elif elem_list == phase_diagram2:
                compounds2.append(binary_data[compound])

        # check to make sure we found compounds in both phase diagrams
        if len(compounds1) > 0 and len(compounds2) > 0:
            # if compounds are in both phase diagrams, calc the similarity index
            s = calculate_similarity(compounds1, compounds2)
            firstkey = ''.join(phase_diagram1)
            if firstkey not in write_data.keys():
                write_data[firstkey] = {}
            write_data[firstkey][''.join(phase_diagram2)] = s

        counter += 1
        if counter%10000 == 0:
            print(counter)


    json.dump(write_data, open('binary_similarities_mse.json', 'w'))
    endtime = datetime.now() - start_time
    print('Run time: %s'%endtime)

if __name__ == "__main__":
    main()
