import json
from itertools import combinations
import pandas as pd
from pymatgen.core import Element, Composition
from datetime import datetime

def remove_null_compositions(A,B):
    '''
    Method that checks two columns of composition vector CSV and returns only
    rows from the CSV with those columns in it
    '''
    final_comps=[]

    # Go over every row in the dataset
    for i in range(len(A)):
        # Check - does this row of the csv have elements A and B in it only?
        if A[i] + B[i] == 1:
            final_comps.append([A[i], B[i]])
    return final_comps

def calculate_similarity(PD1, PD2):
    similarity_index = 0.0 # start similarity index at 0
    total_num_compounds = len(PD1) + len(PD2) # get number of compunds in the phase diagrams
    for i in range(len(PD1)):
        # check if there are matching stoichiometries in the phase diagrams
        if PD1[i] in PD2 or list(reversed(PD1[i])) in PD2:
            similarity_index += 2 # add 2 to similarity_index

    return similarity_index/total_num_compounds # normalize similarity index and return

def main():
    start_time = datetime.now() #starts timer
    df = pd.read_csv('binary_comp_vectors_nodups.csv') # reads data
    elements = list(df.columns) # get all column headers
    print('Starting with %s elements'%len(elements))

    # drop any columns that are entirely 0
    drop_columns = []
    for e in elements:
        if (df[e] == 0).all():
            drop_columns.append(e)
    df = df.drop(columns = drop_columns)
    elements = list(df.columns)
    print('Ending with %s elements'%len(elements))

    print('generating phase diagrams')
    phase_diagrams = list(combinations(elements,2)) # lists all combination of 2 elements
    print('generated %s phase diagrams'%len(phase_diagrams))
    print('generating comparisons')
    comb = list(combinations(phase_diagrams,2)) # lists combinations of phase diagrams
    phase_dia = {} # dictionary to hold similarity index for phase diagrams

    print('entering compare loop. Going through %s comparisons.'%len(comb))
    count = 0
    for d in comb:
        # get columns of original data at each given element in a phase diagram
        A = df[d[0][0]]
        B = df[d[0][1]]
        C = df[d[1][0]]
        D = df[d[1][1]]

        count += 1
        if count%10000 == 0:
            print('count is %s'%count)

         # Before calling method - check if there all columns have nonzero data
        PD1 = remove_null_compositions(A,B)
        PD2 = remove_null_compositions(C,D)

        if len(PD1) > 0 and len(PD2) > 0:
            sim = calculate_similarity(PD1, PD2)

            a = A.name
            b = B.name
            c = C.name
            d = D.name
            if a + b not in phase_dia.keys():
                phase_dia[a + b] = {}
                phase_dia[a + b][c + d] = sim
            else:
                phase_dia[a + b][c + d] = sim

    print('Runtime for code: %s'%datetime.now() - start_time)

    # write dictionary to JSON
    with open('binary_similarities.json', 'w') as outfile:
        json.dump(phase_dia, outfile)
    print('Runtime with writing: %s'%datetime.now() - start_time)

if __name__ == "__main__":
    main()
