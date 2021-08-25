from collections import defaultdict
import sys
import pickle
import os
def find_neighbors(raw_id, simMat, n_neighbors):
    
    allPOIs = simMat[raw_id] # this is also a defaultdict
    POI_sim_list = []
    for eachPOI in allPOIs:
        eachSim = allPOIs[eachPOI]
        POI_sim_list.append((eachPOI,eachSim))
    rankedPOIs = sorted(POI_sim_list, key=lambda tup: tup[1], reverse=True)
    rankedPOIs = rankedPOIs[:n_neighbors]
    
    return rankedPOIs

from collections import defaultdict
import pickle
import os

simMat = defaultdict()
simMat[1] = defaultdict()
simMat[1][2] = 0.35
simMat[1][3] = 0.45
simMat[1][4] = -0.35
simMat[1][5] = 0.22

simMat[2] = defaultdict()
simMat[2][3] = 0.35
simMat[2][4] = -0.15
simMat[2][5] = 0.22

simMat[3] = defaultdict()
simMat[3][4] = -0.65
simMat[3][5] = 0.82

simMat[4] = defaultdict()
simMat[4][5] = -0.05

#fileName = "testFile"
#file = os.path.abspath(__file__+"/..")+ "/" + fileName
#with open(file, 'wb') as handle:
    #pickle.dump(simMat, handle, protocol=pickle.HIGHEST_PROTOCOL)
#with open(file, 'rb') as handle:
    #simMat_copy = pickle.load(handle)
#print(simMat_copy)

def calc_time(func):
    print('a')
    import time
    def wrapper(*args, **kwargs):
        print('b')
        s = time.time()
        v = func(*args, **kwargs)
        print('c')
        e = time.time()

        print("Total execution time for", func.__name__, "is", e-s)
        return v
    print('d')
    return wrapper

@calc_time
def print_1_to_n(n):
    for i in range(1, n+1):
        print(i)

print(sys.path)