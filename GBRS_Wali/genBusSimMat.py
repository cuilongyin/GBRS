import csv
import sys
import os
import pandas as pd
from collections import defaultdict
from numpy import dot
from numpy.linalg import norm
import pickle
import cProfile

def createPandasDataFrame(fileName):
    inputFile = os.path.abspath(__file__+"/..")+ "\\" + fileName
    print(f"Reading this file: {inputFile}")
    df = pd.read_csv(inputFile)
    return df
#this function generates four fucntions, this is how people usually translate IDs from raw to inner
#I explained what is raw and inner ID in later sections.
def genFourDics(fileName):# this applies to windows, if you are using a linex or Mac, change next line
    rPath = os.path.abspath(__file__+"/..")+ "\\" + fileName # <---- this is what I mean
    ratingsPath = rPath
    uid = []
    iid = []
    with open(ratingsPath, newline = '', encoding = 'ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # pay attention to this row!!!!
        for row in reader:
            uid.append(row[0])
            iid.append(row[1])
    to_inner_uid = defaultdict()
    to_inner_iid = defaultdict()
    to_outer_uid = defaultdict()
    to_outer_iid = defaultdict()
    #================== uid ==============================
    for eachID in uid:
        if eachID not in to_inner_uid:
            innerID = len(to_inner_uid)
            to_outer_uid[innerID] = eachID
            to_inner_uid[eachID] = innerID
    #================== iid ==============================
    for eachID in iid:
        if eachID not in to_inner_iid:
            innerID = len(to_inner_iid)
            #print(innerID)
            to_outer_iid[innerID] = eachID
            to_inner_iid[eachID] = innerID  
            #  
    return to_inner_uid, to_outer_uid, to_inner_iid, to_outer_iid        

def computeCosine(vec1,vec2):
    return dot(vec1, vec2)/(norm(vec1)*norm(vec2))

def toFloat(aStringSeries):
    aFloatList = []
    if not isinstance(aStringSeries, str):
        aStringList = aStringSeries
        aStringList = aStringSeries[0][1:-1].split(',')
    else:
        aStringList = aStringSeries[1:-1].split(',')
    for each in aStringList:
        aFloatList.append(float(each))
    return aFloatList


def originalGenFunc(fileName):
    df = createPandasDataFrame(fileName)
    result = defaultdict()
    for bus in df["bus_id"]:
        currVec = df.loc[df["bus_id"] == bus]['vector'].values
        currVec = toFloat(currVec)
        sims = []
        for vec in df["vector"]:
            vec = toFloat(vec)
            sims.append(computeCosine(currVec, vec))
        result[bus]=sims
    return result

def generateBusSimMat(fileName):

    df = createPandasDataFrame(fileName)

    result = defaultdict()
    for _, row1 in df.iterrows():
        currBus = row1.bus_id
        result[currBus] = defaultdict()
        currVec = row1.vector
        currVec = toFloat(currVec)

        for _, row2 in df.iterrows(): 
            targVec = row2.vector
            targVec = toFloat(targVec)
            sim = computeCosine(currVec, targVec)
            targBus = row2.bus_id
            result[currBus][targBus] = sim

    return result

def writeToFile(result, writePath):
    names = []
    names.append('bus_id')
    for eachBus in result:
        names.append(eachBus)
    with open(writePath,'w', newline = '', encoding = 'ISO-8859-1') as writefile:  
        writer = csv.writer(writefile, sys.stdout, lineterminator = '\n')
        writer.writerow(names)
        for eachBus in result:
            rowToWrite = []
            rowToWrite.append(eachBus)
            for eachSim in result[eachBus]:
                rowToWrite.append(eachSim)
            writer.writerow(rowToWrite) 

def readSimMat(matFilePath):
    IndexToBus = defaultdict()
    simMat     = defaultdict()
    with open(matFilePath, 'r', newline = '', encoding = 'ISO-8859-1') as readfile:
        reader = csv.reader(readfile)
        for row in reader:
            n = 1
            for eachBus in row[1:]:
                IndexToBus[n] = eachBus
                n += 1
            break
        for row in reader:
            simMat[row[0]] =defaultdict() 
            for i in range(1, len(row)):
                simMat[row[0]][IndexToBus[i]] = row[i]
    return simMat 

#currently useless ==============================
def extract_s_train(s,trainset):
    s_train = defaultdict()
    for _, i, _ in trainset.all_ratings():
        s_train[i] = s[trainset.to_raw_iid(i)]
    return s_train
#================================================

def find_neighbors(raw_id, simMat, n_neighbors):
    
    allPOIs = simMat[raw_id] # this is also a defaultdict
    POI_sim_list = []
    for eachPOI in allPOIs:
        eachSim = allPOIs[eachPOI]
        POI_sim_list.append((eachPOI,eachSim))
    rankedPOIs = sorted(POI_sim_list, key=lambda tup: tup[1], reverse=True)
    rankedPOIs = rankedPOIs[:n_neighbors]
    
    return rankedPOIs


def cal_sum_sij(iNeighbors):
    _sum = 0
    for eachNeighbor in iNeighbors:
        _sum += eachNeighbor[1]
    return _sum




fileName = "aggregatedVectors.csv"
writePath = os.path.dirname(os.path.realpath(__file__)) + "\\simMat.bin"
result = generateBusSimMat(fileName)
#writeToFile(result,writePath)
#matFilePath = os.path.dirname(os.path.realpath(__file__)) + "/simMat.csv"
#busSimMat = readSimMat(matFilePath)
#file = os.path.abspath(__file__+"/..")+ "/" + fileName
#print(#result)
writePath = os.path.dirname(os.path.realpath(__file__)) + "\\simMat.bin"
with open(writePath, 'wb') as handle:
    pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open(writePath, 'rb') as handle:
    busSimMat = pickle.load(handle)
print(busSimMat)
print(result == busSimMat)
#cProfile.run('generateBusSimMat(fileName)')
#print(len(result)) 
#for x in result:
    #print(x, result[x][:3], '...')
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    