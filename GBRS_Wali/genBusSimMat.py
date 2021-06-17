import pandas as pd
import os
from collections import defaultdict
from numpy import dot
from numpy.linalg import norm

def createPandasDataFrame(fileName):
    inputFile = os.path.abspath(__file__+"/..")+ "\\" + fileName
    print(f"Reading this file: {inputFile}")
    df = pd.read_csv(inputFile)
    return df

def computeCosine(vec1,vec2):
    return dot(vec1, vec2)/(norm(vec1)*norm(vec2))

def toFloat(aStringSeries):
    aFloatList = []
    if not isinstance(aStringSeries, str):
        aStringList = aStringSeries[0][1:-1].split(',')
    else:
        aStringList = aStringSeries[1:-1].split(',')
    for each in aStringList:
        aFloatList.append(float(each))
    return aFloatList

def generateBusSimMat(fileName):

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


def extract_s_train(s,trainset):
    s_train = defaultdict()
    for _, i, _ in trainset.all_ratings():
        s_train[i] = s[trainset.to_raw_iid(i)]
    return s_train

def cal_sum(s_train, i):
    return sum(s_train[i])

def find_POI_neighbors(i, s_train, num_of_neighbors):
    rankedPOI = s_train[i].sort(reverse=True)
    return rankedPOI
    
    
fileName = "aggregatedVectors.csv"


result = generateBusSimMat(fileName)

print(len(result)) 
for x in result:
    print(x, result[x][:3], '...')
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    