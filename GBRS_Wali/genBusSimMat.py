import csv
import sys
import os
import pandas as pd
from collections import defaultdict
from numpy import dot
from numpy.linalg import norm

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
# This function conferts a dataframe from raw IDs to inner IDs.
# The inputs are a dataframe of raw IDs and a string name of the data file
def convertToInner(df_input, fileName):
    df = df_input
    to_inner_uid, to_outer_uid, to_inner_iid, to_outer_iid = genFourDics(fileName)
    #['user_id', 'bus_id', 'rating', 'date','lat','lon','text']
    #userId,movieId,rating,timestamp
    uids = df['userId']
    inner_uid = []
    for eachOuter in uids:
        inner_uid.append(to_inner_uid[str(eachOuter)])
    iids = df['movieId']
    inner_iid = []
    for eachOuter in iids:
        inner_iid.append(to_inner_iid[str(eachOuter)])
    d = {'userId':inner_uid, 'movieId': inner_iid}
    df1 = pd.DataFrame(data=d)
    df['userId'] = df1['userId']
    df['movieId']  = df1['movieId']
    return df
#This function conferts a dataframe from inner IDs to raw IDs.
#The inputs are same things compared to the above one.
def convertToOuter(df_input, fileName):
    df = df_input
    to_inner_uid, to_outer_uid, to_inner_iid, to_outer_iid = genFourDics(fileName)
    #['user_id', 'bus_id', 'rating', 'date','lat','lon','text']
    uids = df['user_id']
    outer_uid = []
    for eachInner in uids:
        if eachInner in to_outer_uid:
            outer_uid.append(to_outer_uid[eachInner])
        else:
            outer_uid.append(eachInner)
    iids = df['bus_id']
    print(type(iids))
    outer_iid = []
    for eachInner in iids:
        outer_iid.append(to_outer_iid[eachInner])
    d = {'user_id':outer_uid, 'bus_id': outer_iid}
    df1 = pd.DataFrame(data=d)
    df['user_id'] = df1['user_id']
    df['bus_id']  = df1['bus_id']
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

def writeToFile(result, writePath):

    with open(writePath,'w', newline = '', encoding = 'ISO-8859-1') as writefile:
        writer = csv.writer(writePath, sys.stdout, lineterminator = '\n')
        writer.writerow([ 'bus_id','bus_id','sims'] )
        for eachBus0 in result:
            for eachBus1 in result[eachBus0]:
                writer.writerow([eachBus, result]) 

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
writePath = os.path.dirname(os.path.realpath(__file__)) + "\\test.csv"






#cProfile.run('generateBusSimMat(fileName)')
#print(len(result)) 
#for x in result:
    #print(x, result[x][:3], '...')
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    