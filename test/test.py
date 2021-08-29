import pandas as pd
import numpy as np
import os
import re
import math
from operator import add
from collections import defaultdict
from sklearn.cluster import SpectralClustering
import time
def columnImpute(df):
    pdf = df.pivot(index='user_id', columns = 'bus_id', values = 'rating')
    pdf_mean = pdf.mean(axis = 0)
    pdf = pdf.fillna(value = pdf_mean, axis = 0)
    df = pdf.T.unstack().reset_index(name='rating')
    return df

df = pd.DataFrame({'user_id':    ['zzz0','zzz0','zzz1','zzz1','zzz1','zzz2','zzz2','zzz3','zzz4','zzz4','zzz5','zzz6'],
                   'bus_id':     [0,1,0,2,4,1,3,7,0,5,8,7],
                   'rating':     [3,2,1,2,3,4,5,4,3,3,2,1],
                   'rating2':    [3,2,1,2,3,4,5,4,3,3,2,1],
                   'lat':        [1,2,1,4,5,2,7,8,1,10,11,7],
                   'lon':        [12,11,12,9,8,11,6,5,12,3,2,6]
                   })                 

#df = columnImpute(df)
#pdf = df.pivot(index='user_id', columns = 'bus_id', values = 'rating')
#model = SpectralClustering(assign_labels='discretize', n_clusters=3, eigen_solver = 'amg', random_state=0)
#pdf.fillna(0, inplace=True)
#model.fit_predict(pdf)
#rint(model.labels_)
#nppdf = pdf.to_numpy()
#print(nppdf)


def cluster_spectral(curr_df, Xth_batch, clusters_per_batch):

    global_mean = curr_df.loc[:,'rating'].mean()
    dfMatrix = curr_df.pivot(index='user_id', columns = 'bus_id', values = 'rating') 
    for uid,_ in dfMatrix.iterrows():
        for iid in dfMatrix:
            if math.isnan(dfMatrix.at[uid,iid]):
                dfMatrix.at[uid,iid] = dfMatrix.mean(axis = 1)[uid] + dfMatrix.mean(axis = 0)[iid] - global_mean

    #=============================================== Finished Imputation =============================================
    userList = []
    latList  = []
    lonList  = []
    for eachUser in curr_df.user_id:
        lat_mean = curr_df.loc[curr_df['user_id'] == eachUser].lat.mean()
        lon_mean = curr_df.loc[curr_df['user_id'] == eachUser].lon.mean()
        userList.append(eachUser)
        latList.append(lat_mean)
        lonList.append(lon_mean)
    print(latList)
    locDf = pd.DataFrame({'user_id':userList,
                                'lat': latList,
                                'lon': lonList 
                                })
    
    locDf = locDf.drop_duplicates().reset_index(drop=True)
    #============================================== Finished Location Simulation==========================================

    user1s  = []
    user2s  = []
    GPSsims = []
    Rsims   = []
    for index1, row1 in locDf.iterrows():
        for index2, row2 in locDf.iterrows():
            A = [row1['lat'],row1['lon']]
            B = [row2['lat'],row2['lon']]
            sim = np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))
            user1s.append(row1['user_id'])
            user2s.append(row2['user_id'])
            GPSsims.append(sim)

    for user1 in dfMatrix.index:
        for user2 in dfMatrix.index:
            A = dfMatrix.loc[user1].values.tolist()
            B = dfMatrix.loc[user2].values.tolist()
            sim = np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))
            Rsims.append(sim)
    ratio = 0.5
    sims1 = [i*ratio for i in GPSsims]
    sims2 = [i*(1-ratio) for i in Rsims]
    combinedSims = list(map(add, sims1, sims2))

    simMat = pd.DataFrame({'user_id_R': user1s,
                           'user_id_C': user2s,
                                 'sim': combinedSims 
                           })
    simMat = simMat.pivot( index='user_id_R', columns = 'user_id_C', values = 'sim') 
     
    fileName = "./PickleJar" +str(Xth_batch) + "thSimMat.test"
    #print(os.path.exists(fileName))
    simMat.to_pickle(fileName)
    return simMat
def loadSim(curr_df, Xth_batch, clusters_per_batch):
 #============================================== Finished Calculate Sims ==========================================
    fileName = "./PickleJar" +str(Xth_batch) + "thSimMat.test"
    if (os.path.exists(fileName)):
        simMat = pd.read_pickle(fileName)
    else:
        simMat = cluster_spectral(curr_df, Xth_batch, clusters_per_batch)

    simArray = np.array(simMat)
    num_clusters = clusters_per_batch
    sc = SpectralClustering(num_clusters, affinity='precomputed')
    sc.fit(simArray)
    #================================================ Finished Clustering ==============================================
    groupIDs = []
    for eachLabel in sc.labels_:
        groupIDs.append(eachLabel + Xth_batch* 1000000)

    toGroupId = defaultdict()
    for i in range(len(groupIDs)):
        toGroupId[dfMatrix.index.values[i]] = groupIDs[i]

    originalIdList = curr_df.user_id
    outputIdList   = []
    for eachId in originalIdList:
        outputIdList.append(toGroupId[eachId])

    curr_df.user_id = outputIdList
    #================================================ Finished Modifying Original DF ========================================   
    return curr_df
#========================================================================================================================
#cluster_spectral(df,1,3)

#print(result)
#df111 = pd.DataFrame({'user_id111':    [1,1,1,1,1,1,1,1,1,1,1,1]})
print(str(int(time.time())))