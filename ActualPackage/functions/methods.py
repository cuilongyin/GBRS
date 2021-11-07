import os
import sys
import csv
import math
import time
import pickle
import platform
import numpy as np
import pandas as pd
from fcmeans import FCM
from operator import add
from surprise import Reader
from surprise import Dataset
from surprise import accuracy
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from surprise.model_selection import train_test_split

import warnings
from sklearn.exceptions import DataConversionWarning
# In[596]:


def createPandasDataFrame(fileName):
    inputFile = os.path.abspath(__file__+"/../../")+ "/PackageTestGround/" + fileName
    print(f"Reading this file: {inputFile}")
    return pd.read_csv(inputFile)


# In[597]:


def filiterYear(df, startYear): # startYear is an integer
    print("Sorting data ...")
    df = df.sort_values(by = 'date')
    print("Done.")
    for i in range(2004, startYear):
        df = df[~df.date.str.contains(str(i))]
    df = df.reset_index(drop = True)
    #print(df)
    return df


# In[598]:


def baseImpute(df):
    global_mean = df.loc[:,'rating'].mean()
    pdf = df.pivot(index='user_id', columns = 'bus_id', values = 'rating')
    for uid,_ in pdf.iterrows():
        for iid in pdf:
            if math.isnan(pdf.at[uid,iid]):
                pdf.at[uid,iid] = pdf.mean(axis = 1)[uid]                                + pdf.mean(axis = 0)[iid]                                - global_mean
    return pdf.T.unstack().reset_index(name='rating')


# In[599]:


def columnImpute(df):
    pdf = df.pivot(index='user_id', columns = 'bus_id', values = 'rating')
    pdf_mean = pdf.mean(axis = 0)
    pdf = pdf.fillna(value = pdf_mean, axis = 0)
    df = pdf.T.unstack().reset_index(name='rating')
    return df


# In[600]:


def UserImpute(df):
    df_mean = df.mean(axis = 1)
    df = df.transpose()
    df = df.fillna(value = df_mean, axis = 0)
    return df.transpose()


# In[601]:


def removeUsers(df, min_NO_ratings):
    print("Removing unqualified users ...")
    dups = df.pivot_table(index=['user_id'], aggfunc='size')
    
    for user in df['user_id']:
        if dups[user] < min_NO_ratings:
            df = df[df.user_id != user] 
    print("Done.")
    return df


# In[602]:


def creatingXthBatch_clustered(df, batch_size, Xth_batch, cluster_size, method, ratio, pickleJarName): #1 based, Do not put 0 or below
    if Xth_batch <= 0:
        raise Exception("1-based, DO NOT put 0 or below")
    
    if len(df.index) > (batch_size*Xth_batch):
        curr_df = df.iloc[(batch_size*(Xth_batch-1)):(batch_size*Xth_batch)]
    else:
        curr_df = df.iloc[(batch_size*(Xth_batch-1)):(len(df.index))]
        print(f"test set not enough, only {len(curr_df.index)} left")
    
    if method == 'kmean' :
        clustered = cluster_KMean_userRating(curr_df, Xth_batch, cluster_size)
    elif method == 'spectral_ratingGPS' :
        clustered = cluster_ratingGPS_part3(curr_df, Xth_batch, cluster_size, ratio, pickleJarName)
    elif method == 'spectral_pure' :
        clustered = cluster_spectral_pure(curr_df, Xth_batch, cluster_size)
    elif method == 'cluster_DBSCAN' :
        clustered = cluster_DBSCAN(curr_df, Xth_batch, cluster_size)
    elif method == 'cluster_FCM' :
        clustered = cluster_FCM(curr_df, Xth_batch, cluster_size)
    else:
        return curr_df
    return clustered

# In[603]:


def creatingXthBatch_unClustered(df, batch_size, Xth_batch): #1 based, Do not put 0 or below
    if Xth_batch <= 0:
        raise Exception("1-based, DO NOT put 0 or below")
    if len(df.index) > (batch_size*Xth_batch):
        curr_df = df.iloc[(batch_size*(Xth_batch-1)):(batch_size*Xth_batch)]
    else:
        curr_df = df.iloc[(batch_size*(Xth_batch-1)):(len(df.index))]
        print(f"test set not enough, only {len(curr_df.index)} left")
    return curr_df


# In[604]:
def workOnBatchDics(df, batchDic_cluster, batchDic_unCluster, Xth_batch, batch_size, cluster_size, method, ratio, pickleJarName):
    batchDic_cluster[Xth_batch]   = creatingXthBatch_clustered(df, batch_size, Xth_batch, cluster_size, method, ratio, pickleJarName)
    batchDic_unCluster[Xth_batch] = creatingXthBatch_unClustered(df, batch_size, Xth_batch)
    return batchDic_cluster, batchDic_unCluster


def createTrainDf_clustered(df, batchDic_cluster, batchDic_unCluster, batch_size, NOofBatches, cluster_size, method, windowSize, ratio, pickleJarName):
    trainList = []
    startFrom = max(NOofBatches - windowSize, 1)
    for i in range(startFrom, NOofBatches+1):
        if i not in batchDic_cluster:
            batchDic_cluster, batchDic_unCluster = workOnBatchDics(df, batchDic_cluster, batchDic_unCluster, i, batch_size, cluster_size, method, ratio, pickleJarName)
            #print(batchDic_unCluster[1].bus_id)
        trainList.append(batchDic_cluster[i])
    trainSet = pd.concat(trainList)   
    trainSet = trainSet.reset_index(drop=True)
    return trainSet


# In[605]:


def createTrainDf_unClustered(batchDic_unCluster, NOofBatches, windowSize):

    startFrom = max(NOofBatches - windowSize, 1)
    trainList = [batchDic_unCluster[i] for i in range(startFrom, NOofBatches+1)]
    #for x in trainList:
    #print(trainList[0])
    return pd.concat(trainList)


# In[606]:


def cluster_KMean_userRating(df, Xth_batch, clusters_per_batch):
    df = columnImpute(df)
    pdf = df.pivot(index='user_id', columns = 'bus_id', values = 'rating') 
    columnNames = pdf.columns
    model = KMeans(n_clusters = clusters_per_batch,random_state=0)
    model.fit_predict(pdf)
    clusters = pd.DataFrame(model.cluster_centers_)
    clusters.columns= columnNames
    df = clusters.T.unstack().reset_index(name='rating')
    df.rename(columns={'level_0': 'user_id'}, inplace=True)
    df['user_id'] = df['user_id'] + 100000*Xth_batch # this is to make each centroids' ID special
                                                    # So this batch's IDs to mess with the next one's'

    return df

def cluster_spectral_pure(df, Xth_batch, clusters_per_batch):
    print(Xth_batch)
    scale = 100000 # use this to accentuate the difference among different users. 
    pdf = df.pivot(index='user_id', columns = 'bus_id', values = 'rating') 
    global_mean = df.loc[:,'rating'].mean()
    
    bu = defaultdict()
    bi = defaultdict()

    for uid,_ in pdf.iterrows():
        if uid not in bu:
            bu[uid] = pdf.mean(axis = 1)[uid]

        for iid in pdf:
            if iid not in bi:
                bi[iid] = pdf.mean(axis = 0)[iid]

            if math.isnan(pdf.at[uid,iid]):
            #if dfMatrix.at[uid,iid].isnan():
                pdf.at[uid,iid] = bu[uid] + bi[iid] - global_mean

    pdf = pdf.multiply(scale)

    nppdf = pdf.to_numpy()
    model = SpectralClustering(assign_labels='discretize', n_clusters=clusters_per_batch, eigen_solver = 'amg', random_state=0)

    warnings.filterwarnings(action='ignore', category=Warning)
    model.fit_predict(nppdf)
    pdf = pdf.multiply(1/scale)
    #print(model.labels_)

    groupIDs = []
    for eachLabel in model.labels_:
        groupIDs.append(eachLabel + Xth_batch* 1000000)

    toGroupId = defaultdict()
    for i in range(len(groupIDs)):
        toGroupId[pdf.index.values[i]] = groupIDs[i]

    originalIdList = df.user_id
    outputIdList   = []
    for eachId in originalIdList:
        outputIdList.append(toGroupId[eachId])

    df = df[df.rating != -1]
    df = df.assign(user_id=outputIdList)
    
    return df

def cluster_FCM(df, Xth_batch, clusters_per_batch):
    print(Xth_batch)
    
    scale = 100000 # use this to accentuate the difference among different users. 
    pdf = df.pivot(index='user_id', columns = 'bus_id', values = 'rating') 
    columnNames = pdf.columns
    global_mean = df.loc[:,'rating'].mean()
    
    bu = defaultdict()
    bi = defaultdict()

    for uid,_ in pdf.iterrows():
        if uid not in bu:
            bu[uid] = pdf.mean(axis = 1)[uid]

        for iid in pdf:
            if iid not in bi:
                bi[iid] = pdf.mean(axis = 0)[iid]

            if math.isnan(pdf.at[uid,iid]):
            #if dfMatrix.at[uid,iid].isnan():
                temp = bu[uid] + bi[iid] - global_mean
                if temp < 0 :
                    temp = 0
                if temp >5:
                    temp = 5
                pdf.at[uid,iid] = temp

    #pdf = pdf.multiply(scale)
    #print(pdf)
    nppdf = pdf.to_numpy()
    model = FCM(n_clusters=clusters_per_batch)
    model.fit(nppdf)

    #pdf = pdf.multiply(1/scale)

    fcm_centers = model.centers
    fcm_labels = model.predict(nppdf)
    #print(fcm_labels)
    #print(fcm_centers)
    clusters = pd.DataFrame(fcm_centers)
    clusters.columns= columnNames
    df = clusters.T.unstack().reset_index(name='rating')
    df.rename(columns={'level_0': 'user_id'}, inplace=True)
    df['user_id'] = df['user_id'] + 100000*Xth_batch
    return df

def cluster_DBSCAN(df, Xth_batch, clusters_per_batch):
    print(Xth_batch)
    scale = 100000 # use this to accentuate the difference among different users. 
    pdf = df.pivot(index='user_id', columns = 'bus_id', values = 'rating') 
    global_mean = df.loc[:,'rating'].mean()
    bu = defaultdict()
    bi = defaultdict()
    for uid,_ in pdf.iterrows():
        if uid not in bu:
            bu[uid] = pdf.mean(axis = 1)[uid]
        for iid in pdf:
            if iid not in bi:
                bi[iid] = pdf.mean(axis = 0)[iid]

            if math.isnan(pdf.at[uid,iid]):
                pdf.at[uid,iid] = bu[uid] + bi[iid] - global_mean
    pdf = pdf.multiply(scale)
    nppdf = pdf.to_numpy()
    #model = DBSCAN(min_samples=clusters_per_batch)
    model = DBSCAN(eps = 1000, min_samples=2)
    model.fit_predict(nppdf)
    print(model.labels_)
    pdf = pdf.multiply(1/scale)
    groupIDs = [eachLabel + Xth_batch* 1000000 for eachLabel in model.labels_]
    toGroupId = defaultdict()
    for i in range(len(groupIDs)):
        toGroupId[pdf.index.values[i]] = groupIDs[i]
    originalIdList = df.user_id
    outputIdList = [toGroupId[eachId] for eachId in originalIdList]
    df = df[df.rating != -1]
    df = df.assign(user_id=outputIdList)
    
    return df

def cluster_ratingGPS_part1(curr_df, Xth_batch, clusters_per_batch, pickleJarName):

    print(f"--- Performing {Xth_batch}th batch clustering (spectral) ---")
    global_mean = curr_df.loc[:,'rating'].mean()
    dfMatrix = curr_df.pivot(index='user_id', columns = 'bus_id', values = 'rating')
    bu = defaultdict()
    bi = defaultdict()

    for uid,_ in dfMatrix.iterrows():
        if uid not in bu:
            bu[uid] = dfMatrix.mean(axis = 1)[uid]

        for iid in dfMatrix:
            if iid not in bi:
                bi[iid] = dfMatrix.mean(axis = 0)[iid]

            if math.isnan(dfMatrix.at[uid,iid]):
            #if dfMatrix.at[uid,iid].isnan():
                dfMatrix.at[uid,iid] = bu[uid] + bi[iid] - global_mean

    #======================= Finished Imputation =======================

    userList = []
    latList  = []
    lonList  = []
    for eachUser in curr_df.user_id:
        lat_mean = curr_df.loc[curr_df['user_id'] == eachUser].lat.mean()
        lon_mean = curr_df.loc[curr_df['user_id'] == eachUser].lon.mean()
        userList.append(eachUser)
        latList.append(lat_mean)
        lonList.append(lon_mean)
    locDf = pd.DataFrame({'user_id': userList,
                          'lat'    : latList ,
                          'lon'    : lonList 
                          })
    
    locDf = locDf.drop_duplicates().reset_index(drop=True)
    #============================================== Finished Location Simulation==========================================
    #: GPS similarities calculation-----------------------------------------------------------
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

    #: rating similarities calculation--------------------------------------------------------
    for user1 in dfMatrix.index:
        for user2 in dfMatrix.index:
            A = dfMatrix.loc[user1].values.tolist()
            B = dfMatrix.loc[user2].values.tolist()
            sim = np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))
            Rsims.append(sim)

    #saving to pickle file 
    part1Package = {
        'user1s':user1s,
        'user2s':user2s,
        'GPSsims':GPSsims,
        'Rsims':Rsims
        }

    #pickleJarName
    #fileName = "./PickleJar/" + str(Xth_batch) + "thPart1Package.pkl"
    fileName = pickleJarName +str(Xth_batch) + "thPart1Package.pkl"
    with open(fileName, 'wb') as pDump:
        pickle.dump(part1Package, pDump)

    return part1Package

def cluster_ratingGPS_part2(curr_df, Xth_batch, clusters_per_batch, ratio, pickleJarName):

    fileName = pickleJarName +str(Xth_batch) + "thPart1Package.pkl"
    if (os.path.exists(fileName)):
        with open(fileName, 'rb') as pDump:
            part1Package = pickle.load(pDump)
    else:
        part1Package = cluster_ratingGPS_part1(curr_df, Xth_batch, clusters_per_batch, pickleJarName) 
    user1s = part1Package['user1s']
    user2s = part1Package['user2s']
    GPSsims = part1Package['GPSsims']
    Rsims = part1Package['Rsims']
    sims1 = [i*ratio for i in GPSsims]
    sims2 = [i*(1-ratio) for i in Rsims]
    combinedSims = list(map(add, sims1, sims2))

    simMat = pd.DataFrame({'user_id_R': user1s,
                           'user_id_C': user2s,
                                 'sim': combinedSims 
                           })
    simMat = simMat.pivot( index='user_id_R', columns = 'user_id_C', values = 'sim') 
    fileName = pickleJarName +str(Xth_batch) + "thSimMat"+   "_ratio(" + str(ratio)  + ").pkl"
    simMat.to_pickle(fileName)
    return simMat
    
def cluster_ratingGPS_part3(curr_df, Xth_batch, clusters_per_batch, ratio, pickleJarName):   
    #============================================== Finished Calculate Sims ==========================================
    fileName = pickleJarName +str(Xth_batch) + "thSimMat"+   "_ratio(" + str(ratio)  + ").pkl"
    if (os.path.exists(fileName)):
        simMat = pd.read_pickle(fileName)
    else:
        simMat = cluster_ratingGPS_part2(curr_df, Xth_batch, clusters_per_batch, ratio, pickleJarName)    
    #print(simMat)
    simArray = np.array(simMat)
    num_clusters = clusters_per_batch
    
    sc = SpectralClustering(num_clusters, affinity='precomputed')
    sc.fit(simArray)
    
    #================================================ Finished Clustering ==============================================

    groupIDs = [eachLabel + Xth_batch* 1000000 for eachLabel in sc.labels_]

    toGroupId = defaultdict()
    for i in range(len(groupIDs)):
        toGroupId[simMat.index.values[i]] = groupIDs[i]

    originalIdList = curr_df.user_id
    outputIdList = [toGroupId[eachId] for eachId in originalIdList]


    curr_df = curr_df.assign(user_id=outputIdList)
    #================================================ Finished Modifying Original DF ========================================   
    return curr_df

# In[607]:


def createTestDf(df, batchDic_unCluster, batch_size, XthBatch):
    
    if XthBatch not in batchDic_unCluster:
        batchDic_unCluster[XthBatch] = creatingXthBatch_unClustered(df, batch_size, XthBatch)
    print(batchDic_unCluster[XthBatch])
    return batchDic_unCluster[XthBatch]

# In[608]:

def readDataFrame(df_train, df_test, df_trainOrignal): # to generate train/test objects for surprise
    rawTrainSet = Dataset.load_from_df(df_train, Reader())
    rawTestSet  = Dataset.load_from_df(df_test, Reader())
    rawTrainOriginal = Dataset.load_from_df(df_trainOrignal, Reader())
    
    trainSet = rawTrainSet.build_full_trainset()
    _, testSet = train_test_split(rawTestSet, test_size=1.0, random_state=1)
    _, originalTrainset = train_test_split(rawTrainOriginal, test_size=1.0, random_state=1)
    trainSetForNonPrivacy = rawTrainOriginal.build_full_trainset()
    return trainSet, testSet, originalTrainset, trainSetForNonPrivacy


# In[611]:


def train(model, trainSet, factors, random , originalDic, num_of_centroids, busSimMat, privacy):
    print("Start training ...")
    
    if privacy == 0:
        if busSimMat != None:
            Algorithm = model( n_factors=factors,  random_state=random, verbose = False, busSimMat = busSimMat)
        else:
            Algorithm = model( n_factors=factors,  random_state=random, verbose = False)
    else:
        if busSimMat != None:
            Algorithm = model( n_factors=factors, random_state=random, originalDic = originalDic,
                            numCtds = num_of_centroids, busSimMat = busSimMat, verbose = False)
        else:
            Algorithm = model( n_factors=factors, random_state=random, originalDic = originalDic,
                            numCtds = num_of_centroids, verbose = False)
    Algorithm.fit(trainSet)
    print("Done ...")
    return Algorithm


# In[612]:

def precision_recall_at_k(predictions, k=10, threshold=4):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls

def test(trainedModel, testSet,log, mae = 1, rmse = 1):
    print("Start testing ...")
    predictions = trainedModel.test(testSet)
    if rmse == 1:
        with open(os.path.abspath(__file__+"/../../")+ "\\resultDumpster\\" + 'predictions.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(predictions)
        acc_rmse = accuracy.rmse(predictions, verbose=True)
        log.write(str(acc_rmse) + ',' )
    if mae == 1:
        acc_mae = accuracy.mae(predictions, verbose=True)
        #log.write(str(acc_mae) + '\n')
        log.write(str(acc_mae) + ',' )
    #predictions = algo.test(testset)
    precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=3)
    precision = sum(precisions.values()) / len(precisions)
    recall = sum(recalls.values()) / len(recalls)
    print(f"Precision: {precision}, Recall: {recall}" )
    log.write(str(precision) + ',' )
    log.write(str(recall) + '\n' )
    print("Done ...")
    return acc_rmse, acc_mae, precision, recall

# In[613]:

def prepareDf(fileName, startYear, min_NO_rating):
    df = createPandasDataFrame(fileName)
    df = filiterYear(df, startYear)
    #df = removeUsers(df, min_NO_rating)
    print(f" There are {len(df.index)} lines of records in this df after processing ..." )
    return df


# In[615]:

def originalTrainListToDic(originalTrainList):
    originalDic = defaultdict()      
    for u,i,r in originalTrainList:
        if u in originalDic:
            originalDic[u].append((i,r))  
        else:
            originalDic[u] = []
            originalDic[u].append((i,r))             
    return originalDic    



# In[615]:

    # you need to have at least some ratings
def furtherFilter(num_rating,df_train, df_trainOrignal, df_test): 
    for user in df_test['user_id'].drop_duplicates():
        if len(df_trainOrignal.loc[df_trainOrignal["user_id"] == user]) <= num_rating:
            df_test = df_test.drop(df_test[df_test.user_id == user].index)
        # cehck number of user ratings <>= you required
    
    #there is no need to calculate the similarities between users and centroids,
    #if the user is not in the test set.
    for user in df_trainOrignal['user_id'].drop_duplicates():
        if len(df_test.loc[df_test["user_id"] == user]) == 0:
            df_trainOrignal = df_trainOrignal.drop(df_trainOrignal[df_trainOrignal.user_id == user].index) 
     
    #Also, you need to keep the number of items the same in original data and in clustered data. 
    # so the number of ratings and the position of ratings can match.
    
    #for item in df_train['bus_id'].drop_duplicates():
        #print(item)
        #if len(df_trainOrignal.loc[df_trainOrignal["bus_id"] == item]) == 0:
            #print(item, "!@#!@#")
            #df_train = df_train.drop(df_train[df_train.bus_id == item].index)     
        
    for item in df_test['bus_id'].drop_duplicates():
        if len(df_trainOrignal.loc[df_trainOrignal["bus_id"] == item]) == 0:
            df_test = df_test.drop(df_test[df_test.bus_id == item].index)
        # check if item existed before   
    return df_train, df_trainOrignal, df_test
        
    
    
# In[614]:

def prpareTrainTestObj(df, batchDic_cluster, batchDic_unCluster, batch_size, NOofBatches, cluster_size, method, windowSize, ratio, pickleJarName):
    print("Preparing training and testing datasets and objects ...")
    df_train = createTrainDf_clustered(df, batchDic_cluster, batchDic_unCluster, batch_size, NOofBatches, cluster_size, method, windowSize, ratio, pickleJarName)
    df_test  = createTestDf(df, batchDic_unCluster, batch_size, NOofBatches+1) 
    print(df_test)
    df_trainOrignal = createTrainDf_unClustered(batchDic_unCluster, NOofBatches, windowSize) 
    # the original rating matrix is not imputed at this point
    
    df_train = df_train[['user_id', 'bus_id', 'rating']]
    df_test  = df_test[['user_id', 'bus_id', 'rating']]
    df_trainOrignal = df_trainOrignal[['user_id', 'bus_id', 'rating']]
    #print(f"there are {len(df_train['bus_id'].drop_duplicates())} items in train" )
    #print(f"there are {len(df_test['bus_id'].drop_duplicates())} items in test" )
    #print(f"there are {len(df_trainOrignal['bus_id'].drop_duplicates())} items in original" )      
    if len(df_train.index) <=1 or len(df_test.index) <=1 or len(df_trainOrignal) <=1:
        raise Exception("One of the dataframe is too small, check the test df first.")

    df_train, df_trainOrignal, df_test  = furtherFilter(1 ,df_train, df_trainOrignal, df_test)
    #df_trainOrignal = columnImpute(df_trainOrignal)
 
    trainSet, testSet, originalTrainSet, trainSetForNonPrivacy = readDataFrame(df_train,df_test,df_trainOrignal)
    OriginalDic = originalTrainListToDic(originalTrainSet)
    print("Done ...")
    return trainSet, testSet, OriginalDic, trainSetForNonPrivacy
    
    
# In[616]:

def batchRun(model, trainSet, originalDic, testSet, num_of_centroids,
             factors, log, busSimMat, privacy, random = 6, MAE = 1, RMSE = 1 ):
                   
    trainedModel = train(model, trainSet, factors, random, originalDic, num_of_centroids, busSimMat = busSimMat, privacy = privacy)
    acc_rmse, acc_mae, precision, recall = test(trainedModel, testSet, log, mae = MAE, rmse = RMSE)
    return acc_rmse, acc_mae, precision, recall

# In[616]:
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def totalRun(model, fileName, startYear, min_NO_rating, totalNOB, cluster_size,
             batch_size, num_of_centroids, factors, POIsims, method, windowSize, ratio, pickleJarName,
             Random = 6, mae = True, rmse = True):
    # if you need to see results, set mae or rmse to True
    # Randome is Random state 
    #blockPrint()
    if platform.system() == 'Windows':
        filePrefix  = os.path.dirname(os.path.realpath(__file__)) + "\\..\\resultDumpster\\" 
    else:
        filePrefix  = os.path.dirname(os.path.realpath(__file__)) + "/../PackageTestGround/Result/" 
        
    output = filePrefix + 'GBRS' + '-cMethod('    + str(method) +')'\
                                 + '-NOB('      + str(totalNOB)  +')'\
                                 + '-cSize('    + str(cluster_size) +')'\
                                 + '-NOC('      + str(num_of_centroids) +')'\
                                 + '-WS('      + str(windowSize) +')'\
                                 + '-time('    + str(int(time.time())) + ')'\
                                 + '.csv'

    log = open(output, 'w')
    log.write('RMSE, MAE, Precision, Recall\n')
    df = prepareDf(fileName, startYear, min_NO_rating)
    if POIsims == True:
        matFilePath = os.path.abspath(__file__+"/../../PackageTestGround" + "/simMat_new.bin") 
        with open(matFilePath, 'rb') as handle:
            busSimMat = pickle.load(handle)

    else:
        print("NO SIM FILES !!!")
        busSimMat = None

    batchDic_cluster = defaultdict()
    batchDic_unCluster = defaultdict()
    resultDic = defaultdict()

    for XthBatch in range(44, totalNOB+1):
        print(f"=================Starting the {XthBatch}th batch=================")
        trainSet, testSet, originalDic, _ = prpareTrainTestObj(df, batchDic_cluster, batchDic_unCluster,batch_size, XthBatch, cluster_size, method, windowSize, ratio, pickleJarName)
        acc_rmse, acc_mae, precision, recall = batchRun(model, trainSet, originalDic, testSet, num_of_centroids, factors, 
                 log, busSimMat, privacy = 1, random = Random, MAE = mae, RMSE = rmse )
        resultDic[XthBatch] = [acc_rmse, acc_mae, precision, recall]
    log.close
    mean = 0
    count  = 0
    for key, value in resultDic.items():
        mean += value[0]
        count += 1
    print(f"Mean RMSE: {mean/count}")
    enablePrint()
    return mean/count


def originalRun(model, fileName, startYear, min_NO_rating, totalNOB,  batch_size,  factors,  POIsims, windowSize,
                 Random = 6, mae = True, rmse = True):
    if platform.system() == 'Windows':
            filePrefix  = os.path.dirname(os.path.realpath(__file__)) + "\\..\\resultDumpster\\" 
    else:
        filePrefix  = os.path.dirname(os.path.realpath(__file__)) + "/../PackageTestGround/Result/" 
        
    output = filePrefix + 'SVD' + '-NOB('      + str(totalNOB)  +')'\
                                 + '-WS('      + str(windowSize) +')'\
                                 + '-time('    + str(int(time.time())) + ')'\
                                 + '.csv'

    log = open(output, 'w')
    log.write('RMSE, MAE\n')
    df = prepareDf(fileName, startYear, min_NO_rating)

    if POIsims == True:
        matFilePath = os.path.abspath(__file__+"/../../PackageTestGround" + "/simMat_new.bin") 
        with open(matFilePath, 'rb') as handle:
            busSimMat = pickle.load(handle)

    else:
        print("NO SIM FILES !!!")
        busSimMat = None
    batchDic_cluster = defaultdict()
    batchDic_unCluster = defaultdict()
    resultDic = defaultdict()
    for XthBatch in range(1, totalNOB+1):
        print(f"=================Starting the {XthBatch}th batch=================")
        _, testSet, _, originalTrainSet = prpareTrainTestObj(df, batchDic_cluster, batchDic_unCluster,batch_size, XthBatch, 0, None, windowSize, None, None)
        acc_rmse, acc_mae, precision, recall = batchRun(model, originalTrainSet, None, testSet,  0, factors, log, busSimMat,
                privacy =0, random = Random, MAE = mae, RMSE = rmse )
        resultDic[XthBatch] = [acc_rmse, acc_mae, precision, recall]
    log.close
    mean = 0
    count  = 0
    for key, value in resultDic.items():
        mean += value[0]
        count += 1
    print(f"Mean RMSE: {mean/count}")
    return mean/count
