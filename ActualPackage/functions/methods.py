import os
import math
import pickle
import platform
import pandas as pd
from surprise import Reader
from surprise import Dataset
from surprise import accuracy
from sklearn.cluster import KMeans
from collections import defaultdict
from surprise.model_selection import train_test_split
# In[596]:


def createPandasDataFrame(fileName):
    inputFile = os.path.abspath(__file__+"/../../")+ "/PackageTestGround/" + fileName
    print(f"Reading this file: {inputFile}")
    df = pd.read_csv(inputFile)
    return df


# In[597]:


def filiterYear(df, startYear): # startYear is an integer
    print("Sorting data ...")
    df = df.sort_values(by = 'date')
    print("Done.")
    for i in range(2004, startYear, 1):
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


def creatingXthBatch_clustered(df, batch_size, Xth_batch, cluster_size): #1 based, Do not put 0 or below
    if Xth_batch <= 0:
        raise Exception("1-based, DO NOT put 0 or below")
    
    if len(df.index) > (batch_size*Xth_batch):
        curr_df = df.iloc[(batch_size*(Xth_batch-1)):(batch_size*Xth_batch)]
    else:
        curr_df = df.iloc[(batch_size*(Xth_batch-1)):(len(df.index))]
        print(f"test set not enough, only {len(curr_df.index)} left")
    clustered = cluster_KMean_userRating(curr_df, Xth_batch, cluster_size)
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


def createTrainDf_clustered(df, batch_size, NOofBatches, cluster_size):
    trainList = []
    trainList = []
    startFrom = 1
    #if NOofBatches - 4 < 1:
        #startFrom = 1
    #else:
        #startFrom = NOofBatches - 4
    for i in range(startFrom, NOofBatches+1):
        trainList.append(creatingXthBatch_clustered(df, batch_size, i, cluster_size))
    trainSet = pd.concat(trainList)   
    trainSet = trainSet.reset_index(drop=True)
    return trainSet


# In[605]:


def createTrainDf_unClustered(df, batch_size, NOofBatches):
    trainList = []
    startFrom = 1
    #if NOofBatches - 4 < 1:
        #startFrom = 1
    #else:
        #startFrom = NOofBatches - 4
    for i in range(startFrom, NOofBatches+1):
        trainList.append(creatingXthBatch_unClustered(df, batch_size, i))
    trainSet = pd.concat(trainList)
    
    return trainSet


# In[606]:


def cluster_KMean_userRating(df, Xth_batch, clusters_per_batch):
    df = columnImpute(df)
    pdf = df.pivot(index='user_id', columns = 'bus_id', values = 'rating') 
    columnNames = pdf.columns
    model = KMeans(n_clusters = clusters_per_batch)
    model.fit_predict(pdf)
    clusters = pd.DataFrame(model.cluster_centers_)
    clusters.columns= columnNames
    df = clusters.T.unstack().reset_index(name='rating')
    df.rename(columns={'level_0': 'user_id'}, inplace=True)
    df['user_id'] = df['user_id'] + 100000*Xth_batch # this is to make each centroids' ID special
                                                    # So this batch's IDs to mess with the next one's'
    # Do for Behnaz, 1: convert the bus_id back
    # 2, increament the user_id or "group id"

    return df


# In[607]:


def createTestDf(df, batch_size, XthBatch):

    testSet = creatingXthBatch_unClustered(df, batch_size, XthBatch)
    return testSet 


# In[608]:

def readDataFrame(df_train, df_test, df_trainOrignal): # to generate train/test objects for surprise
    rawTrainSet = Dataset.load_from_df(df_train, Reader())
    rawTestSet  = Dataset.load_from_df(df_test, Reader())
    rawTrainOriginal = Dataset.load_from_df(df_trainOrignal, Reader())
    
    trainSet = rawTrainSet.build_full_trainset()
    _, testSet = train_test_split(rawTestSet, test_size=1.0, random_state=1)
    _, originalTrainset = train_test_split(rawTrainOriginal, test_size=1.0, random_state=1)
    return trainSet, testSet, originalTrainset


# In[611]:


def train(model, trainSet, factors, epochs, random , originalDic, num_of_centroids, busSimMat):
    print("Start training ...")
    if busSimMat != None:
        Algorithm = model( n_factors=factors, n_epochs=epochs, random_state=random, originalDic = originalDic,
                           numCtds = num_of_centroids, busSimMat = busSimMat, verbose = False)
    else:
        Algorithm = model( n_factors=factors, n_epochs=epochs, random_state=random, originalDic = originalDic,
                           numCtds = num_of_centroids, verbose = False)
    Algorithm.fit(trainSet)
    print("Done ...")
    return Algorithm


# In[612]:


def test(trainedModel, testSet,log, mae = 1, rmse = 1):
    print("Start testing ...")
    predictions = trainedModel.test(testSet)
    if rmse == 1:
        acc_rmse = accuracy.rmse(predictions, verbose=True)
        log.write(str(acc_rmse) + ',' )
    if mae == 1:
        acc_mae = accuracy.mae(predictions, verbose=True)
        log.write(str(acc_mae) + '\n')
    print("Done ...")

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
            originalDic[u].append(r)  
        else:
            originalDic[u] = []
            originalDic[u].append(r)             
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
    
    for item in df_train['bus_id'].drop_duplicates():
        if len(df_trainOrignal.loc[df_trainOrignal["bus_id"] == item]) == 0:
            df_train = df_train.drop(df_train[df_train.bus_id == item].index)     
        
    for item in df_test['bus_id'].drop_duplicates():
        if len(df_trainOrignal.loc[df_trainOrignal["bus_id"] == item]) == 0:
            df_test = df_test.drop(df_test[df_test.bus_id == item].index)
        # check if item existed before   
    return df_train, df_trainOrignal, df_test
        
    
    
# In[614]:

def prpareTrainTestObj(df, batch_size, NOofBatches, cluster_size):
    print("Preparing training and testing datasets and objects ...")
    df_train = createTrainDf_clustered(df, batch_size, NOofBatches, cluster_size)
    df_test  = createTestDf(df, batch_size, NOofBatches+1)
    df_trainOrignal = createTrainDf_unClustered(df, batch_size, NOofBatches) # the original rating matrix is not imputed at this point
    df_train = df_train[['user_id', 'bus_id', 'rating']]
    df_test  = df_test[['user_id', 'bus_id', 'rating']]
    df_trainOrignal = df_trainOrignal[['user_id', 'bus_id', 'rating']]
    #print(f"there are {len(df_train['bus_id'].drop_duplicates())} items in train" )
    #print(f"there are {len(df_test['bus_id'].drop_duplicates())} items in test" )
    #print(f"there are {len(df_trainOrignal['bus_id'].drop_duplicates())} items in original" )      
    if len(df_train.index) <=1 or len(df_test.index) <=1 or len(df_trainOrignal) <=1:
        raise Exception("One of the dataframe is too small, check the test df first.")
    
    df_train, df_trainOrignal, df_test  = furtherFilter(4,df_train, df_trainOrignal, df_test)
    df_trainOrignal = columnImpute(df_trainOrignal)
 
    trainSet, testSet, originalTrainSet = readDataFrame(df_train,df_test,df_trainOrignal)
    OriginalDic = originalTrainListToDic(originalTrainSet)
    print("Done ...")
    return trainSet, testSet, OriginalDic 
    
    
# In[616]:

def batchRun(model, trainSet, originalDic, testSet, num_of_centroids,
             factors, log, busSimMat, epochs = 40, random = 6, MAE = 1, RMSE = 1 ):

    trainedModel = train(model, trainSet, factors, epochs, random, originalDic, num_of_centroids, busSimMat)
    test(trainedModel, testSet, log, mae = MAE, rmse = RMSE)


# In[616]:


def totalRun(model, fileName, startYear, min_NO_rating, totalNOB, cluster_size,
             batch_size, num_of_centroids, factors, POIsims, maxEpochs = 40, 
             Random = 6, mae = True, rmse = True):
    # if you need to see results, set mae or rmse to True
    # Randome is Random state 
    if platform.system() == 'Windows':
        filePrefix  = os.path.dirname(os.path.realpath(__file__)) + "\\" 
    else:
        filePrefix  = os.path.dirname(os.path.realpath(__file__)) + "/../PackageTestGround/Result/" 
        
    output = filePrefix + 'GBRS' + '_startYear_'    + str(startYear)\
                                 + '_minRatings_'   + str(min_NO_rating)\
                                 + '_NOB_'          + str(totalNOB)\
                                 + '_clusterSize_'  + str(cluster_size)\
                                 + '_num_of_centroids_'  + str(num_of_centroids)\
                                 + '.txt'
    log = open(output, 'w')
    log.write('RMSE, MAE\n')
    df = prepareDf(fileName, startYear, min_NO_rating)
    if POIsims == True:
        matFilePath = os.path.abspath(__file__+"/..")+ "/" + "testFile"
        with open(matFilePath, 'rb') as handle:
            busSimMat = pickle.load(handle)
    else:
        busSimMat = None
    for XthBatch in range(1,totalNOB+1):
        print(f"=================Starting the {XthBatch}th batch=================")
        trainSet, testSet, originalDic = prpareTrainTestObj(df, batch_size, XthBatch, cluster_size)
        batchRun(model, trainSet, originalDic, testSet, num_of_centroids, factors, 
                 log, busSimMat, epochs = maxEpochs, random = Random, MAE = mae, RMSE = rmse )
    log.close