#!/usr/bin/env python
# coding: utf-8
# test
# In[594]:
import math
import os
from sklearn.cluster import KMeans
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split
import platform
import pandas as pd
from surprise import AlgoBase
from surprise.utils import get_rng
from surprise import PredictionImpossible
from collections import defaultdict
from numpy import dot
from numpy.linalg import norm
import numpy as np
import csv


# In[595]:


class myModel(AlgoBase):

    def __init__(self, n_factors=100, n_epochs=20, biased=True, init_mean=0,
                 init_std_dev=.1, lr_all=.005,
                 reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
                 reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
                 random_state=None, verbose=False, originalDic=None):

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.random_state = random_state
        self.verbose = verbose
        self.originalDic = originalDic
        self.centroidRatingDic = defaultdict()
        self.simComputed = False
        self.simDic = defaultdict()
        self.num_predicted = 0      
        
        AlgoBase.__init__(self)

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)
        self.sgd(trainset)

        return self

    def sgd(self, trainset):

        rng = get_rng(self.random_state)

        bu = np.zeros(trainset.n_users, np.double)
        bi = np.zeros(trainset.n_items, np.double)
        pu = rng.normal(self.init_mean, self.init_std_dev,
                        (trainset.n_users, self.n_factors))
        qi = rng.normal(self.init_mean, self.init_std_dev,
                        (trainset.n_items, self.n_factors))
 
        global_mean = self.trainset.global_mean
        if not self.biased:
            global_mean = 0

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print("Processing epoch {}".format(current_epoch))
            for u, i, r in trainset.all_ratings():

                # compute current error
                dot = 0  # <q_i, p_u>
                for f in range(self.n_factors):
                    dot += qi[i, f] * pu[u, f]
                err = r - (global_mean + bu[u] + bi[i] + dot)

                # update biases
                if self.biased:
                    bu[u] += self.lr_bu * (err - self.reg_bu * bu[u])
                    bi[i] += self.lr_bi * (err - self.reg_bi * bi[i])

                # update factors
                for f in range(self.n_factors):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    pu[u, f] += self.lr_pu * (err * qif - self.reg_pu * puf)
                    qi[i, f] += self.lr_qi * (err * puf - self.reg_qi * qif)
                    #qi[i,f] += self.lr_qi * (err * puf - (self.reg_qi + self.reg_qi2 (sum (s[i]))\
                                                          #+self.reg_qi2 (sum(s[i][j]*qi)) )

        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi

    def impute_train(self, u, i):

        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if self.biased:
            est = self.trainset.global_mean

            if known_user:
                est += self.bu[u]

            if known_item:
                est += self.bi[i]

            if known_user and known_item:
                est += np.dot(self.qi[i], self.pu[u])

        else:
            if known_user and known_item:
                est = np.dot(self.qi[i], self.pu[u])
            else:
                raise PredictionImpossible('User and item are unknown.')

        return est   
    
    def estimateCentroidRating(self, u, i ):
        #ratings is a list containing all ratings from one group
        #[(item,rating),(item,rating),(item,rating),(item,rating) ...]
        ratings =  self.trainset.ur[u]
        filtered = filter(lambda x : x[0] == i, ratings)
        target = list(filtered)
    
        if len(target) == 0: # if the rating is there, return it, or impute it.
            return self.impute_train(u, i)
        else:
            return target[0][1]
        

    def imputeCentroidRatingMat(self):    # complete the centroids'rating matrix
        num_users = 0
        for user in list(self.trainset.all_users()):
            num_users += 1
            if num_users%5 == 0:
                print(f"---|---imputed for {num_users} groups/centroids already ...")
            for item in list(self.trainset.all_items()):
                if user in self.centroidRatingDic:
                    self.centroidRatingDic[user].append(self.estimateCentroidRating(user,item))
                else:
                    self.centroidRatingDic[user] = []
                    self.centroidRatingDic[user].append(self.estimateCentroidRating(user,item))
                
        return self
    
    
    def computeCosine(self, vec1,vec2):
        return dot(vec1, vec2)/(norm(vec1)*norm(vec2))

    
    def findMostSimilar(self, currentVec):
        centroids = []
        sims  = []
    
        for centroid in self.centroidRatingDic:
            centroids.append(centroid)
            sim = self.computeCosine(currentVec, self.centroidRatingDic[centroid])
            sims.append(sim)
        mostSimilarCentroid = centroids[sims.index(max(sims))]
        return mostSimilarCentroid
    
    def computeSimMatrix(self): # for each group, 
        print("Strat calculating sim ....") 
        original_dic_complete = self.originalDic  
        print("---Strat calcculating centroids rating matrix ....")
        self.imputeCentroidRatingMat()
        print("---Done")
        n = 0
        # when finding the most similar centroids, search for several centroids 
        # instead of one, try 2, 3,  ... 10 
        for originalUser in original_dic_complete:
            user_vec = original_dic_complete[originalUser]
            most_similar_group_in_train = self.findMostSimilar(user_vec)
            self.simDic[originalUser] = most_similar_group_in_train
            n += 1
            if n%100 == 0:
                print(f"simDic is calculating, {n} users in original are updated...")
        print("Done sim calculating ....")
        return self           
    
    def estimate(self, u,i):
        self.num_predicted += 1
        #print(type(u))
        u = u.split('UKN__')[1] #surprise will ad UKN__ infront of every user index since 
                                # it cannot find it in the oringinal trainset
        if self.simComputed == False:
            self.computeSimMatrix()
            self.simComputed = True

        most_similiar_centroid = self.simDic[u]

        #if isinstance(i, str):
            #return self.trainset.global_mean
        rating_vec = self.centroidRatingDic[most_similiar_centroid]
        if self.num_predicted%100 == 0:
            print(f"Have finisehd predicting {self.num_predicted} ratings..." )
        return rating_vec[i]


# In[596]:


def createPandasDataFrame(fileName):
    inputFile = os.path.abspath(__file__+"/..")+ "\\" + fileName
    print(f"Reading this file: {inputFile}")
    df = pd.read_csv(inputFile)
    return df


# In[597]:


def filiterYear(df, startYear): # startYear is an integer
    print("Sorting data ...")
    df = df.sort_values(by = 'date')

   
    
    for i in range(2004, startYear, 1):
        df = df[~df.date.str.contains(str(i))]
    df = df.reset_index(drop = True)

    print("Done.")
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
            #df = df[df.user_id != user] 
            df = df.drop(df[df.user_id == user].index)
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
    curr_df = convertToInner(curr_df)
    clustered = cluster_KMean_userRating(curr_df, Xth_batch, cluster_size)
    clustered = convertToOuter(clustered)
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
    for i in range(1, NOofBatches+1):
        trainList.append(creatingXthBatch_clustered(df, batch_size, i, cluster_size))
    trainSet = pd.concat(trainList)   
    trainSet = trainSet.reset_index(drop=True)
    return trainSet


# In[605]:


def createTrainDf_unClustered(df, batch_size, NOofBatches):
    trainList = []
    for i in range(1, NOofBatches+1):
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
    print(df)
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


def train(model, trainSet, factors, epochs, random , originalDic):
    Algorithm = model( n_factors=factors, n_epochs=epochs, random_state=random, originalDic = originalDic)
    Algorithm.fit(trainSet)
    return Algorithm


# In[612]:


def test(trainedModel, testSet,log, mae = 1, rmse = 1):
    
    predictions = trainedModel.test(testSet)
    if rmse == 1:
        acc_rmse = accuracy.rmse(predictions, verbose=True)
        log.write(str(acc_rmse) + ',' )
    if mae == 1:
        acc_mae = accuracy.mae(predictions, verbose=True)
        log.write(str(acc_mae) + '\n')


# In[613]:


def prepareDf(fileName, startYear, min_NO_rating):
    df = createPandasDataFrame(fileName)
    df = filiterYear(df, startYear)
    df = removeUsers(df, min_NO_rating)

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
def checkUnkownUserItem(num_rating, df_trainOrignal, df_test): 
    for user in df_test['user_id'].drop_duplicates():
        if len(df_trainOrignal.loc[df_trainOrignal["user_id"] == user]) < num_rating:
            df_test = df_test.drop(df_test[df_test.user_id == user].index)
        # cehck number of user ratings <>= you required
    
    for item in df_test['bus_id'].drop_duplicates():
        if len(df_trainOrignal.loc[df_trainOrignal["bus_id"] == item]) == 0:
            df_test = df_test.drop(df_test[df_test.bus_id == item].index)
        # check if item existed before
    return df_test
        
    
    
# In[614]:

def prpareTrainTestObj(df, batch_size, NOofBatches, cluster_size):
    df_train = createTrainDf_clustered(df, batch_size, NOofBatches, cluster_size)
    df_test  = createTestDf(df, batch_size, NOofBatches+1)
    df_trainOrignal = createTrainDf_unClustered(df, batch_size, NOofBatches) # the original rating matrix is not imputed at this point
    df_train = df_train[['user_id', 'bus_id', 'rating']]
    df_test  = df_test[['user_id', 'bus_id', 'rating']]
    df_trainOrignal = df_trainOrignal[['user_id', 'bus_id', 'rating']]
    
    if len(df_train.index) <=1 or len(df_test.index) <=1 or len(df_trainOrignal) <=1:
        raise Exception("One of the dataframe is too small, check the test df first.")
    
    df_test  = checkUnkownUserItem(3, df_trainOrignal, df_test)
    df_trainOrignal = columnImpute(df_trainOrignal)
    trainSet, testSet, originalTrainSet = readDataFrame(df_train,df_test,df_trainOrignal)
    OriginalDic = originalTrainListToDic(originalTrainSet)
    return trainSet, testSet, OriginalDic 
    
 # In[614]:
     
def genFourDics(fileName):
    rPath = os.path.abspath(__file__+"/..")+ "\\" + fileName
    ratingsPath = rPath
    
    uid = []
    iid = []

    with open(ratingsPath, newline = '', encoding = 'ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            uid.append(row[0])
            iid.append(row[1])


    to_inner_uid = defaultdict()
    to_inner_iid = defaultdict()
    to_outer_uid = defaultdict()
    to_outer_iid = defaultdict()
    
    #uid = sorted(uid)
    #iid = sorted(iid)
    
    #================== uid ==============================
    for eachID in uid:
        if eachID not in to_inner_uid:
            #dasfgsadfgsdfgsdgd
            innerID = len(to_inner_uid)
            #print(eachID, innerID)
            to_outer_uid[innerID] = eachID
            to_inner_uid[eachID] = innerID

    #================== iid ==============================
    for eachID in iid:
        if eachID not in to_inner_iid:
            innerID = len(to_inner_iid)
            to_outer_iid[innerID] = eachID
            to_inner_iid[eachID] = innerID
            
    return to_inner_uid, to_outer_uid, to_inner_iid, to_outer_iid

# In[616]:

def convertToInner(df):
    to_inner_uid, to_outer_uid, to_inner_iid, to_outer_iid = genFourDics(fileName)
    inner_uid = [to_inner_uid[x] for x in df['user_id']]
    inner_iid = [to_inner_iid[y] for y in df['bus_id']]
    df['user_id'] = inner_uid
    df['bus_id']  = inner_iid
    #print(df)
    return df

# In[616]:

def convertToOuter(df):
    #print(df)
    to_inner_uid, to_outer_uid, to_inner_iid, to_outer_iid = genFourDics(fileName)
    outer_uid = [(str(x)+'_clustered') for x in df['user_id']]
    outer_iid = [to_outer_iid[y] for y in df['bus_id']]
    df['user_id'] = outer_uid
    df['bus_id']  = outer_iid
    return df

# In[616]:

def batchRun(model, trainSet, originalDic, testSet, log, epochs = 40, random = 6, MAE = 1, RMSE = 1 ): 
    trainedModel = train(model, trainSet, factors, epochs, random, originalDic)
    test(trainedModel, testSet, log, mae = MAE, rmse = RMSE)


# In[616]:


def totalRun(fileName, startYear, min_NO_rating, totalNOB, cluster_size, maxEpochs = 40, Random = 6, mae = True, rmse = True):
    # if you need to see results, set mae or rmse to True
    # Randome is Random state 
    if platform.system() == 'Windows':
        filePrefix  = os.path.dirname(os.path.realpath(__file__)) + "\\" 
    else:
        filePrefix  = os.path.dirname(os.path.realpath(__file__)) + "/" 
        
    output = filePrefix + 'GBRS' + '_startYear_'    + str(startYear)\
                                 + '_minRatings_'   + str(min_NO_rating)\
                                 + '_NOB_'          + str(totalNOB)\
                                 + '_clusterSize_'  + str(cluster_size)\
                                 + '.txt'
    log = open(output, 'w')
    log.write('RMSE, MAE\n')
    
    df = prepareDf(fileName, startYear, min_NO_rating)
    for XthBatch in range(1,totalNOB+1):
        print(f"=================Starting the {XthBatch}th batch=================")
        trainSet, testSet, originalDic = prpareTrainTestObj(df, batch_size, XthBatch, cluster_size)
        batchRun(model, trainSet, originalDic, testSet, log, epochs = maxEpochs, random = Random, MAE = mae, RMSE = rmse )
    log.close

# In[ ]:


fileName = "UC.csv"
#fileName = "Urbana_Champaign_intIndex.csv"
startYear = 2007
min_NO_rating = 3      # 3: 22785 lines
batch_size = 1519     
cluster_size = 10      #clusters per batch
totalNOB = 14          #number of Batch, not including the test batch
factors = 5
model = myModel
totalRun(fileName,startYear, min_NO_rating, totalNOB, cluster_size )


# In[ ]:




