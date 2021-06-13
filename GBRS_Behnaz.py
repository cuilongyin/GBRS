#!/usr/bin/env python
# coding: utf-8
# test
#test by behnaz
# In[594]:
import math
import re
import os
import platform
import pandas as pd
import numpy as np
from statistics import mean
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise import AlgoBase
from surprise.utils import get_rng
from surprise import PredictionImpossible
from collections import defaultdict
from numpy import dot
from numpy.linalg import norm
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


def cluster_KMean_userRating(curr_df, Xth_batch, clusters_per_batch):
    # df = columnImpute(df)
    # pdf = df.pivot(index='user_id', columns = 'bus_id', values = 'rating') 
    # columnNames = pdf.columns
    # model = KMeans(n_clusters = clusters_per_batch)
    # model.fit_predict(pdf)
    # clusters = pd.DataFrame(model.cluster_centers_)
    # clusters.columns= columnNames
    # df = clusters.T.unstack().reset_index(name='rating')
    # df.rename(columns={'level_0': 'user_id'}, inplace=True)
    # df['user_id'] = df['user_id'] + 100000*Xth_batch # this is to make each centroids' ID special
    #                                                 # So this batch's IDs to mess with the next one's'
    # # Do for Behnaz, 1: convert the bus_id back
    # # 2, increament the user_id or "group id"
    # print(df)
    # return df df1 = curr_df
    # df = columnImpute(df)
    
    #df1 = df
    # pdf = df.pivot(index='user_id', columns = 'bus_id', values = 'rating') 
    # columnNames = pdf.columns
    # model = KMeans(n_clusters = clusters_per_batch)
    # model.fit_predict(pdf)
    # clusters = pd.DataFrame(model.cluster_centers_)
    # clusters.columns= columnNames
    # df = clusters.T.unstack().reset_index(name='rating')
    # df.rename(columns={'level_0': 'user_id'}, inplace=True)
    # df['user_id'] = df['user_id'] + 100000*Xth_batch # this is to make each centroids' ID special
    #                                                 # So this batch's IDs to mess with the next one's'
    # # Do for Behnaz, 1: convert the bus_id back
    # # 2, increament the user_id or "group id"
    df1 = curr_df
    print("-----------------", len(df1))
    print(df1)
    df1.sort_values(by=['user_id'], inplace=True)
    print(df1)
    print("Start Behnaz code")
    dict1 = df1.to_dict() 
    #val_list is data  in list
    val_list = list(dict1.values())
    #%%%%%%%%%%% unique users ID and business ID %%%%%%%%%%%
    #ui is unique userid list
    list_set1 = set(list(val_list[0].values()))
    ui = (list(list_set1)) 
    #bi is unique userid list
    list_set2= set(list(val_list[1].values()))
    bi= (list(list_set2))
    #%%%%%%%%%%%% 
    #%%%%%%%%%%%% create Nan Matrix%%%%%%%%%
    arr = [[None]*len(bi)]*len(ui)
    df3 = pd.DataFrame(arr, 
                    columns=(bi),
                    index = [ui])
    #df3 is dataframe matrix
    df3 = df3.fillna(value=np.nan)
    #df3 = df3.sort_values(by= ["user_id"], ascending=True)
    
    print("Created rating nan matrix")
    print("-------",len(df3))
    #print(df3)
    # # #%%%%%%%%%%%% Insert Rating To Place %%%%%%%%%  
    k = 0
    i = 0
    uf1=  dict1.get('user_id')
    #print("uf1")
    #print(uf1)
    #print("ppppp",uf1.get(11))
    bf1 = dict1.get('bus_id')
    #print(uf1.values())
    for i in enumerate (uf1.values()):

          k=df1.index[i[0]]
          uf2 = uf1.get(k)
          a = df1[df1['user_id']== uf2]['bus_id']
          #print(a)
          uiindex = ui.index(uf2)
          for j in range(a.size):
              a1 = a.iloc[j:j+1]
              t = a.index[j]
              a2 = a1.iloc[0]
              biindex = bi.index(a2)
              df3.iloc[uiindex, biindex] = (df1.rating[t])
              
    print("Inserted rating to the correct place")
    print("-------")
    dfMatrix = df3
    #dfMatrix is dataframe matrix with real rate
    #print(dfMatrix)    
    
    # #%%%%%%%%%%%% 
    # #%%%%%%%%%%%%delete users who just gave one rating %%%%%%%%%%%%%%%   
    # #print("started to delet user who have one rate")
    # # deletindex =[]
    # # count =0
    # # for i in range(len(dfMatrix)):
    # #     if dfMatrix.iloc[i].count() < 2 :
    # #             count +=1
    # #             deletindex.append(i)
    # # dfMatrix.drop(dfMatrix.index[deletindex],inplace =True)

    # # empty_cols = [col for col in dfMatrix.columns if dfMatrix[col].isnull().all()]
    # # # Drop these columns from the dataframe
    # # dfMatrix.drop(empty_cols,
    # #     axis=1,
    # #     inplace=True)
    # # print(dfMatrix)
    # #%%%%%%%%%%%%%%%% similar user location centroid%%%%%%%%%%%%%
    df2 = df1.copy()
    uniquedataframe = df2.drop_duplicates(subset=['user_id'],keep="first" , inplace=False)
    #print(uniquedataframe)
    uniquedataframe.sort_values(by=['user_id'], inplace=True)
    print(uniquedataframe)
    uniquelist = uniquedataframe.values.tolist()
    a_list= df1.values.tolist()
    m2 = df1.to_dict()
    val_list = list(m2.values())
    bs = val_list[0]
    val_l = list(bs.values())
    locdataframe = pd.DataFrame( columns = ['user_id','lat','lon'])
    
    for j in uniquelist:
        indices = []
        l1= []
        l2= []
        for i in range(len(a_list)):
              if val_l[i] == j[0]:
                  indices.append(i)
                  l1.append(a_list[i][4])
                  l2.append(a_list[i][5])

        l1avg = mean(l1)
        l2avg = mean(l2)
        temp = {'user_id': j[0],'lat':l1avg, 'lon':l2avg}
        locdataframe = locdataframe.append(temp, ignore_index = True)
      
           
    locdataframe = locdataframe.astype({"user_id": int, "lat": float, "lon" :float})
    locdataframe.sort_values(by=['user_id'], inplace=True) 
    locdataframe.reset_index(drop=True, inplace=True)
    #locdataframe is dataframe of location
    print("Prepared location file")
    print("-------")
    print(locdataframe)
    # # #%%%%%%%%%%%%%%find correct user for the GPS%%%%%%%%%%%%
    # s=[]
    # for row in dfMatrix.index:
    #     res = re.sub("\D", "", str(row))
    #     s.append(int(res))
    # #dfloc = df1.copy()
    # a =[]
    # a = locdataframe["user_id"].tolist()
    # set_difference = set(a) - set(s)
    # locdataframe.reset_index(drop=True, inplace=True)
    # for i in set_difference: 
    #     locdataframe.drop(locdataframe[locdataframe['user_id'] == (i)].index,inplace=True)
    # print("hhhhhhhhhhhhh")
    # print(locdataframe)    
    # #%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%%%Create nan matrix for RATING and GPS and GPS & RATING %%%%%%%%%%%%%
    usersgps = locdataframe.iloc[:,0]
    usersgps_list = set(usersgps)
    unique_usersgps = list(usersgps_list)
    coordinate =(len(unique_usersgps))
    #print(coordinate)
    arr = [[None]*coordinate]*coordinate
    gpsMatrix = pd.DataFrame(arr, 
                    columns=(unique_usersgps ),
                    index = [unique_usersgps ])
    gpsMatrix = gpsMatrix.fillna(value=np.nan)
    print("Created NAN matrix for GPS similarity") 
    print("-------")
    #print(gpsMatrix)
    rateMatrix= gpsMatrix.copy()
    print("Created NAN matrix for RATING similarity") 
    print("-------")
    rateAndgpsMatrix= gpsMatrix.copy()
    print("Created NAN matrix for RATING & GPS similarity") 
    print("-------")
    # # # # #%%%%%%%%%%%%%%%%%%%Create similarity GPS matrix %%%%%%%%%%%%%%%
    print("Start to make GPS Smilarity matrix")
    dictloc =locdataframe.to_dict()
    dictgpsMatrix = gpsMatrix.to_dict()
    df_dictgpsMatrix = pd.DataFrame(dictgpsMatrix)
    for i in range (coordinate):
            ii = df_dictgpsMatrix.columns[i]
            for j in range (coordinate):
                jj = df_dictgpsMatrix.columns[j]
                A = [dictloc["lat"][i],dictloc["lon"][i]]
                B = [dictloc["lat"][j],dictloc["lon"][j]]
                similarity = np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))
                #print (f"Cosine Similarity :{similarity }")
                dictgpsMatrix[ii][jj,] =  similarity
              
    print("Created cosine similarity(GPS) ")
    print("--------")
    #print(dictgpsMatrix)
    dfGPSMatrix= pd.DataFrame(dictgpsMatrix)
    #print(dfGPSMatrix)
    #dfGPSMatrix.to_csv("p.csv")
       
    # # #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # # #%%%%%%%%%%%%%%%%%%%%insert value into empty space by using baseline model%%%%%%%%%%%%
    # # # #-------
    def baselinemodel(mu,userid,itemid):
    
        itemAvg = dfMatrix.iloc[:,itemid].mean()
        itemDif = itemAvg - mu
        userAvg = dfMatrix.iloc[userid].mean()
        userDif = userAvg - mu
        #mu + userDif + itemDif 
        return mu + userDif + itemDif 
    # # #------- 
    print("Start to insert  value in  the empty space by using baseline model")
    mu =  dfMatrix.mean(axis=0).mean()
    #mu1 = dfMatrix.mean(axis=1).mean()
    #print(mu , mu1)
    arr = dfMatrix.to_numpy()
    rowlength = len(dfMatrix)
    columnlength = len(dfMatrix.columns)
    for i in range (rowlength):
        #print("i",i)
        for j in range (columnlength):
            rate = baselinemodel(mu,i,j)
            if rate > 5:
                  rate=5
            elif rate  < 1:
                  rate =1
            if(np.isnan(arr[i][j]).any()):
                          arr[i][j] = rate

    completedf = pd.DataFrame(arr, columns = dfMatrix.columns, index = dfMatrix.index) 
    completedf.to_csv("CompleteRatingMatrix.csv")
    print("Inserted value into empty space in rating matrix by using baseline model")
    # # #%%%%%%%%%%%%%%%%Create similarity rating matrix %%%%%%%%%%%%%
    completedf['mean'] = completedf.mean(axis=1)
    users_rate_avg = completedf.to_dict()
    #print(completedf)
    dictratingMatrix = rateMatrix.to_dict()
    print("=============")
    # # # #%%%%%%%%%%%%  Insert Cosin similarity%%%%%%%%%%%%%%%%%
    print("Start to make RATING Smilarity matrix")
    for row in rateMatrix.index:
          i = int(re.sub("\D", "", str(row)))
          #print("... ",i)
          for column in range (len(rateMatrix.columns)):
              j= int(rateMatrix.columns[column])
              A = [users_rate_avg['mean'][i,],users_rate_avg['mean'][j,]]
              B = [users_rate_avg['mean'][j,],users_rate_avg['mean'][i,]]
              similarity = np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))
              dictratingMatrix[i][j,] = similarity
    print("Created Matrix with rating cosine similarity(Rating) ") 
    dfRATINGMatrix= pd.DataFrame(dictratingMatrix)
    #print(dfRATINGMatrix)
    # # # #%%%%%%%%%%%%%%%%%%%
    # # # #%%%%%%%%%%%%%%%%Add two Similarity %%%%%%%%%%%%%%%%%
    print("Started to Add two Similarity")
    #ratel=dfGPSMatrix.values.tolist()
    #Ratingdict = dictratingMatrix
    #GPSdict = dictgpsMatrix
    #gpsl= dfRATINGMatrix.values.tolist()
    dictrateAndgpsMatrix = rateAndgpsMatrix.to_dict()
    #size=len(dictratingMatrix)
    #print(size)
    #insert value
    Lambda = 0.5
    #kop=0
    for row in rateMatrix.index:
      o = int(re.sub("\D", "", str(row)))
      #print("...",o)  
      for column in range (len(rateMatrix.columns)):
            p = int(rateMatrix.columns[column])
            #print("---",p)
            simRate = dictratingMatrix[o][p,]
            simLoc  = dictgpsMatrix [o][p,]
            valueToInsert = (Lambda * simLoc)+ ((1-Lambda) * simRate)
            #print(valueToInsert)
            dictrateAndgpsMatrix[o][p,] = valueToInsert
            #kop +=1
    print("Created Matrix Gps and rating Similarity")
    dfrateAndgpsMatrix= pd.DataFrame(dictrateAndgpsMatrix)     
    print(dfrateAndgpsMatrix)
    # # #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Clustering part%%%%%%%%%%%%%%
    print("Start to cluster ")
    simArray = np.array(dfrateAndgpsMatrix)
    num_clusters = clusters_per_batch
    sc = SpectralClustering(num_clusters, affinity='precomputed')
    sc.fit(simArray)
    ratep=uniquedataframe.rating
    bus=uniquedataframe.bus_id
    #useid=df1.user_id
    outputdict = dict(enumerate(sc.labels_.flatten(), 1))
    listofgg= list(zip(sc.labels_ * 1000000,bus,ratep))
    listofgg111= list(zip(df1.user_id,df1.bus_id,df1.rating))
    output1= pd.DataFrame(listofgg, columns=['group_id', 'bus_id','rating'])
    output2= pd.DataFrame(listofgg111, columns=['user_id', 'bus_id','rating'])
    print("-------------",len(output1))
    #---------- Recover Data Frame After Clustering
    #print(output1)
    #print(output2)
    deleteduplicate = output2.drop_duplicates(subset=['user_id'])
    deleteduplicate.reset_index(drop=True, inplace=True)
    #print(deleteduplicate)
    df_all_rows = pd.concat([output2.user_id],axis = 1)
    #print(df_all_rows)
    print("-----------------------------------------")

    for i in range (len(output1)):
      
      value = output1.group_id[i]
      ccc = deleteduplicate.user_id[i]
      df_all_rows[df_all_rows.eq(ccc).any(1)] = value
      
      lastpart = pd.concat([df_all_rows.user_id, output2.bus_id, output2.rating],axis = 1)    
    #print(lastpart )
    print("Clustering is done")
    print("Done Behnaz code ")
    return lastpart


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




