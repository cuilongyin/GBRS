import os
import pandas as pd
from collections import defaultdict

def genFourDics(df):# this applies to windows, if you are using a linex or Mac, change next line
    uid = []
    iid = []
    for user in df['user_id']:
        uid.append(user)
    for bus in df['bus_id']:
        iid.append(bus)
    to_inner_uid = defaultdict()
    to_inner_iid = defaultdict()
    to_outer_uid = defaultdict()
    to_outer_iid = defaultdict()
#================== uid ==============================
    for eachID in uid:
        if eachID not in to_inner_uid:
            innerID = len(to_inner_uid)
            to_outer_uid[innerID] = eachID
            to_inner_uid[eachID] = int(innerID)
    #================== iid ==============================
    for eachID in iid:
        if eachID not in to_inner_iid:
            innerID = len(to_inner_iid)
            #print(innerID)
            to_outer_iid[innerID] = eachID
            to_inner_iid[eachID] = int(innerID)  
            #  
    return to_inner_uid, to_outer_uid, to_inner_iid, to_outer_iid                

# This function conferts a dataframe from raw IDs to inner IDs.
# The inputs are a dataframe of raw IDs and a string name of the data file
def convertToInner(df_input, to_inner_uid, to_inner_iid):
    df = df_input
    #to_inner_uid, _, to_inner_iid, _ = genFourDics(df)
    uids = df['user_id']
    inner_uid = []
    for eachOuter in uids:
        inner_uid.append(int(to_inner_uid[eachOuter]))
    iids = df['bus_id']
    inner_iid = []
    for eachOuter in iids:
        iid = int(to_inner_iid[eachOuter])
        inner_iid.append(iid)

    ratings = []
    for eachRating in df['rating']:
        ratings.append(eachRating)

    #dates = []
    #for eachTime in df['date']:
        #dates.append(eachTime)

    d = {'user_id':inner_uid, 'bus_id': inner_iid, 'rating' : ratings}#, 'date' : dates}
    df1 = pd.DataFrame(data=d)
    
    return df1


#This function conferts a dataframe from inner IDs to raw IDs.
#The inputs are same things compared to the above one.
def convertToOuter(df_input, to_outer_uid, to_outer_iid):
    df = df_input
    uids = df['userId']
    outer_uid = []
    for eachInner in uids:
        if eachInner in to_outer_uid:
            outer_uid.append(to_outer_uid[eachInner])
        else:
            outer_uid.append(eachInner)
    iids = df['bus_id']
    #print(type(iids))
    outer_iid = []
    for eachInner in iids:
        outer_iid.append(to_outer_iid[eachInner])
    d = {'user_id':outer_uid, 'bus_id': outer_iid}
    df1 = pd.DataFrame(data=d)
    df = df1[['user_id', 'bus_id']].copy()
    return df

def filiterYear(df, startYear): # startYear is an integer
    print("Sorting data ...")
    df = df.sort_values(by = 'date')
    print("Done.")
    for i in range(2004, startYear):
        df = df[~df.date.str.contains(str(i))]
    df = df.reset_index(drop = True)
    return df

def prepareDf(fileName, startYear):
    df = createPandasDataFrame(fileName)
    df = filiterYear(df, startYear)
    print(f" There are {len(df.index)} lines of records in this df after processing ..." )
    #print(df.head())
    #to_inner_uid, to_outer_uid, to_inner_iid, to_outer_iid = genFourDics(df)
    #df = convertToInner(df, to_inner_uid, to_inner_iid)
    #print(f" There are {len(df.index)} lines of records in this df after processing ..." )
    #print(df.head())
    return df

def createPandasDataFrame(fileName):
    inputFile = os.path.abspath(__file__ + "/..") + "\\" + fileName
    print(f"Reading this file: {inputFile}")
    return pd.read_csv(inputFile)

def creatingXthBatch_unClustered(df, batch_size, Xth_batch): #1 based, Do not put 0 or below
    if Xth_batch <= 0:
        raise Exception("1-based, DO NOT put 0 or below")
    if len(df.index) > (batch_size*Xth_batch):
        curr_df = df.iloc[(batch_size*(Xth_batch-1)):(batch_size*Xth_batch)]
    else:
        curr_df = df.iloc[(batch_size*(Xth_batch-1)):(len(df.index))]
        print(f"test set not enough, only {len(curr_df.index)} left")
    return curr_df

def workOnBatchDics(df, batchDic_unCluster, Xth_batch, batch_size):
    batchDic_unCluster[Xth_batch] = creatingXthBatch_unClustered(df, batch_size, Xth_batch)
    return batchDic_unCluster

def createTrainDf(df, batchDic_unCluster, batch_size, NOofBatches, windowSize):
    trainList = []
    startFrom = max(NOofBatches - windowSize, 1)
    for i in range(startFrom, NOofBatches+1):
        if i not in batchDic_unCluster:
            batchDic_unCluster[i] = creatingXthBatch_unClustered(df, batch_size, i)
        trainList.append(batchDic_unCluster[i])
    trainSet = pd.concat(trainList)   
    trainSet = trainSet.reset_index(drop=True)
    return trainSet

def createTestDf(df, batchDic_unCluster, batch_size, XthBatch):
    
    if XthBatch not in batchDic_unCluster:
        batchDic_unCluster[XthBatch] = creatingXthBatch_unClustered(df, batch_size, XthBatch)
    return batchDic_unCluster[XthBatch]

def createValiDf(df, valiDic, batch_size, XthBatch):
    
    if XthBatch not in valiDic:
        valiDic[XthBatch] = creatingXthBatch_unClustered(df, batch_size, XthBatch)
    return valiDic[XthBatch]

def furtherFilter(num_rating, df_train, df_test, df_validate): 
    for user in df_test['user_id'].drop_duplicates():
        if len(df_train.loc[df_train["user_id"] == user]) <= num_rating:
            df_test = df_test.drop(df_test[df_test.user_id == user].index)

    for item in df_test['bus_id'].drop_duplicates():
        if len(df_train.loc[df_train["bus_id"] == item]) == 0:
            df_test = df_test.drop(df_test[df_test.bus_id == item].index)
            
            
    for item in df_validate['bus_id'].drop_duplicates():
        if len(df_train.loc[df_train["bus_id"] == item]) == 0:
            df_validate = df_validate.drop(df_validate[df_validate.bus_id == item].index)
    for user in df_validate['user_id'].drop_duplicates():
        if len(df_train.loc[df_train["user_id"] == user]) == 0:
            df_validate = df_validate.drop(df_validate[df_validate.user_id == user].index)
 
    return df_train, df_test, df_validate

def prpareTrainTestObj(df, batchDic_unCluster,valiDic, batch_size, NOofBatches, windowSize):
    print("Preparing training and testing datasets and objects ...")
    df_train = createTrainDf(df, batchDic_unCluster, batch_size, NOofBatches, windowSize)
    df_test  = createTestDf(df, batchDic_unCluster, batch_size, NOofBatches+1) 
    df_validate = createValiDf(df, valiDic, int(batch_size/1), NOofBatches+2)
    

    
    df_train = df_train[['user_id', 'bus_id', 'rating']]
    df_test  = df_test[['user_id', 'bus_id', 'rating']]
    df_validate = df_validate[['user_id', 'bus_id', 'rating']]
   
    if len(df_train.index) <=1 or len(df_test.index) <=1 or len(df_validate.index) <=1:
        raise Exception("One of the dataframe is too small, check the validation df first.")
                                                  #(num_rating, df_train, df_test, df_validate)
    df_train, df_test, df_validate = furtherFilter(1 ,df_train, df_test, df_validate)
    
    print("Done ...")
    return df_train, df_test, df_validate


totalNOB = 191
fileName  = "Phoenix.csv"
startYear = 2007
batch_size = 3000    
windowSize = 1

valiDic = defaultdict()
batchDic_unCluster = defaultdict()

df = prepareDf(fileName, startYear)
for XthBatch in range(1, totalNOB+1):                  
    df_train, df_test, df_validate = prpareTrainTestObj(df, batchDic_unCluster,valiDic, batch_size, XthBatch, windowSize)
    to_inner_uid, to_outer_uid, to_inner_iid, to_outer_iid = genFourDics(df_train)
    df_train = convertToInner(df_train, to_inner_uid, to_inner_iid)
    df_test  = convertToInner(df_test, to_inner_uid, to_inner_iid)
    df_validate = convertToInner(df_validate, to_inner_uid, to_inner_iid)
    
    df_train['user_id'].drop_duplicates().to_csv(f"./DataPreProcessing/batchFiles_Phoenix/{XthBatch}_uc.userlist", index=False, header=False)
    df_train['bus_id'].drop_duplicates().to_csv(f"./DataPreProcessing/batchFiles_Phoenix/{XthBatch}_uc.itemlist", index=False, header=False)
    df_train[['user_id', 'bus_id', 'rating']].to_csv(f"./DataPreProcessing/batchFiles_Phoenix/{XthBatch}_uc.train.rating", index=False, header=False)
    df_test[['user_id', 'bus_id', 'rating']].to_csv(f"./DataPreProcessing/batchFiles_Phoenix/{XthBatch}_uc.test.rating", index=False, header=False)
    df_validate[['user_id', 'bus_id', 'rating']].to_csv(f"./DataPreProcessing/batchFiles_Phoenix/{XthBatch}_uc.valid.rating", index=False, header=False)
    df_all = pd.concat([df_train, df_test]) 
    df_all.to_csv(f"./DataPreProcessing/batchFiles_Phoenix/{XthBatch}_uc.all.rating", index=False, header=False)