import numbers
import os
import csv
import pandas as pd
import numpy as np
from collections import defaultdict

def createPandasDataFrame(fileName): # this applies to windows, if you are using a linex or Mac, change next line
    inputFile = os.path.abspath(__file__+"/..")+ "\\" + fileName # <---- this is what I mean
    print(f"Reading this file: {inputFile}")
    df = pd.read_csv(inputFile)
    return df

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

def convertToOuter(df_input):
    df = df_input
    fileName = 'UC.csv'
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

def partition(df, train_ratio):
    trainSize = int(len(df.index) * train_ratio)
    #curr_df = df.iloc[(batch_size*(Xth_batch-1)):(batch_size*Xth_batch)]
    train_df = df.iloc[0:(trainSize - 1)]
    test_df  = df.iloc[(trainSize):(len(df.index))]
    return train_df, test_df

def get_rng(random_state):
    # if you are not familiar with random state, ignore this.
    if random_state is None:
        return np.random.mtrand._rand
    elif isinstance(random_state, (numbers.Integral, np.integer)):
        return np.random.RandomState(random_state)
    if isinstance(random_state, np.random.RandomState):
        return random_state
    raise ValueError('Wrong random state. Expecting None, an int or a numpy '
                     'RandomState instance, got a '
                     '{}'.format(type(random_state)))

def to_u_i_r_list(df): #modify this function when you need to use other column names 
    u_i_rList = []
    for row in df.itertuples():
        u_i_rList.append((row.userId, row.movieId, row.rating))# <----this is what I mean
    return u_i_rList

def toMatrixDicU(u_i_r_list):
    MatrixDicU = defaultdict()      
    for u,i,r in u_i_r_list:
        if u in MatrixDicU:
            if i in MatrixDicU[u]:
                MatrixDicU[u][i] = r
            else:
                MatrixDicU[u] = defaultdict()
                MatrixDicU[u][i] = r  
        else:
            MatrixDicU[u] = defaultdict()
            MatrixDicU[u] = defaultdict() 
            MatrixDicU[u][i] = r             
    return MatrixDicU

def toMatrixDicI(u_i_r_list):
    toMatrixDicI = defaultdict()      
    for u,i,r in u_i_r_list:
        if i in toMatrixDicI:
            if u in toMatrixDicI[i]:
                toMatrixDicI[i][u] = r
            else:
                toMatrixDicI[i] = defaultdict()
                toMatrixDicI[i][u] = r  
        else:
            toMatrixDicI[i] = defaultdict()
            toMatrixDicI[i] = defaultdict() 
            toMatrixDicI[i][u] = r             
    return toMatrixDicI

def eachEpoch(u_i_r_list, MatrixDicU, MatrixDicI, pu, qi, n_users, n_items, n_factors, reg_pu, reg_qi): 
    user_num = np.zeros((n_users, n_factors))
    user_denom = np.zeros((n_users, n_factors))
    item_num = np.zeros((n_items, n_factors))
    item_denom = np.zeros((n_items, n_factors))
    all_ratings = [row[2] for row in u_i_r_list]
    global_mean = sum(all_ratings)/len(all_ratings)
    for u, i, r in u_i_r_list:
        # compute current estimation and error
        dot = 0  # <q_i, p_u>
        for f in range(n_factors):
            dot += qi[i, f] * pu[u, f]
        est = global_mean + dot
        
        # compute numerators and denominators
        for f in range(n_factors):
            user_num[u, f] += qi[i, f] * r
            user_denom[u, f] += qi[i, f] * est
            item_num[i, f] += pu[u, f] * r
            item_denom[i, f] += pu[u, f] * est
    # Update user factors
    all_users = list(set([eachRcrd[0] for eachRcrd in u_i_r_list]))
    for u in all_users:
        n_ratings = len(MatrixDicU[u])
        for f in range(n_factors):
            user_denom[u, f] += n_ratings * reg_pu * pu[u, f]
            pu[u, f] *= user_num[u, f] / user_denom[u, f]
    # Update item factors
    all_items = list(set([eachRcrd[1] for eachRcrd in u_i_r_list]))
    for i in all_items:
        n_ratings = len(MatrixDicI[i])
        for f in range(n_factors):
            item_denom[i, f] += n_ratings * reg_qi * qi[i, f]
            qi[i, f] *= item_num[i, f] / item_denom[i, f]
    return pu, qi

def sgd(u_i_r_list, n_factors, n_epochs = 50, reg_pu=.06,
        reg_qi=.06, init_low=0, init_high=1, random_state=None):
    rng = get_rng(random_state)
    MatrixDicU = toMatrixDicU(u_i_r_list)
    MatrixDicI = toMatrixDicI(u_i_r_list)
    n_users = len(MatrixDicU)
    n_items = len(MatrixDicI)
    pu = rng.uniform(init_low, init_high, size=(n_users, n_factors))
    qi = rng.uniform(init_low, init_high, size=(n_items, n_factors))
    n = 1
    for epoch in range(n_epochs):
        print( f"Starting epoch {n} ...")
        n += 1
        pu, qi = eachEpoch(u_i_r_list, MatrixDicU, MatrixDicI, pu, qi, n_users, n_items, n_factors, reg_pu, reg_qi)
    return pu, qi

def predict(u, i, pu, qi):
    if u not in pu:
        print(f"User {u} not in traning set ...")
        return 3
    if i not in qi:
        print(f"Item {i} not in traning set ...")
        return 3
    return np.dot(qi[i], pu[u])

def calMae(testList, pu, qi):
    predictions = []
    for u, i, r in testList:
        prediction = predict(u, i, pu, qi)
        predictions.append((r,prediction))
    MAE = np.mean([float(abs(true_r - est)) for (true_r, est) in predictions])
    return MAE


# create PD:
fileName = 'ratings.csv'
ratio = 0.75
n_factors = 10

outer_df = createPandasDataFrame(fileName)     # outer_df is the dataframe that contains the original ID
inner_df = convertToInner(outer_df, fileName)  # inner_df is the dataframe that has the inner ID, like 0,1,2,3 ...
train_df, test_df = partition(inner_df, ratio) # partition here, the train, ratio is how many percent you
                                               # want in the training set
trainList = to_u_i_r_list(train_df)
testList  = to_u_i_r_list(test_df)
#print(([row[1] for row in trainList]))
#to_inner_uid, to_outer_uid, to_inner_iid, to_outer_iid = genFourDics(fileName)
#print(to_inner_iid)
pu, qi = sgd(trainList, n_factors, n_epochs = 50, reg_pu=.06, reg_qi=.06, init_low=0, init_high=1, random_state=None)

