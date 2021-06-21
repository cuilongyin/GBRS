import numbers
import os
import csv
import pandas as pd
import numpy as np
from collections import defaultdict


#Creat a pandas dataframe from a file, the file is in the same folder, check its format
#the input is a string of the file name
def createPandasDataFrame(fileName): # this applies to windows, if you are using a linex or Mac, change next line
    inputFile = os.path.abspath(__file__+"/..")+ "\\" + fileName # <---- this is what I mean
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

# this function partition the entire dataset into the training set and testing set
# one dataframe is needed and the second one is the ratio. For example you have 10 records, 
# if the ratio is 0.6, then 6 records are in the training set and 4 are in the test set.
def partition(df, train_ratio):
    trainSize = int(len(df.index) * train_ratio)
    #curr_df = df.iloc[(batch_size*(Xth_batch-1)):(batch_size*Xth_batch)]
    train_df = df.iloc[0:(trainSize - 1)]
    test_df  = df.iloc[(trainSize):(len(df.index))]
    return train_df, test_df

# if you do not want to mess with random state, leave it alone, don't mind it.
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

# this function extract a simple list out of the dataframe.
# the structure of the list is [(u,i,r),(u,i,r),(u,i,r),(u,i,r), ...] u:user, i: movie, r: rating
def to_u_i_r_list(df): #modify this function when you need to use other column names 
    u_i_rList = []
    for row in df.itertuples():
        u_i_rList.append((row.userId, row.movieId, row.rating))# <----this is what I mean
    return u_i_rList

# this function create a rating matrix, rows are users, and columns are items.
# however, it is in the format of a default dictionary. So you can only use it like MatrixDicU[u][i]
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

# this function create a rating matrix, rows are movie names, and columns are users.
# however, it is in the format of a default dictionary. So you can only use it like MatrixDicI[i][u]
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

# this is the major iterations of how one epoch process the data. 
# you need the u_i_r_list from the training set
# the MatrixDicU and MtraixDicI are easy to create using the above funcitons if you have the u_i_r_list.
# pu, qi are the vectors from lower-rank matrices P and Q, read NMF paper if you are confused.
# n_users is the number of users
# n_items is the number of movies
# n_factors is the number of latent factors when you perform matrix factorization.
# reg_pu, reg_qi are the learning rate for pu and qi, I have default values for them
# However, if you think you are confident enough, feel free to change them

def sgd(u_i_r_list, n_factors, n_epochs = 50, lr_pu = 0.2, lr_qi = 0.2, reg_pu=.02,
        reg_qi=.02, init_mean = 0, init_std_dev = 0.1, random_state=None):
        rng = get_rng(random_state)
        MatrixDicU = toMatrixDicU(u_i_r_list)
        MatrixDicI = toMatrixDicI(u_i_r_list)
        n_users = len(MatrixDicU)
        n_items = len(MatrixDicI)
        pu  = rng.normal(init_mean, init_std_dev,(n_users, n_factors))
        qi  = rng.normal(init_mean, init_std_dev,(n_items, n_factors))

        for current_epoch in range(n_epochs):
            print("Processing epoch {}".format(current_epoch))
            for u, i, r in u_i_r_list:
                # compute current error
                dot = 0
                for f in range(n_factors):
                    dot += qi[i, f] * pu[u, f]
                err = r - dot
                # update factors
                for f in range(n_factors):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    pu[u, f] += lr_pu * (err * qif - reg_pu * puf)
                    qi[i, f] += lr_qi * (err * puf - reg_qi * qif)
        return pu, qi 

# this is how you predic a particular rating given the user and movie
# by now you should have known what the params mean.
def predict(u, i, pu, qi):
    if u not in pu:
        #print(f"User {u} not in traning set ...")
        return 3.5
    if i not in qi:
        #print(f"Item {i} not in traning set ...")
        return 3.5
    return np.dot(qi[i], pu[u])

# This one calculate the mean absolute errors, you do need to retrieve it then print it
# you need the trained pu and qi and the u_i_r_list from the test set.
# Yes, I call it testList.
def calMae(testList, pu, qi):
    predictions = []
    for u, i, r in testList:
        prediction = predict(u, i, pu, qi)
        predictions.append((r,prediction))
    MAE = np.mean([float(abs(true_r - est)) for (true_r, est) in predictions])
    return MAE


# create PD:
fileName = 'ratings.csv'
# choose your partition ratio
ratio = 0.75
# select the number of latent factors
n_factors = 10
# load up the data
outer_df = createPandasDataFrame(fileName)     # outer_df is the dataframe that contains the original ID
inner_df = convertToInner(outer_df, fileName)  # inner_df is the dataframe that has the inner ID, like 0,1,2,3 ...
train_df, test_df = partition(inner_df, ratio) # partition here, the train, ratio is how many percent you
                                               # want in the training set
# partition the data
trainList = to_u_i_r_list(train_df)           
testList  = to_u_i_r_list(test_df)

# train the model
pu, qi = sgd(trainList, n_factors)

# test the model
mae = calMae(testList, pu, qi)
print(f"MAE is {mae} ")