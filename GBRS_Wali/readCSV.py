import csv
import os 
import sys
import gensim
import operator
import pandas as pd
from collections import defaultdict
from callback import EpochLogger

maxInt = sys.maxsize
while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)
readFrom  = os.path.dirname(os.path.realpath(__file__)) + "\\Urbana_Champaign_Comments.csv"
writeTo   = os.path.dirname(os.path.realpath(__file__)) + "\\Urbana_Champaign_BusAggCom.csv"
def writeToVec(filePath,writeTo):
    df = pd.read_csv(filePath)
    print(len(df.drop_duplicates(subset=['bus_id'])))
    gdf = df.groupby(['bus_id'])
    rList = []
    for x in gdf:
        userID = x[0]
        comments = ""
        for _, row in x[1].iterrows():
            comments = comments + row.text
            row = [userID, comments]
        rList.append(row)    

    result = pd.DataFrame(rList, columns = ['bus_id', 'text'])
    result.to_csv(writeTo, encoding='utf-8', index=False)

#writeToVec(readFrom, writeTo)
def readCSV(fileName):# 
    #rPath = os.path.abspath(__file__+"/../..")+ "\\ActualPackage\\resultDumpster\\" + fileName # <---- this is what I mean
    #print(rPath)
    ratingsPath = fileName
    uid = []
    iid = []
    print("####################################################################################")
    #resultWeight = {1:0, 2:0, 3:0, 4:0, 5:0}
    with open(ratingsPath, newline = '', encoding = 'ISO-8859-1') as csvfile1:
        #C:\Users\cuilo\Desktop\Git_Hub_Repo\GBRS\ActualPackage\resultDumpster
        #{1: 5080, 2: 3402, 3: 4485, 4: 8185, 5: 12839}
        userList = defaultdict()
        reader1 = csv.reader(csvfile1)
        for row1 in reader1:
            userList[row1[0]] = {1:0, 2:0, 3:0, 4:0, 5:0}


    with open(ratingsPath, newline = '', encoding = 'ISO-8859-1') as csvfile2:
        reader2 = csv.reader(csvfile2)

        for row2 in reader2:
            user = row2[0]
            rating = row2[2]
            if rating ==  '1':
                userList[user][1] += 1
            elif rating == '2':
                userList[user][2] += 1
            elif row2[2] == '3':
                userList[user][3] += 1
            elif row2[2] == '4':
                userList[user][4] += 1
            else:
                userList[user][5] += 1
    return userList          

def overallRating(fileName):
    #rPath = os.path.abspath(__file__+"/../..")+ "\\ActualPackage\\resultDumpster\\" + fileName # <---- this is what I mean
    #print(rPath)
    ratingsPath = fileName
    uid = []
    iid = []
    print("####################################################################################")
    #resultWeight = {1:0, 2:0, 3:0, 4:0, 5:0}

    dic = {1:0, 2:0, 3:0, 4:0, 5:0}


    with open(ratingsPath, newline = '', encoding = 'ISO-8859-1') as csvfile2:
        reader2 = csv.reader(csvfile2)

        for row2 in reader2:
            
            rating = row2[2]
            if rating ==  '1':
                dic[1] += 1
            elif rating == '2':
                dic[2] += 1
            elif row2[2] == '3':
                dic[3] += 1
            elif row2[2] == '4':
                dic[4] += 1
            else:
                dic[5] += 1
    return dic

d = overallRating(readFrom)
print(d)
userList = readCSV(readFrom)
#print(userList)
for eachK in userList:
    #print(eachK, userList[eachK])    
    if max(userList[eachK].items(), key=operator.itemgetter(1))[0] == 1: # most ratings are 1 or 5 or what you want
        #print(max(userList[eachK].items(), key=operator.itemgetter(1)))
        if sum(userList[eachK].values()) > 4:
            if max(userList[eachK].items(), key=operator.itemgetter(1))[1] > sum(userList[eachK].values())/4:
                print('--------------------')
                print(eachK)
                print(userList[eachK])
            