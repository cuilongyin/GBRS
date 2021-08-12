import csv
import os 
import sys
import gensim
import pandas as pd
from callback import EpochLogger


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

writeToVec(readFrom, writeTo)
