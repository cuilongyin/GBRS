import sys
import csv
import os
import pandas as pd
from operator import add

writePath = os.path.dirname(os.path.realpath(__file__)) + "\\aggregatedVectors.csv"

def createPandasDataFrame(fileName):
    inputFile = os.path.abspath(__file__+"/..")+ "\\" + fileName
    print(f"Reading this file: {inputFile}")
    df = pd.read_csv(inputFile)
    return df

def aggregateVectors(df):

    df  = df.reset_index()
    df['vector'] = df['vector'].replace('\n','', regex=True)

    total = df['length'].sum()       # total divider
    vectors = iter(df['vector'])     # make it iterable
    aggregatedRow = df['vector'][0]  # place holder for the aggrated vectors
    aggregatedRow = aggregatedRow[1:-1]
    aggregatedRow = [float(x) for x in aggregatedRow.split()]

    vecLength = df['length'][0]      # place holder for the length of current vecotr
    aggregatedRow = [each*vecLength for each in aggregatedRow]
    
                                     # update first vector with its weight
    next(vectors)                    # skipt first row since it was processed
    n = 1                            # index n tracks which row you are at
    for currRow in vectors:

        vecLength = df['length'][n]  # weight of the current vector
        n += 1
        currRow = currRow[1:-1]
        currRow = [float(x) for x in currRow.split()]
        currRow = [each*vecLength for each in currRow]
        aggregatedRow = list(map(add, currRow, aggregatedRow )) #entry wise adding

    averagedRow = [each/total for each in aggregatedRow]

    return averagedRow 
    

fileName = 'Urbana_Champaign_Vectors.csv'
df = createPandasDataFrame(fileName)
busList = list(set(list(df['bus_id'])))

vecList = []
for eachID in busList:
    eachDf  = df.loc[df['bus_id'] == eachID]
    eachVec = aggregateVectors(eachDf)
    vecList.append(eachVec)
    print(f"Processed {len(vecList)} POIs ...")


print(len(busList))
print(len(vecList))
resultData = {'bus_id': busList, 'vector': vecList}
resultFrame = pd.DataFrame(data=resultData)
#resultFrame.to_csv (writePath, index = False, header=True)
#print(resultFrame)
