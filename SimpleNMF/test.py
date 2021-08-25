from collections import defaultdict
import pandas as pd
import os
def createPandasDataFrame(fileName):
    inputFile = os.path.abspath(__file__+"/..")+ "\\" + fileName
    print(f"Reading this file: {inputFile}")
    df = pd.read_csv(inputFile)
    return df

def to_u_i_r_list(df): #modify this function when you need to use other titles
    u_i_rList = []
    for row in df.itertuples():
        u_i_rList.append((row.userId, row.movieId, row.rating))# <----this is what I mean
    return u_i_rList

#inp = [{'ccc1':10, 'c2':100}, {'ccc1':11,'c2':110}, {'ccc1':12,'c2':120}]
#df = pd.DataFrame(inp)
df = createPandasDataFrame('ratings.csv')
dfL = to_u_i_r_list(df)
rL = [a[2] for a in dfL]
print(sum(rL)/len(rL))