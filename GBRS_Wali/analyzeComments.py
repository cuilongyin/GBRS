import csv
import os 
import sys
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
maxInt = sys.maxsize
while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)
readFrom  = os.path.dirname(os.path.realpath(__file__)) + "\\Urbana_Champaign_Vectors(1Entry).csv"
writeTo   = os.path.dirname(os.path.realpath(__file__)) + "\\commentAnalyze.csv"


def readCSV(fileName):# 
    #rPath = os.path.abspath(__file__+"/../..")+ "\\ActualPackage\\resultDumpster\\" + fileName # <---- this is what I mean
    #print(rPath)
    ratingsPath = fileName
    lengths = []
    commentNumbers = []
    ratings = []
    with open(fileName, 'r', newline = '', encoding = 'ISO-8859-1') as readfile:
        reader = csv.reader(readfile)
        next(reader)
        for row in reader:
            l = row[2]
            c = row[3][1:-1]
            #print(c[1:-1])
            r = row[4]
            lengths.append(l)
            commentNumbers.append(c)
            ratings.append(r)

    return lengths, commentNumbers, ratings


def writeCSV(writeTo, list1,list2,list3):

    with open(writeTo,'w', newline = '', encoding='utf-8') as writefile:
        
        writer = csv.writer(writefile, sys.stdout, lineterminator = '\n')
        writer.writerow(['length', 'comment', 'rating'] )

        for i in range(len(list1)):
          
            l = list1[i]
            c = list2[i]
            r = list3[i]

            line = [l,c,r]
            writer.writerow(line)

#listL, listC, listR = readCSV(readFrom)
#writeCSV(writeTo,listL, listC, listR)
#listL = pd.to_numeric(listL[:1000])
#listC = pd.to_numeric(listC[:1000])
#listR = pd.to_numeric(listR[:1000])
#df = pd.DataFrame(list(zip(listL, listC, listR)), columns =['length', 'comments','rating'])

#print(df['rating'].corr(df['length']))


#normalized_df=(df-df.mean())/df.std()
#normalized_df.to_excel("commentsAnalyzing.xlsx")

#print(normalized_df)
#df.plot(kind='bar')
#plt.show()