import csv
import os 
import sys
import gensim
import operator
import pandas as pd
from collections import defaultdict

rootPath = os.path.abspath(__file__+"/../")
#fileName = "\\processedFile.csv"
fileName = "\\UC.csv"

filePath = rootPath + fileName 
print("123",filePath)
with open(filePath, newline = '', encoding = 'ISO-8859-1') as csvfile:
    reader = csv.reader(csvfile)
    #read.next()
    for row in reader:
        print(row)