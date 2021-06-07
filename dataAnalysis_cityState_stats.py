import csv
import sys
import os 
from collections import defaultdict

dir_read    = os.path.dirname(os.path.realpath(__file__)) + "\\processedFile.csv"
n = 0

state_stats = defaultdict()
city_stats  = defaultdict()
with open(dir_read, 'r', newline = '', encoding = 'ISO-8859-1') as readfile : 
    reader = csv.reader(readfile)
    for row in reader:
        if len(row) >1:
            if row[4] in city_stats:
                city_stats[row[4]] += 1
            else: 
                city_stats[row[4]] = 0
            if row[5] in state_stats:
                state_stats[row[5]] += 1
            else: 
                state_stats[row[5]] = 0
            n += 1
        else:
            print(row)

print(city_stats)
print('#======================================================#')
print(state_stats)