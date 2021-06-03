import csv
import sys
import os 
from collections import defaultdict

#get the path of the file
dir_read     = os.path.dirname(os.path.dirname(__file__)) + "\\Original Dataset\\yelp_review.csv"
dir_busness  = os.path.dirname(os.path.dirname(__file__)) + "\\Original Dataset\\yelp_business.csv"
dir_write    = os.path.dirname(os.path.realpath(__file__)) + "\\processedFile.csv"

#preview the first several lines of the file
#for review file:  'review_id', 'user_id', 'business_id', 'stars', 'date', 'text', 'useful', 'funny', 'cool'
#for business file: business_id, name, neighborhood, address, city, state, postal_code, latitude, longitude, stars, review_count, is_open, categories
preview = 0
if preview == 1:
    with open(dir_read, newline = '', encoding = 'ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile)
        n = 0
        for row in reader:
            print(row[1:3])
            n += 1
            if n == 1000:
                break

#extract GPS information and create  a dictionary about busnesses
busDic = defaultdict()
with open(dir_busness, newline = '', encoding = 'ISO-8859-1') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        iid   = row[0]
        city  = row[4]
        state = row[5]
        lat   = row[7]
        lon   = row[8]
        line  = [city, state, lat, lon]
        busDic[iid] = line

#write the processed information into a third file called processed
with open(dir_read, 'r', newline = '', encoding = 'ISO-8859-1') as readfile, \
     open(dir_write,'w', newline = '', encoding = 'ISO-8859-1') as writefile:
    
    reader = csv.reader(readfile)
    writer = csv.writer(writefile, sys.stdout, lineterminator = '\n')
    writer.writerow(['user_id', 'bus_id', 'rating', 'date', 'city', 'state','lat','lon','text'] )
    next(reader)
    #n = 0
    for row in reader:
        uid    = row[1]
        iid    = row[2]
        rating = row[3]
        date   = row[4]
        text   = row[5]
        city   = (busDic[iid])[0]
        state  = (busDic[iid])[1]
        lat    = (busDic[iid])[2]
        lon    = (busDic[iid])[3]
        line = [uid,iid,rating,date,city,state,lat,lon,text]
        writer.writerow(line)
        #if n == 10000:
            #break
       # n += 1
        

            

