import csv
import sys
import os 
from collections import defaultdict

#get the path of the file
dir_read     = os.path.dirname(os.path.dirname(__file__)) + "\\GBRS\\UC.csv"
#dir_busness  = os.path.dirname(os.path.dirname(__file__)) + "\\Original Dataset\\yelp_business.csv"
dir_write    = os.path.dirname(os.path.dirname(__file__)) + "\\GBRS\\PhoenixAna2.csv"

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
usrDic = defaultdict()
users = defaultdict()
items = defaultdict()
with open(dir_read, newline = '', encoding = 'ISO-8859-1') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    count = 0
    for row in reader:
        uid   = row[0]
        iid   = row[1]
        if uid in usrDic:
            usrDic[uid] += 1
        else:
            usrDic[uid] = 1
        if uid not in users:
            users[uid] = 1
        if iid not in items:
            items[iid] = 1
        count += 1
    print(f"users: {len(users)}, items: {len(items)}, count: {count}")

#write the processed information into a third file called processed
countDic = defaultdict()
with open(dir_write,'w', newline = '', encoding = 'ISO-8859-1') as writefile:
    
    writer = csv.writer(writefile, sys.stdout, lineterminator = '\n')
    writer.writerow(['NO. Ratings', 'NO. Customers'] )
    for key in usrDic:
        uid    = key
        count  = usrDic[key]
        if count in countDic:
            countDic[count] += 1
        else:
            countDic[count] = 1
    
    for key in countDic:
        x_numOfRatings  =  key
        y_numOfPeople   = countDic[key]
        line = [x_numOfRatings, y_numOfPeople]
        writer.writerow(line)

        

            

