
from collections import defaultdict
import csv
import sys
''
prefix1 = 'C:\\Users\\cuilo\\Desktop\\Git_Hub_Repo\\GBRS\\DataPreProcessing\\batchFiles_Phoenix_MetaMF\\'
prefix2 = 'C:\\Users\\cuilo\\Desktop\\Git_Hub_Repo\\GBRS\\DataPreProcessing\\batchFiles_Phoenix_DMF\\'
def reorderID(input_):
    uid = []
    iid = []
    cities = []
    ratings = []
    #timeStamp = []
    with open(input_, newline = '', encoding = 'ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile)
        #next(reader)
        for row in reader:
            uid.append(row[0])
            iid.append(row[1])
            city = 'Urbana-Champaign'
            cities.append(city)
            rating = float(row[2])
            #print(rating)
            ratings.append(rating)
            #timeStamp.append(row[4])

    Gmean = sum(ratings)/(len(ratings)+1)
    print(Gmean)
    duid = defaultdict()
    diid = defaultdict()

    n = 1
    for x in uid:
        if x not in duid:
            duid[x] = n
            n += 1

    n = 1
    for x in iid:
        if x not in diid:
            diid[x] = n
            n += 1

    print("User number is:", len(duid))
    print("Item number is:", len(diid))
    output = input_ + "_index1"
    with open(output, 'w', newline = '', encoding = 'ISO-8859-1') as out:
        print("Output file opened ...")
        writer = csv.writer(out, sys.stdout, lineterminator = '\n')
        for i in range(len(uid)):
            #print(ratings[i])
            row = [duid[uid[i]], diid[iid[i]], cities[i], ratings[i]]
            writer.writerow(row)
        print("Done !")
    
    return duid,diid

def changeOrderAcoordingly(duid,diid,path_):
    uid = []
    iid = []
    cities = []
    ratings = []
    #timeStamp = []
    with open(path_, newline = '', encoding = 'ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile)
        #next(reader)
        for row in reader:
            uid.append(row[0])
            iid.append(row[1])
            city = 'Urbana-Champaign'
            cities.append(city)
            rating = float(row[2])
            #print(rating)
            ratings.append(rating)
            #timeStamp.append(row[4])

    output = path_ + "_index1"
    with open(output, 'w', newline = '', encoding = 'ISO-8859-1') as out:
        print("Output file opened ...")
        writer = csv.writer(out, sys.stdout, lineterminator = '\n')
        for i in range(len(uid)):
            print(ratings[i])
            row = [duid[uid[i]], diid[iid[i]], cities[i], ratings[i]]
            writer.writerow(row)
        print("Done !")

def generateCityUser(input_,output_):

    userCity = defaultdict()
    outputDic = defaultdict()

    with open(input_, newline = '', encoding = 'ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile)
        n = 0
        for row in reader:
            city = 'Urbana-Champaign'
            user = row[0]
            
            if city in userCity:
                if user not in userCity[city]:
                    userCity[city].append(user)
            else:
                userCity[city] = []
                userCity[city].append(user)
            n += 1
            if n %10000 == 0:
                print(n)

    #print(f"there are {len(userCity['NYC'])} users")            
    print("finished1")
    for key in userCity:
        outputDic[key] = ''
        for x in userCity[key]:   
            
            outputDic[key] += x + ';' 
        outputDic[key] = outputDic[key][:-1]
    print("finished2")
    
    with open(output_, 'w', newline = '', encoding = 'ISO-8859-1') as out:
        print("Output file opened ...")
        writer = csv.writer(out, sys.stdout, lineterminator = '\n')
        for key in outputDic:
            row = [key, outputDic[key]]
            writer.writerow(row)
        print("Done !")


for i in range(1,191):
    name = str(i)
    #duid,diid = reorderID((prefix+name+'_uc.all.rating'))
    #changeOrderAcoordingly(duid,diid,(prefix+name+'_uc.train.rating'))
    #changeOrderAcoordingly(duid,diid,(prefix+name+'_uc.test.rating'))
    generateCityUser((prefix1+name+'_uc.all.rating'), (prefix2+name+'_uc.all.rating.cityUser'))


#reorderID((prefix+'test'))
#generateCityUser((prefix+'test'))

