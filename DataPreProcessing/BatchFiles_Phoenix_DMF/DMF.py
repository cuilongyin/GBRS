from collections import defaultdict
import csv
import sys
import random
import numpy as np
import math
import time
#import utils
import argparse

def predictRating(pu, qi, data):
    predictedRating = np.dot(pu, qi)
    if data == 1:
        maxR = 1
        minR = 0
    else:
        maxR = 5
        minR = 1
    if predictedRating > maxR:
        predictedRating = maxR
    elif predictedRating < minR:
        predictedRating = minR
    return predictedRating

#done
def computeError(realRating, predictedRating):
    return realRating - predictedRating
#done
def computeRMSE(dataset, userVectors, itemComVectors, itemPerVectors, data):
    rmse = 0
    mae = 0
    rmseOld = 0
    oldCount = 0
    rmseNew = 0
    newCount = 0
    count = 0
    for row in dataset:
        userID = int(row['userID'])
        itemID = int(row['itemID'])
        rating = float(row['rating'])#1.0

        # Since list index in python starts from 0 and user ID
        # in MovieLens dataset start from 1, all IDs should minus one.
        userVector = userVectors[userID]
        itemComVector = itemComVectors[userID][itemID]
        itemPerVector = itemPerVectors[userID][itemID]
        itemVector = itemComVector + itemPerVector
        predictedRating = predictRating(userVector, itemVector, data)
        error = computeError(rating, predictedRating)
        temp = error ** 2
        rmse += temp
        mae += abs(error)
        count += 1
    rmse /= count  # len(dataset)
    mae /= count
    rmse = math.sqrt(rmse)
    return rmse, mae, rmseNew, rmseOld, count, newCount, oldCount
#done

def initFactorVectors(rows, columns):
    return np.random.randn(rows, columns) / math.sqrt(columns)

def initBiases(number):
    return [0] * number

# compute P@5, R@5, P@10, R@10
def computePrecision(test4Rank, userVectors, itemComVectors, itemPerVectors, itemNum, data):
    precision5 = 0.0
    precision10 = 0.0
    recall5 = 0.0
    recall10 = 0.0

    rightUserNum = 0
    for userID in test4Rank:

        ratings = {}

        # predict all the ratings for userID
        for itemID in range(itemNum):
            # filter out the items in train set
            # if not itemID in trainItems:
            userVector = userVectors[userID]
            itemComVector = itemComVectors[userID][itemID]
            itemPerVector = itemPerVectors[userID][itemID]
            itemVector = itemComVector + itemPerVector
            predictedRating = predictRating(userVector, itemVector, data)
            ratings[itemID] = predictedRating

        # rank items based on ratings, desc
        sortedRating = sorted(ratings, key=lambda k: ratings[k], reverse=True)

        items4Rank = test4Rank[userID] # could be all the items
        hit = 0
        index = 0
        # increment hit when the recommended item is in the actual visited item list.
        for item_predict in sortedRating:
            for iRating_true in items4Rank:
                item = iRating_true.split(',')[0]
                if int(item) == int(item_predict):
                    hit += 1
                    break
            index += 1
            if index == 5:
                break

        precision5 += float(hit) / 5.0
        recall5 += float(hit) / float(len(items4Rank))

        hit = 0
        index = 0
        for item_predict in sortedRating:
            for iRating_true in items4Rank:
                item = iRating_true.split(',')[0]
                if int(item) == int(item_predict):
                    hit += 1
                    break
            index += 1
            if index == 10:
                break

        precision10 += float(hit) / 10.0
        recall10 += float(hit) / float(len(items4Rank))
    
    precision5 /= len(test4Rank)
    precision10 /= len(test4Rank)
    recall5 /= len(test4Rank)
    recall10 /= len(test4Rank)
    return precision5, recall5, precision10, recall10

def initLatentVectors(rows, columns):
    return np.random.randn(rows, columns) / math.sqrt(columns)

def initItemPerVectors(cityUsers, userNum, itemNum, featureK):
    itemVectors4Users = {}
    for city in cityUsers:
        users = cityUsers[city].split(';')
        tmp = np.random.randn(itemNum, featureK) / math.sqrt(featureK)
        for user in users:
            #print("cuilongyin", user)
            itemVectors4Users[int(user)] = tmp
    return itemVectors4Users

def initItemComVectors(userNum, itemNum, featureK):
    itemVectors4Users = {}
    tmp = np.random.randn(itemNum, featureK) / math.sqrt(featureK)
    for user in range(userNum):
            itemVectors4Users[user] = tmp
    return itemVectors4Users


def train(userNum, itemNum, neighborNum, cityUsers, featureK, trainSet, testSet, testRank, epochs, LambdaU, LambdaV, LambdaZ, alpha, lrDecay, filePrefix, isSave, data, negativeNum, log):

    userVectors = initLatentVectors(userNum, featureK)
    itemComVectors = initItemComVectors(userNum, itemNum, featureK)
    itemPerVectors = initItemPerVectors(cityUsers, userNum, itemNum, featureK)

    finalEpoch = epochs
    lastRMSE = 1000
    totalStart = time.time()
    for epoch in range(finalEpoch):
        random.shuffle(trainSet)
        start = time.time()
        for row in trainSet:
            #print(row)
            userID = int(row['userID'])
            itemID = int(row['itemID'])
            city = row['city']
            rating = float(row['rating'])
            
            # Since list index in python starts from 0 and user ID
            # in MovieLens dataset start from 1, all IDs should minus one.
            #print(initItemPerVectors.shape())
            #if userID == len(itemPerVectors):
              #continue
            userVector = userVectors[userID]
            itemComVector = itemComVectors[userID][itemID]
            itemPerVector = itemPerVectors[userID][itemID]
            curItemVector = itemComVector + itemPerVector
            
            predictedRating = predictRating(userVector, curItemVector, data)
            error = computeError(rating, predictedRating)

            # calculate the gradients of user and item
            deltaU = LambdaU * userVector - error * curItemVector
            deltaV = LambdaV * itemComVector - error * userVector
            deltaZ = LambdaZ * itemPerVector - error * userVector

            # update the preferences of user
            userVector -= alpha * deltaU
            itemComVector -= alpha * deltaV
            itemPerVector -= alpha * deltaZ

            # randomly select k neighbor from the current city
            curCityUsers = cityUsers[city].split(';')
            #print(curCityUsers)
            if len(curCityUsers) < neighborNum: 
                neighbors = random.sample(curCityUsers, len(curCityUsers))
            else:
	            neighbors = random.sample(curCityUsers, neighborNum)
            #print(neighbors)
            for nb in neighbors:
                if int(nb) != userID:
                    NBitemVector = itemComVectors[int(nb)][itemID]
                    NBitemVector -= alpha * deltaV


        testRMSE, testMAE, testRmseNew, testRmseOld, count, newCount, oldCount = computeRMSE(testSet, userVectors, itemComVectors, itemPerVectors, data)
        end = time.time()

        if epoch % 5 == 0:
            p5, r5, p10, r10 = computePrecision(testRank, userVectors, itemComVectors, itemPerVectors, itemNum, data)
            end = time.time()
            print(epoch, testRMSE, testMAE, p5, r5, p10, r10, end - start)
            log.write(str(epoch) + ',' + str(testRMSE) + ',' + str(testMAE) + ',' + str(p5) + ',' + str(
                r5) + ',' + str(p10) + ',' + str(r10) + ',' + str(end - start) + '\n')
        #else:
            #print(epoch, testRMSE, testMAE,  end - start)
            #log.write(str(epoch) + ',' + str(testRMSE)+ ',' + str(testMAE) + ',' + str(end - start) + '\n')
        log.flush()
        # if lastRMSE < testRMSE or lastRMSE - testRMSE < 0.00001:
        #    break
        if lastRMSE > testRMSE:
            lastRMSE = testRMSE
            # else:
            #    break

        # deday learning rate
        if epoch % 10 == 0:
            alpha = alpha * lrDecay

    totalEnd = time.time()
    print('RMSE of MF for testing set:', testRMSE)
    print('MAE of MF for testing set:', testMAE)
    print('time cost: ', totalEnd - totalStart)
    return testRMSE,testMAE

def readFile(path):
    uiratings = []
    with open(path) as fd:
        for line in fd:
            rec = line.strip().split(',')
            uiratings.append({'userID': rec[0], 'itemID': rec[1], 'city':"Urbana-Champaign", 'rating': rec[2]}) 
    return uiratings

def readFile4Rank(path):
    uiRatings = {}
    with open(path) as fd:
        for line in fd:
            rec = line.strip().split(',')
            user = int(rec[0])
            item = int(rec[1])
            rating = int(rec[3])
            if user not in uiRatings:
            #if not uiRatings.has_key(user):
                uiRatings[user] = []
            uiRatings[user].append(str(item) + ',' + str(rating))
    return uiRatings

def generateUsers2Rank(userNum, userNum2Rank):
    users = []
    cnt = 0
    while True:
        #print(userNum, userNum2Rank)
        user = random.randint(1, userNum)
        if user not in users:
            users.append(user)
            cnt += 1
        if cnt == userNum2Rank:
            break
    return users

def readFile4RankRandom(path, users2Rank):
    uiRatings = {}
    with open(path) as fd:
        for line in fd:
            rec = line.strip().split(',')
            user = int(rec[0])
            item = int(rec[1])
            rating = 1
            if user in users2Rank:
                if user not in uiRatings:
                #if not uiRatings.has_key(user):
                    uiRatings[user] = []
                uiRatings[user].append(str(item) + ',' + str(rating))
    return uiRatings

def readCityUser(path):
    uiratings = {}
    with open(path) as fd:
        for line in fd:
            rec = line.strip().split(',')
            city = rec[0]
            users = rec[1]
            uiratings[city] = users
    return uiratings

def countNum(path): # return the number of user and item
    uid = []
    iid = []
    cities = []
    ratings = []
    #timeStamp = []
    with open(path, newline = '', encoding = 'ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            uid.append(row[0])
            iid.append(row[1])
            city = 'Urbana-Champaign'
            cities.append(city)
            rating = float(row[2])
            ratings.append(rating)
            #timeStamp.append(row[4])

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

    userNum = len(duid)
    itemNum = len(diid)
    print("finished reading ...")
    return userNum,itemNum

def main(args,i):
        print(f"==============================================={i}==========================================================")
        #DataPreProcessing\BatchFiles_Phoenix_DMF\180_uc.all.rating.cityUser
        filePrefix1 = ('DataPreProcessing/BatchFiles_Phoenix_MetaMF/' + str(i))
        filePrefix2 = ('DataPreProcessing/BatchFiles_Phoenix_DMF/' + str(i))
        trainSet = readFile(filePrefix1 + '_uc.train.rating')#t_ccc_dmf_tist_train_data5.txt
        testSet = readFile(filePrefix1 + '_uc.test.rating')#t_ccc_dmf_tist_test_data5.txt
        cityUsers = readCityUser(filePrefix2 + '_uc.all.rating.cityUser')#t_ccc_dmf_tist_city5.txt
        #User number is: 11935
        #Item number is: 1579
        a,b = countNum((filePrefix1 + '_uc.all.rating'))
        userNum = a #6524
        itemNum = b #3197
        print(userNum, itemNum)
        featureK = args.featureK
        epochs = args.epochs
        LambdaU = args.LambdaU
        LambdaV = args.LambdaV
        LambdaZ = args.LambdaZ
        alpha = args.alpha
        lrDecay = 0.95
        isSave = False
        data = 5  # max rating is 1
        userNum2Rank = int(userNum/2)
        negativeNum = args.negativeNum
        neighborNum = args.neighborNum
        #print("This is a test1 ...")
        users2Rank = generateUsers2Rank(userNum, userNum2Rank)
        #print("This is a test2 ...")
        testRank = readFile4RankRandom(filePrefix1 + '_uc.test.rating', users2Rank)

        filePrefix3 = ('DataPreProcessing/result/')
        output = filePrefix3 + str(i) + 'dmf-tist-n' + str(neighborNum) + '-lambdau-' + str(LambdaU) + '-lambdav-' + str(LambdaV) + '-lambdaz-' + str(LambdaZ) + '-k-' + str(featureK) + '.txt'
        log = open(output, 'w')

        print('epoch, testRMSE, testMAE, p5, r5, p10, r10, time per iter')
        log.write('epoch, testRMSE, testMAE, p5, r5, p10, r10, time per iter\n')

        RMSE, MAE = train(userNum, itemNum, neighborNum, cityUsers, featureK, trainSet, testSet, testRank, epochs, LambdaU, LambdaV, LambdaZ, alpha, lrDecay, filePrefix1, isSave, data, negativeNum, log)
        row = [RMSE, MAE]
        writer.writerow(row)
        log.close



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'dmf')
    parser.add_argument('--featureK', action='store', dest='featureK', default=1, type=int)
    parser.add_argument('--LambdaU', action='store', dest='LambdaU', default=0.7, type=float)
    parser.add_argument('--LambdaV', action='store', dest='LambdaV', default=0.05, type=float)
    parser.add_argument('--LambdaZ', action='store', dest='LambdaZ', default=0.1, type=float)
    parser.add_argument('--alpha', action='store', dest='alpha', default=0.05, type=float)
    parser.add_argument('--epochs', action='store', dest='epochs', default=20, type=int)
    parser.add_argument('--negativeNum', action='store', dest='negativeNum', default=1, type=int)
    parser.add_argument('--neighborNum', action='store', dest='neighborNum', default=2, type=int)
    parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")

    args=parser.parse_args()

    output = 'DataPreProcessing/BatchFiles_Phoenix_DMF/' + 'result' + '.csv'
    with open(output, 'w', newline = '', encoding = 'ISO-8859-1') as out:
      writer = csv.writer(out, sys.stdout, lineterminator = '\n')
      writer.writerow(['RMSE','MAE'])
      for i in range(56,66):
        main(args, i)
