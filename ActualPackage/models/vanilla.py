
import numpy as np
from surprise import AlgoBase
from surprise.utils import get_rng
from surprise import PredictionImpossible
from collections import defaultdict
from numpy import dot
from numpy.linalg import norm
class GBRS_vanilla(AlgoBase):
    
    def __init__(self, n_factors=100, n_epochs=20, biased=True, init_mean=0,
                 init_std_dev=.1, lr_all=.005,
                 reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
                 reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
                 random_state=None, verbose=False, originalDic=None,
                 numCtds = None):

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.random_state = random_state
        self.verbose = verbose
        self.originalDic = originalDic
        self.centroidRatingDic = defaultdict()
        self.simComputed = False
        self.simDic = defaultdict()
        self.num_predicted = 0      
        self.num_centroids = numCtds
        
        AlgoBase.__init__(self)

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)
        self.sgd(trainset)

        return self

    def sgd(self, trainset):
        print("Strat sgd ...")
        rng = get_rng(self.random_state)

        bu = np.zeros(trainset.n_users, np.double)
        bi = np.zeros(trainset.n_items, np.double)
        pu = rng.normal(self.init_mean, self.init_std_dev,
                        (trainset.n_users, self.n_factors))
        qi = rng.normal(self.init_mean, self.init_std_dev,
                        (trainset.n_items, self.n_factors))
 
        global_mean = self.trainset.global_mean
        if not self.biased:
            global_mean = 0

        for current_epoch in range(self.n_epochs):
            n = 0
            if self.verbose:
                print("Processing epoch {}".format(current_epoch))
            for u, i, r in trainset.all_ratings():

                # compute current error
                dot = 0  # <q_i, p_u>
                for f in range(self.n_factors):
                    dot += qi[i, f] * pu[u, f]
                err = r - (global_mean + bu[u] + bi[i] + dot)

                # update biases
                if self.biased:
                    bu[u] += self.lr_bu * (err - self.reg_bu * bu[u])
                    bi[i] += self.lr_bi * (err - self.reg_bi * bi[i])

                # update factors
                for f in range(self.n_factors):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    pu[u, f] += self.lr_pu * (err * qif - self.reg_pu * puf)
                    qi[i, f] += self.lr_qi * (err * puf - self.reg_qi * qif)
                    #qi[i,f] += self.lr_qi * (err * puf - (self.reg_qi + self.reg_qi2 (sum (s[i]))\
                                                          #+self.reg_qi2 (sum(s[i][j]*qj)) )
                                                          # s dic of dic
                n += 1
        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi
        print("Done ...")

    def impute_train(self, u, i):

        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if self.biased:
            est = self.trainset.global_mean

            if known_user:
                est += self.bu[u]

            if known_item:
                est += self.bi[i]

            if known_user and known_item:
                est += np.dot(self.qi[i], self.pu[u])

        else:
            if known_user and known_item:
                est = np.dot(self.qi[i], self.pu[u])
            else:
                raise PredictionImpossible('User and item are unknown.')

        return est   
    
    def estimateCentroidRating(self, u, i ):
        #ratings is a list containing all ratings from one group
        #[(item,rating),(item,rating),(item,rating),(item,rating) ...]
        ratings =  self.trainset.ur[u]
        filtered = filter(lambda x : x[0] == i, ratings)
        target = list(filtered)
    
        if len(target) == 0: # if the rating is there, return it, or impute it.
            return self.impute_train(u, i)
        else:
            return target[0][1]
        

    def imputeCentroidRatingMat(self):    # complete the centroids'rating matrix
        num_users = 0
        for user in list(self.trainset.all_users()):
            num_users += 1
            if num_users%5 == 0:
                print(f"---|---imputed for {num_users} groups/centroids already ...")
            for item in list(self.trainset.all_items()):
                if user in self.centroidRatingDic:
                    self.centroidRatingDic[user].append((item, self.estimateCentroidRating(user,item)))
                else:
                    self.centroidRatingDic[user] = []
                    self.centroidRatingDic[user].append((item, self.estimateCentroidRating(user,item)))
        return self
    
    
    def computeCosine(self, vec1,vec2):
        #it is easy to debug this way
        var1 = dot(vec1, vec2)
        var2 = (norm(vec1)*norm(vec2))
        result = var1/var2
        return result

    
    def findMostSimilars(self, user):
        centroids = []
        sims  = []
        #print(user)
        userRatingVec = [ x[1] for x in self.originalDic[user]]
        itemVec   = [ self.trainset.to_inner_iid(x[0]) for x in self.originalDic[user] ]

        for eachGroup in list(self.trainset.all_users()):
            groupRatingVec = []
            groupItemRatingVec = self.centroidRatingDic[eachGroup]
            for item_rating in groupItemRatingVec:
                if item_rating[0] in itemVec:
                    groupRatingVec.append(item_rating[1])
            sim = self.computeCosine(userRatingVec, groupRatingVec)
            sims.append(sim)
            centroids.append(eachGroup)
        sorted_centroids = [ctd for sims, ctd in sorted(zip(sims, centroids),key=lambda pair: pair[0])]
        sorted_sims      = [sims for sims, ctd in sorted(zip(sims, centroids),key=lambda pair: pair[0])]
        return sorted_centroids[:self.num_centroids], sorted_sims[:self.num_centroids]
        
    
    def computeSimMatrix(self): # for each group, 
        print("Strat calculating sim ....") 
        original_dic_complete = self.originalDic  
        #print("---Strat calcculating centroids rating matrix ....")
        self.imputeCentroidRatingMat()
        #print("---Done")
        n = 0
        # when finding the most similar centroids, search for several centroids 
        # instead of one, try 2, 3,  ... 10 
        for originalUser in original_dic_complete:
            #user_vec = original_dic_complete[originalUser]
            centroidsVec, correspondingSims = self.findMostSimilars(originalUser)
            self.simDic[originalUser] = (centroidsVec, correspondingSims)
            #print(centroidsVec)
            n += 1
            if n%100 == 0:
                print(f"simDic is calculating, {n} users in original are updated...")
        print("Done sim calculating ....")
        return self          
    
    def estimate(self, u, i):
        #print(f"user {u}, item {i}")
        self.num_predicted += 1
        
        u = u.split('UKN__')[1] #surprise will ad UKN__ infront of every user index since 
                                # it cannot find it in the oringinal trainset
        if self.simComputed == False:
            self.computeSimMatrix()
            self.simComputed = True

        (rankedCtd, correspondingSims) = self.simDic[u]
        
        vecList = []
        for eachCtd in rankedCtd:
            vecList.append(np.array([x[1] for x in self.centroidRatingDic[eachCtd]]))
        #===============================================================
        SUM = 0 * vecList[0]

        for index in range(len(correspondingSims)):
            SUM = SUM + correspondingSims[index] * vecList[index]

        rating_vec = SUM/sum(correspondingSims)

        groupRatings = []
        for index in range(len(vecList)):
            groupRatings .append(vecList[index][i])
        #===============================================================
        #change to weighted average might be better, try change this part.
        if self.num_predicted%100 == 0:
            print(f"Have finisehd predicting {self.num_predicted} ratings..." )
        print( f" user: {u} item: { self.trainset.to_raw_iid(i)}  est = {rating_vec[i]}  all the group ratings are {groupRatings} ")
        print('----------')
        return rating_vec[i]
       

