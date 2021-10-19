
import numpy as np
from surprise import AlgoBase
from surprise.utils import get_rng
from surprise import PredictionImpossible
from collections import defaultdict


class SVD_POIsims(AlgoBase):

    def __init__(self, n_factors=40, n_epochs=200, biased=False, init_mean=0,
                 init_std_dev=.1, lr_all=.01,
                 reg_all=.5, lr_bu=None, lr_bi=None, lr_pu=None,
                 lr_qi=None,
                 reg_bu=None, reg_bi=None, reg_pu=None, 
                 reg_qi=None, reg_qj=None,
                 random_state=None, verbose=False,
                 busSimMat = None):
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
        self.reg_qj = reg_qj if reg_qj is not None else reg_all
        self.random_state = random_state
        self.verbose = verbose
        self.centroidRatingDic = defaultdict()
        self.simComputed = False
        self.simDic = defaultdict()
        self.num_predicted = 0      
        self.busSimMat = busSimMat
        AlgoBase.__init__(self)

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)
        self.sgd(trainset)

        return self

    def find_neighbors(self, raw_iid, n_neighbors = 16):
        if raw_iid in self.busSimMat:
            allPOIs = self.busSimMat[raw_iid] # this is also a defaultdic
        else:
            allPOIs = []
        POI_sim_list = []
        for eachPOI in allPOIs:
            if eachPOI in self.trainset._raw2inner_id_items:
                eachSim = allPOIs[eachPOI]
                POI_sim_list.append((eachPOI,eachSim))
        rankedPOIs = sorted(POI_sim_list, key=lambda tup: tup[1], reverse=True)
        rankedPOIs = rankedPOIs[1:n_neighbors+1]
        return rankedPOIs

    def sgd(self, trainset):
        print("Start sgd ...")
        #print(trainset)
        #print(trainset.n_users)
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
                
                # find i's neighbors
                i_raw = trainset.to_raw_iid(i)
                i_j_simList = self.find_neighbors(i_raw)
                jList = [trainset.to_inner_iid(each[0]) for each in i_j_simList]
                simList = [each[1] for each in i_j_simList]

                # update factors
                for f in range(self.n_factors):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    qjfs = []
                    for j in jList:
                        qjfs.append(qi[j, f])
                    pu[u, f] += self.lr_pu * (err * qif - self.reg_pu * puf)
                    qi[i,f] += self.lr_qi * (err * puf - (self.reg_qi + self.reg_qj * sum(simList)) * qif + self.reg_qj * np.dot(simList, qjfs) )

        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi

        #qiT = np.transpose(qi)
        #product = np.matmul(pu,qiT)
        #np.savetxt("mtx1.csv", product, delimiter=",")

        print("Done ...")


    def estimate(self, u, i):
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
       
       

