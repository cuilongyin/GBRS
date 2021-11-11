import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
import ActualPackage.models.vanilla as vanila
import ActualPackage.models.POIsims         as POI
import ActualPackage.models.POIsimsPlusPlus as POIpp
import ActualPackage.models.SVD as SVD
import ActualPackage.models.SVD_POIsims as SVD_POIsims
import ActualPackage.functions.methods as functions
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
print("libraries loaded")
def main_vanilla():
    model = vanila.GBRS_vanilla
    fileName = "UC.csv"
    #fileName = "Phoenix.csv"
    startYear = 2007
    min_NO_rating = 9999999999   # total is 576065, filtering is too slow because of the matrix being too large.
    #batch_size = 900
    batch_size = 1000    
    cluster_size = 11      #clusters per batch
    #totalNOB = 1           #number of Batch, not including the test batch
    totalNOB = 33
    factors = 17
    num_of_centroids = 14
    POIsims = 0
    windowSize = 1
    method = 'spectral_ratingGPS' # kmean, spectral_ratingGPS, spectral_pure, cluster_DBSCAN, cluster_FCM
    ratio = 1 # this parameter only is used when  the method =  'spectral_ratingGPS'
    #pickleJarName = "./PickleJar_Phoenix/" #"./PickleJar/"
    pickleJarName = "./PickleJar/" + "batchSize_" + str(batch_size) + "/"
    if not os.path.exists(pickleJarName):
        os.makedirs(pickleJarName)
    #methods = ['kmean', 'spectral_ratingGPS', 'spectral_pure', 'cluster_DBSCAN', 'cluster_FCM']
    #for i in range(1,34):
        #windowSize = i
    #for i in range(11):
        #ratio = i * 0.1
        
 
    result = functions.totalRun(model, fileName, startYear, min_NO_rating,
                       totalNOB, cluster_size, batch_size, num_of_centroids, 
                       factors, POIsims, method, windowSize, ratio, pickleJarName)
    return -result
def optimize_vanilla(cluster_size, factors, num_of_centroids):
    model = vanila.GBRS_vanilla
    fileName = "UC.csv"
    startYear = 2007
    min_NO_rating = 9999999999   # total is 576065, filtering is too slow because of the matrix being too large.
    batch_size = 1000
    cluster_size = int(cluster_size)      #clusters per batch
    totalNOB = int(33770/batch_size)           #number of Batch, not including the test batch
    factors = int(factors)
    num_of_centroids = int(num_of_centroids)
    POIsims = 0
    windowSize = 1
    method = 'spectral_ratingGPS' # kmean, spectral_ratingGPS, spectral_pure, cluster_DBSCAN, cluster_FCM
    ratio = 1 # this parameter only is used when  the method = 'spectral_ratingGPS' 
    pickleJarName = "./PickleJar/" + "batchSize_" + str(batch_size) + "/"
    if not os.path.exists(pickleJarName):
        os.makedirs(pickleJarName)
    #print(pickleJarName)

    result = functions.totalRun(model, fileName, startYear, min_NO_rating,
                       totalNOB, cluster_size, batch_size, num_of_centroids, 
                       factors, POIsims, method, windowSize, ratio, pickleJarName)
    return -result
#0-5: 1.3619   5-10: 1.3775  10-15: 1.3568 15-20: 1.3541  20-25: RMSE: 1.3812

def main_POIsims():
    
    model = POI.GBRS_POIsims
    fileName = "UC.csv"
    startYear = 2007
    min_NO_rating = 9999999998      # 3: 22785 lines
    batch_size = 900     
    cluster_size = 6      #clusters per batch
    totalNOB = 33           #number of Batch, not including the test batch
    factors = 3
    num_of_centroids = 9
    POIsims = 1
    method = 'spectral_ratingGPS'
    windowSize = 1
    ratio = 1 # this parameter only is used when  the method =  'spectral_ratingGPS'
    pickleJarName = "./PickleJar/"
    
    functions.totalRun(model, fileName, startYear, min_NO_rating, 
                       totalNOB, cluster_size, batch_size, num_of_centroids, 
                       factors, POIsims, method, windowSize, ratio, pickleJarName)
    
def main_SVD():
    model = SVD
    fileName = "UC.csv"
    #fileName = "Phoenix.csv"
    startYear = 2007
    min_NO_rating = 9999999999   # total is 576065, filtering is too slow because of the matrix being too large.
    batch_size = 900
    totalNOB = 33           #number of Batch, not including the test batch
    factors = 1
    windowSize = 1
    POIsims = 0
    functions.originalRun(model, fileName, startYear, min_NO_rating,
                       totalNOB,  batch_size,  factors,  POIsims, windowSize)

def main_SVD_POIsims():
    model = SVD_POIsims

    fileName = "UC.csv"
    #fileName = "Phoenix.csv"
    startYear = 2007
    min_NO_rating = 9999999999   # total is 576065, filtering is too slow because of the matrix being too large.
    batch_size = 900
    totalNOB = 33           #number of Batch, not including the test batch
    factors = 1
    windowSize = 1
    POIsims = 1

    functions.originalRun(model, fileName, startYear, min_NO_rating,
                       totalNOB,  batch_size,  factors, POIsims, windowSize)

def main_POIPP():
    model = POIpp.GBRS_POIsimsPP
    fileName = "UC.csv"
    startYear = 2007
    min_NO_rating = 9999999998      # 3: 22785 lines
    batch_size = 900     
    cluster_size = 6      #clusters per batch
    totalNOB = 33           #number of Batch, not including the test batch
    factors = 3
    num_of_centroids = 9
    POIsims = 1
    method = 'spectral_ratingGPS'
    windowSize = 1
    ratio = 1 # this parameter only is used when  the method =  'spectral_ratingGPS'
    pickleJarName = "./PickleJar/"
    
    functions.totalRun(model, fileName, startYear, min_NO_rating, 
                       totalNOB, cluster_size, batch_size, num_of_centroids, 
                       factors, POIsims, method, windowSize, ratio, pickleJarName)

def OPT_function():
    pbounds = {
               'cluster_size' : (1,20),
               'factors' : (1,40),
               'num_of_centroids':(1,20),
               }
            
    optimizer = BayesianOptimization(
        f = optimize_vanilla,
        pbounds=pbounds,
        verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1
        )
    logger = JSONLogger(path="./logs.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.maximize(
        init_points=20,
        n_iter=180
        )  

if __name__ == "__main__":

    main_vanilla()
    #main_POIsims()
    #main_POIPP()
    #main_SVD()
    #main_SVD_POIsims()
    
    #OPT_function()
    

 
#  GBRS, GBRS + review
#  Minor Difference #user: tQBrhzi7ixctWgthHHSHgQ item: ZxQlHVm0pj0ERqpwhEHc6w  est = 3.7819958004220897  all the group ratings are [3.677328258019558, 3.668531629075991, 3.690211985717554, 4.322812821045875, 4.325878979435208, 4.308147836038116, 3.3966278691485887, 3.3871689274501158, 3.4117308888724485, 3.6402864359715954]