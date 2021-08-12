import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
import ActualPackage.models.vanilla as vanila
import ActualPackage.models.POIsims as POI
import ActualPackage.functions.methods as functions

def main_vanilla():
    model = vanila.GBRS_vanilla
    fileName = "UC.csv"
    startYear = 2007
    min_NO_rating = 9999999999   # total is 576065, filtering is too slow because of the matrix being too large.
    batch_size = 1000    
    cluster_size = 3      #clusters per batch
    totalNOB = 33           #number of Batch, not including the test batch
    factors = 3
    num_of_centroids = 30
    POIsims = 0
    method = 'kmean'
    functions.totalRun(model, fileName, startYear, min_NO_rating,
                       totalNOB, cluster_size, batch_size, num_of_centroids, 
                       factors, POIsims, method)
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
    num_of_centroids = 6
    POIsims = 1
    method = 'kmean'
    functions.totalRun(model, fileName, startYear, min_NO_rating, 
                       totalNOB, cluster_size, batch_size, num_of_centroids, 
                       factors, POIsims, method)
    

#def main_
if __name__ == "__main__":
    main_vanilla()
    #main_POIsims()


#  GBRS, GBRS + review
#  Minor Difference #user: tQBrhzi7ixctWgthHHSHgQ item: ZxQlHVm0pj0ERqpwhEHc6w  est = 3.7819958004220897  all the group ratings are [3.677328258019558, 3.668531629075991, 3.690211985717554, 4.322812821045875, 4.325878979435208, 4.308147836038116, 3.3966278691485887, 3.3871689274501158, 3.4117308888724485, 3.6402864359715954]