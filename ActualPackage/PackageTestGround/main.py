import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
import ActualPackage.models.vanilla as vanila
import ActualPackage.models.POIsims as POI
import ActualPackage.functions.methods as functions

def main_vanilla():
    model1 = vanila.GBRS_vanilla
    fileName = "UC.csv"
    startYear = 2007
    min_NO_rating = 9999999999   # total is 576065, filtering is too slow because of the matrix being too large.
    batch_size = 900     
    cluster_size = 1      #clusters per batch
    totalNOB = 33           #number of Batch, not including the test batch
    factors = 3
    num_of_centroids = 9
    POIsims = 0
    functions.totalRun(model, fileName, startYear, min_NO_rating,
                       totalNOB, cluster_size, batch_size, num_of_centroids, 
                       factors, POIsims)

def main_POIsims():
    
    model = POI.GBRS_POIsims
    fileName = "UC.csv"
    startYear = 2007
    min_NO_rating = 4      # 3: 22785 lines
    batch_size = 900     
    cluster_size = 1      #clusters per batch
    totalNOB = 33           #number of Batch, not including the test batch
    factors = 3
    num_of_centroids = 9
    POIsims = 1
    functions.totalRun(model, fileName, startYear, min_NO_rating, 
                       totalNOB, cluster_size, batch_size, num_of_centroids, 
                       factors, POIsims)

    
#def main_
if __name__ == "__main__":
    #main_vanilla()
    main_POIsims()