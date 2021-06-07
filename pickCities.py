import csv
import sys
import os 

dir_read    = os.path.dirname(os.path.realpath(__file__)) + "\\processedFile.csv"
dir_write   = os.path.dirname(os.path.realpath(__file__)) + "\\"
#open(dir_write, 'a').close()
#user_id	bus_id	rating	date	city	state	lat	lon	text

def extractCity(dir_read, dir_write, city, tittle, append):
    if append == 1:
        x = 'a'
    else:
        x = 'w'
    with open(dir_read, 'r', newline = '', encoding = 'ISO-8859-1') as readfile, \
         open(dir_write, x , newline = '', encoding = 'ISO-8859-1') as writefile:
    
             reader = csv.reader(readfile)
             writer = csv.writer(writefile, sys.stdout, lineterminator = '\n')
             if tittle == 1:
                 writer.writerow(['user_id', 'bus_id', 'rating', 'date','lat','lon'] )
             next(reader)
             
             for row in reader:
                if  len(row) > 1:
                    if city == row[4]:
                        uid    = row[0]
                        iid    = row[1]
                        rating = row[2]
                        date   = row[3]
                        lat    = row[6]
                        lon    = row[7]
                        #text   = row[8]
                        line = [uid, iid, rating, date, lat, lon]
                        writer.writerow(line)
                else:
                    print("There is an anomaly")
                    print(row)
        
def extractCities(dir_read, dir_write_base,fileName, citiList):
     
    extractCity(dir_read, dir_write, citiList[0], 1, 0 )
    for i in range(1,len(citiList)):
        extractCity(dir_read, dir_write, citiList[i], 0, 1 )

def main():
    
    citiList=['Urbana','Champaign']
    fileName = ''
    for eachName in citiList:
        fileName = fileName + eachName + ".csv"
    extractCities(dir_read, dir_write_base, fileName, citiList)

if __name__=="__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    