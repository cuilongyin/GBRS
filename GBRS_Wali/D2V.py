import csv
import os 
import sys
import gensim
#from scipy import spatial
#from nltk.tokenize import word_tokenize
from callback import EpochLogger

maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

def genStopwords(dir_stopW):
    stoplist = []
    with open(dir_stopW, newline = '', encoding = 'ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile)   
        for row in reader:
            stopWord = row[0]
            stoplist.append(stopWord)
    stopwords = set(stoplist)
    return stopwords
def tokenize(text, stopwords, max_len = 200):
    return [token for token in gensim.utils.simple_preprocess(text, max_len=max_len) if token not in stopwords]
def genTrainset(dir_read, dir_stopW):
    docs = []
    n = 1
    stopwords = genStopwords(dir_stopW)
    with open(dir_read, 'r', newline = '', encoding = 'ISO-8859-1') as readfile:
        reader = csv.reader(readfile) 
        next(reader)
        for row in reader:
            if  len(row) == 9:
                s = tokenize(row[8], stopwords)
                tag = str(n)
                n  += 1
                docs.append(gensim.models.doc2vec.TaggedDocument(words = s,tags = [tag]))
            else:
                print(row)
    return docs
def AggregateCom(filePath,writeTo):
    df = pd.read_csv(filePath)
    print(len(df.drop_duplicates(subset=['bus_id'])))
    gdf = df.groupby(['bus_id'])
    rList = []
    for x in gdf:
        userID = x[0]
        comments = ""
        for _, row in x[1].iterrows():
            comments = comments + row.text
            row = [userID, comments]
        rList.append(row)    

    result = pd.DataFrame(rList, columns = ['bus_id', 'text'])
    result.to_csv(writeTo, encoding='utf-8', index=False)     
def writeVector(readFrom, writeTo, dir_stopW, modelName):
    with open(readFrom, 'r', newline = '', encoding = 'ISO-8859-1') as readfile, \
         open(writeTo,'w', newline = '', encoding = 'ISO-8859-1') as writefile:
    
        stopwords = genStopwords(dir_stopW)
        reader = csv.reader(readfile)
        writer = csv.writer(writefile, sys.stdout, lineterminator = '\n')
        writer.writerow(['user_id', 'bus_id', 'vector'] )
        model = gensim.models.doc2vec.Doc2Vec.load(modelName)
    
        next(reader)
        lineN = 0
        for row in reader:
            lineN += 1
            uid    = row[0]
            iid    = row[1]
            if lineN % 100 == 0:
                print(f"writing line {lineN}...")
            vector = model.infer_vector(tokenize(row[4], stopwords))
            length = len(gensim.utils.simple_preprocess(row[4], max_len=200))
            line = [uid,iid,length,vector]
            writer.writerow(line)
    
def train(dir_read, dir_stopW, max_epochs, vec_size):
    
    tagged_data = genTrainset(dir_read, dir_stopW)
    epoch_logger = EpochLogger()
    model = gensim.models.doc2vec.Doc2Vec(vector_size = vec_size,
                                          min_count   = 2, 
                                          epochs      = max_epochs,
                                          window      = 2,
                                          dm          = 1,
                                          callbacks   = [epoch_logger])
    model.build_vocab(tagged_data)
    
    # dm defines the training algorithm. 
    # If dm=1 means ‘distributed memory’ (PV-DM) 
    # and dm =0 means ‘distributed bag of words’ (PV-DBOW).
    # Distributed Memory model preserves the word order in a document 
    # whereas Distributed Bag of words just uses the bag of words approach, 
    # which doesn’t preserve any word order

    #for epoch in range(max_epochs):
        #print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples = model.corpus_count,
                epochs         = model.epochs)
        # decrease the learning rate
        #model.alpha -= 0.0002
        # fix the learning rate, no decay
        #model.min_alpha = model.alpha
        
    model.save("D2V.model")
#traninig==========================================
#train(dir_read, dir_stopW, 40, 30)


#writeVector2 is a function that converts aggregated comments into vectors. The comments are from all users. 
def writeVector2(readFrom, writeTo, dir_stopW, modelName):
    with open(readFrom, 'r', newline = '', encoding = 'ISO-8859-1') as readfile, \
         open(writeTo,'w', newline = '', encoding='utf-8') as writefile:
    
        stopwords = genStopwords(dir_stopW)
        reader = csv.reader(readfile)
        writer = csv.writer(writefile, sys.stdout, lineterminator = '\n')
        writer.writerow(['bus_id', 'vector'] )
        model = gensim.models.doc2vec.Doc2Vec.load(modelName)
    
        next(reader)
        lineN = 0
        for row in reader:
            lineN += 1
            iid    = row[0]
            if lineN % 100 == 0:
                print(f"writing line {lineN}...")
            vector = model.infer_vector(tokenize(row[1], stopwords))
            vector = vector.tolist()
            #print(vector)
            #print(type(vector))
            line = [iid,vector]
            writer.writerow(line)


dir_read  = os.path.dirname(os.path.realpath(__file__)) + "\\processedFile.csv"
dir_stopW = os.path.dirname(os.path.realpath(__file__)) + "\\stopWords.txt"
#readFrom  = os.path.dirname(os.path.realpath(__file__)) + "\\Urbana_Champaign_Comments.csv"
#writeTo   = os.path.dirname(os.path.realpath(__file__)) + "\\Urbana_Champaign_Vectors_test.csv"
readFrom  = os.path.dirname(os.path.realpath(__file__)) + "\\Urbana_Champaign_BusAggCom.csv"
writeTo   = os.path.dirname(os.path.realpath(__file__)) + "\\Urbana_Champaign_AggVector.csv"


modelName = 'C:\\Users\\cuilo\\Desktop\\Git_Hub_Repo\\GBRS\\GBRS_Wali\\D2V.model'
#infering==========================================
#model = gensim.models.doc2vec.Doc2Vec.load(fileName)
writeVector2(readFrom, writeTo, dir_stopW, modelName)
