"""
        NAME        : SAYALI KHADE
        STUDENT ID  : 1001518264 
"""

import math
filename = './debate.txt'
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter

stop_words = set(stopwords.words('english'))
#documentFreqMap stores idf of all terms in the corpus
documentFreqMap={}

"""
    tokenizationFunction : It converts every paragraph of the entire document into lowercase first 
    and then creates tokens.
    Input   : a list of all paragraphs stored as individual documents
    Output  : returns a list of tokens document wise.
"""
def tokenizationFunction(doc):
    
    #accessing every paragraph and creating tokens and appending it to a tokens list
    for para in doc:
        #converting a list of paragraph into lowercase
        print(doc)
        words=str(para).lower()
        tokens.append(tokenizer.tokenize(words))    
    return tokens   

"""
    preprocessingFunction : For every tokenized documents it is performing preprocessing steps 
    like stopword removal and stemming of tokenized words
    Input   : a list of tokens document wise
    Output  : returns a dict(stop) containing a list of all stemmed words document wise.
"""
def preprocessingFunction(tokens):
    
    for m in range(0,len(tokens)):
        for n in range(0,len(tokens[m])): 
            #checks if the token is not a stopword then 
            #will perform stemming on it and store it in stop
            if tokens[m][n] not in stop_words :
                tokens[m][n]=stemmer.stem(tokens[m][n])
                stop[m].append(tokens[m][n]) 
    return stop             


"""
    calculateTF : For every prepocessed/stemmed token in the document it calculates its term frequency
    document wise
    Input   : a list of stemmed tokens document wise
    Output  : returns a list called count which contains a dictionary that maps token to its frequency 
    document wise.
    Using counter from collections library to count the frequency of tokens.
    Counter returns a dictionary.
"""
def calculateTF(stop):

    for key in stop :
        count.append(dict(Counter(stop.get(key))))  
    #print(count) 
    return count    


"""
    getidf : For every token of query it returns the idf value.
    documentFrqMap : dictionary that contains all the calculated idf values of stemmed tokens.
    so the input token is compared with this dictionary to get the idf value.
    Input   : tokens of query
    Output  : returns idf value of the token if it is present else returns -1
    
"""
def getidf(tokens1):
    #print("token",tokens1)
    if tokens1 in documentFreqMap:
        val= documentFreqMap[tokens1]
    else :
        val =-1
    return val

"""
    getqvec : It performs the preprocessng steps of tokenization,stopword removal and stemming 
    on the query string.Calculates the tf-idf of the query.     
    Input   : query string
    Output  : returns normalized tf-idf query vector
    
"""
def getqvec(qstring):
  
   processedtokens=[]
   frequencycnt={}
   vectoridf={}
   result={}
   
   #performing tokenization on query string
   vectokens=tokenizer.tokenize(qstring.lower())
   
   #checking if token is not a stopword then performing stemming on the token and
   #appending to processedtoken list
   for vt in vectokens:
       if vt not in stop_words:
           vt=stemmer.stem(vt)
           processedtokens.append(vt)
           
   #Calculating term frequency of words in the query
   #Citation[1]-Term frequency calculation.Reference mentioned at the end.      
   for word in processedtokens:
         termcount = frequencycnt.get(word,0)
         frequencycnt[word] = termcount + 1        
   #print(frequencycnt)   
   
   #Storing idf of the tokens 
   #in case the function getidf returns -1 the df for that token is 1
   for tokens1 in processedtokens:
       val1=getidf(tokens1)
       if val1 == -1 :
          vectoridf[tokens1]=math.log10(N/1); 
       else:   
           vectoridf[tokens1]=val1    
   #print(vectoridf)
   
   #calculating tf-idf weight for query string
   for allterms in frequencycnt :
       if allterms in vectoridf:
           result[allterms] =(1+math.log10(frequencycnt[allterms]))* vectoridf[allterms] 
   #print(result)
   
   #calculating normalized length of the query string
   normalsum=0
   for normalizedval in result:
       normalsum=normalsum+(result[normalizedval]*result[normalizedval]) 
   normalsum=math.sqrt(normalsum)  
   #print(normalsum)
   
   #calculating normalized tf-idf value by dividing the tf-idf of each token with its normalized length
   normalizedtfidfval={}
   for normalizedtfidfvector in result :
       normalizedtfidfval[normalizedtfidfvector]=result[normalizedtfidfvector]/normalsum
   #print(normalizedtfidfval)    
       
   return normalizedtfidfval  

#defining variables
    
tokenizer = RegexpTokenizer('[a-zA-Z]+')
stemmer = PorterStemmer()
frequency={}
tf_idf={}
tf={}
tokens=[]
processeddata=[]
totaldocs=[]
doc =[]
stop=defaultdict(list)
listOfDocuments=[]
i=0
filtered_sentence = []
count=[]

#documentFreqMap=defaultdict(list)
#open and reading text file and storing in doc list paragraph wise
f = open(filename, "r", encoding='UTF-8') 
doc1 = f.read() 
f.close()

doc=doc1.split("\n\n")
totaldocs=doc
N=len(doc)
#preprocessing of documents
tokens=tokenizationFunction(doc)
stop=preprocessingFunction(tokens)
count=calculateTF(stop)

#Calculating idf of terms
for doc in count:
    for key in doc:
        countKey = 0
        #iterate over count to get different documents
        for mapKey in count:  
            if key in mapKey:
                countKey = countKey + 1
        documentFreqMap[key] =math.log10(N/countKey)
#print(documentFreqMap)
        
tfidfMap={}
docwisetfidfList = []

#calculatinf tf-idf of all terms in documents
for tfs in count:
    tempMap = {}
    for key in tfs:
        #storing tf idf of every elemnt doc wise
        tf_idf=(1+ math.log10(tfs[key]))*documentFreqMap[key]
        tempMap[key]=tf_idf
    docwisetfidfList.append(tempMap)
#print(docwisetfidfList)

normalizedvectorlength=[]

#calculating the normalized length for every document
for tfidfs in docwisetfidfList:
    normalizedsum=0
    for normalizedval in tfidfs:
        normalizedsum=normalizedsum+(tfidfs[normalizedval]*tfidfs[normalizedval])
    normalizedvectorlength.append(math.sqrt(normalizedsum))
#print(normalizedvectorlength)
   
normalizedtfidfvalList=[]
cnter=0
#calculating normalized tf-idf of every term by dividing tf-idf of every token with its normalized length
for normalterms in docwisetfidfList:
    tempMap = {}
    for item in normalterms:
        tempMap[item]=normalterms[item]/normalizedvectorlength[cnter]
    normalizedtfidfvalList.append(tempMap)
    cnter=cnter+1
#print(normalizedtfidfvalList)    
    
"""
    query : It performs the cosine similarity between the query string anf the documents     
    Input   : query string
    Output  : returns highest similarity score between the query and the docs and the document which more
    relevant to the query based on the score.
    
"""
def query(qstring):
   
    vectokens=getqvec(qstring)
    #print(vectokens)
    ss={}  
    ssList = []
    cnter = 0
    for docterms in normalizedtfidfvalList:
        similaritysum=0
        for terms in vectokens:                    
            if terms in docterms:
               similaritysum=similaritysum+(vectokens.get(terms,1)*docterms.get(terms,1))
        ss[cnter]=similaritysum
        ssList.append(similaritysum)
        cnter = cnter + 1     
    #print(ssList)
   
    highersimilarityval=max(ss.values())
    highersimilarityvalIndex= ssList.index(highersimilarityval)
    #print(highersimilarityval)
    
    if highersimilarityval !=0 :
        paragraph=totaldocs[highersimilarityvalIndex]
    else:
        highersimilarityval=0.0000
        paragraph="No Match Found" 
        #print(paragraph)

    return paragraph+"\n",highersimilarityval
   
"""
#Testing one of the tranined data
similaritymatchedparagraph, similaritymatchscore= query("clinton first amendment kavanagh")
#similaritymatchscore=query("The alternative, as cruz has proposed, is to deport 11 million people from this country")
print(similaritymatchedparagraph)
print(similaritymatchscore)
#print(similaritysums)
print("%s%.4f" % query("unlike any other time, it is under attack"))
print("%.4f" % getidf(stemmer.stem("beer")))
print(getqvec("The alternative, as cruz has proposed, is to deport 11 million people from this country"))
print("%.4f" % getidf(stemmer.stem("oil")))
print("%s%.4f" % query("clinton first amendment kavanagh"))
print("%s%.4f" % query("vector entropy"))
"""
### REFERENCE USED -TERM FREQUENCY CALCULATION
### CITATION[1] -Ali, Abder-Rahman. “Counting Word Frequency in a File Using Python.” Code Envato Tuts+, 1 July 2016, code.tutsplus.com/tutorials/counting-word-frequency-in-a-file-using-python--cms-25965.