# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 02:39:32 2018

@author: sayal

Name        :   Sayali Khade
Student ID  :   1001518264
"""

import numpy 
import pandas as pd
import nltk
import pickle
import argparse

from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


parser = argparse.ArgumentParser()

parser.add_argument("-m", "--mode", help="Mode input :-",type=str)
parser.add_argument("-i", "--input", help="Enter the file name :- ",type=str)

args = parser.parse_args()

#if args.mode=='train':
 #   fname=args.input
#reading the csv file and it returns a dataframe
doc1=pd.read_csv('voting_data.csv')
#    doc1=pd.read_csv(fname)
#print(doc1)

'''
The Tf-Idf vectorizer is used for feature extraction and uses the following parameters
sublinear_tf: applies sublinear tf scaling ie it replaces term-frequency with 1+log(tf)
norm : used for normalizing term vectors
encoding : used to decode bytes/files
stop_words : returns appropriate stop list in 'english'
min_df : while building a vocabulary it will ignore the terms that have a document frequency strictly lower than the given threshold value
analyzer : determines whether the feature should be made of a word or character 'n-grams'
token_pattern : regex denotes what a token will comprise of.
'''

tfidf = TfidfVectorizer(sublinear_tf=True, norm='l2', encoding='utf-8', stop_words='english', min_df=1 ,analyzer='word',token_pattern='[a-zA-Z]+')
#fit_transform learns about the vocabulary and idf , return a term document matrix
features = tfidf.fit_transform(doc1['text']).toarray()
labels = doc1['label']

'''
The train_test_split function splits the arrays or matrices into random test and train subsets. 
It makes use of the following parameters as an input :-
1.features : array of features
2.labels : array of labels/class
3.stratify=labels :where data is split in stratified way
4.random_state=0  : as values int,random_state is the seed value used by random number generator
5.test_size and train_size : represents the proportion of training and test data

Output : returns train-test split of inputs
'''
train_feature, test_feature, train_class, test_class = train_test_split(features, labels, stratify=labels, random_state=0,test_size=0.25,train_size=0.75)

'''
The LinearSVC function is the Linear Support Vector Classification
It makes use of the following parameters as an input :-
1.random_state=0 : it is the seed for random number generator

The fit method fits the model according to the training data.
'''

linearsvm = LinearSVC(random_state=0).fit(train_feature, train_class)
print("Test set score: {:.3f}".format(linearsvm.score(test_feature, test_class)))
print("Training set score: {:.3f}".format(linearsvm.score(train_feature, train_class)))

#prdeicts the class labels of test_feature
prediction = linearsvm.predict(test_feature)

print("Confusion matrix:")
'''
The pd.crosstab computes a cross-tabulation of 2 or more factors.
Takes input as following parameters : test_class, the prediction ,rownames  : if true should match the number of row arrays passed,
colnames =matches the number of column arrays passed and margins =adds the row/column margin
crosstab function returns dataframe
'''
print(pd.crosstab(test_class, prediction, rownames=['True'], colnames=['Predicted'], margins=True))

#calculates the score sing CV;in our case stratified 10-fold cv
scores = cross_val_score(linearsvm, features, labels, cv=10)
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))

target_names=labels

#represents text report with the main classification metrics like precision,recall,f1-score
print(classification_report(test_class, prediction, target_names=set(target_names)))
   

#storing the trained model as pickle object
joblib.dump(linearsvm, 'SVM.pkl')
#load a pickle object
trained_model = joblib.load('SVM.pkl')
print("Test set score: {:.3f}".format(trained_model.score(test_feature,test_class)))

vocabulary = 'And as you know, nobody can reach the White House without the Hispanic vote.'
vocabulary = [vocabulary]
#the transform method transforms the input to doument-term matrix
feature = tfidf.transform(vocabulary).toarray()
#passing the feature to model for predction
prediction = trained_model.predict(feature)
print(prediction[0])