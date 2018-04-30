# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 17:06:22 2018

@author: Ahmed Taher
"""

import pickle
import pandas as pd
from sklearn.metrics import classification_report


data2 = pd.read_csv('D:\\SecondTerm\\NLP\\Project\\svm-text-classification-api-master\\Data\\test.csv', encoding='utf8').as_matrix()

#load Count
filename1 = "CountVactorizedWeights.sav"
loaded_model = pickle.load(open(filename1, 'rb'))

#load Tfidf
filename2 = "TFIdfVactorizedWeights.sav"
loaded_model2 = pickle.load(open(filename2, 'rb'))

#Make Test With Count
X_TESTT1 = loaded_model.transform(row[0] for row in data2)
Y_TESTT1 = [row[1] for row in data2]

#Make Test With Tfidf
X_TESTT2 = loaded_model2.transform(row[0] for row in data2)
Y_TESTT2 = [row[1] for row in data2]




#NaiveBias Test Count
filenameNaiveBias1 = 'NaiveBias_Count_model.sav'
NaiveBias = pickle.load(open(filenameNaiveBias1, 'rb'))
predict = NaiveBias.predict(X_TESTT1)
report = classification_report(Y_TESTT1, predict)
print ("NaiveBias Count")
print (report)
C = 0 
for x in range (0,len(Y_TESTT1)):
    if Y_TESTT1[x]==predict[x]:
      C = C +1   
print ("accuracy  :",(C/len(Y_TESTT1))*100)

#NaiveBias Test Tfidf
filenameNaiveBias2 = 'NaiveBias_Tfidf_model.sav'
NaiveBias2 = pickle.load(open(filenameNaiveBias2, 'rb'))
predict = NaiveBias2.predict(X_TESTT2)
report = classification_report(Y_TESTT2, predict)
print ("NaiveBias Tfidf")
print (report)
C = 0 
for x in range (0,len(Y_TESTT2)):
    if Y_TESTT2[x]==predict[x]:
      C = C +1   
print ("accuracy  :",(C/len(Y_TESTT1))*100)

print ("---------------------------")


#Logistic regression Test Count
filenameLG1 = 'LogisticRegression_Count_model.sav'
LG_model = pickle.load(open(filenameLG1, 'rb'))
predict = LG_model.predict(X_TESTT1)
report = classification_report(Y_TESTT1, predict)
print ("Logistic regression Count")
print (report)
C = 0 
for x in range (0,len(Y_TESTT1)):
    if Y_TESTT1[x]==predict[x]:
      C = C +1   
print ("accuracy  :",(C/len(Y_TESTT1))*100)

#Logistic regression Test Tfidf
filenameLG2 = 'LogisticRegression_Tfidf_model.sav'
LG_model2 = pickle.load(open(filenameLG2, 'rb'))
predict = LG_model2.predict(X_TESTT2)
report = classification_report(Y_TESTT2, predict)
print ("Logistic regression Tfidf")
print (report)
C = 0 
for x in range (0,len(Y_TESTT2)):
    if Y_TESTT2[x]==predict[x]:
      C = C +1   
print ("accuracy  :",(C/len(Y_TESTT2))*100)

print ("---------------------------")


#Knn Test Count
filenameKNN = 'KNN_Count_model.sav'
KNN_model = pickle.load(open(filenameKNN, 'rb'))
predict = KNN_model.predict(X_TESTT1)
report = classification_report(Y_TESTT1, predict)
print ("KNN Count")
print (report)
C = 0 
for x in range (0,len(Y_TESTT1)):
    if Y_TESTT1[x]==predict[x]:
      C = C +1   
print ("accuracy  :",(C/len(Y_TESTT1))*100)

#Knn Test Tfidf
filenameKNN2 = 'KNN_Tfidf_model.sav'
KNN_model2 = pickle.load(open(filenameKNN2, 'rb'))
predict = KNN_model2.predict(X_TESTT2)
report = classification_report(Y_TESTT2, predict)
print ("KNN Tfidf")
print (report)
C = 0 
for x in range (0,len(Y_TESTT2)):
    if Y_TESTT2[x]==predict[x]:
      C = C +1   
print ("accuracy  :",(C/len(Y_TESTT2))*100)

print ("---------------------------")