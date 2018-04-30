# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 19:25:16 2018

@author: Ahmed Taher
"""


from keras.models import model_from_json  
from keras.optimizers import SGD
import cv2
import numpy as np
import keras
from keras.models import Model
import pickle
import pandas as pd
from sklearn.metrics import classification_report

arr =["EAP","HPL","MWS" ]

# load json and create model
json_file = open('NLPWeightsFinal.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("NLPWeightsFinal.h5")
print("Loaded model from disk")
loaded_model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])    

loaded_model2 = pickle.load(open("VactorizedWeightsFinal.sav", 'rb'))
data2 = pd.read_csv('D:\\SecondTerm\\NLP\\Project\\svm-text-classification-api-master\\Data\\test.csv', encoding='utf8').as_matrix()
X_TESTT = loaded_model2.transform(row[0] for row in data2).toarray()
Y_TESTT = [row[1] for row in data2]
T = loaded_model2.transform(["Adolphe Le Bon, clerk to Mignaud et Fils, deposes that on the day in question, about noon, he accompanied Madame L'Espanaye to her residence with the francs, put up in two bags."]).toarray()
U = np.array(T)
print (T[0])
print (U)
classes = loaded_model.predict(X_TESTT)
TTT=[]
for x in range(0, 28):
    print(classes[x])
    print(np.argmax(classes[x]))
    print(arr[np.argmax(classes[x])])
    print (Y_TESTT[x])
    TTT.append(arr[np.argmax(classes[x])])
    print ('*******************')
report = classification_report(Y_TESTT, TTT)
print (report)
C = 0 
for x in range (0,len(Y_TESTT)):
    if Y_TESTT[x]==TTT[x]:
      C = C +1   
print ("accuracy  :",(C/len(Y_TESTT))*100)