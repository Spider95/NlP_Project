"""
Created on Tue Apr 17 10:45:10 2018

@author: Ahmed Taher
"""

import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import pickle
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.datasets import cifar10
from keras.models import Sequential, model_from_json
from keras.layers import Convolution2D,Lambda,LSTM, MaxPooling2D, ZeroPadding2D, Activation,GlobalAveragePooling2D, Dropout, Flatten, Dense,Reshape
from keras.layers import merge,Input
import pickle

def extract_features(docs):  
  features = count_vectorizer.fit_transform(docs).toarray()
  return features

def build_classification_report(clf, X_test, Y_test):
  y_true = Y_test
  docs =  X_test
  tfidf = count_vectorizer.transform(docs)
  y_pred = clf.predict(tfidf)
  report = classification_report(y_true, y_pred)
  return report

data = pd.read_csv('/content/NLP/train.csv', encoding='utf8').as_matrix()
count_vectorizer = CountVectorizer(analyzer='word', stop_words=nltk.corpus.stopwords.words('english'), strip_accents='ascii')

X = extract_features(row[0] for row in data)
Y = [row[1] for row in data]
print (X[0])
output = []
# create an empty array for our output
output_empty = [0] * 3
for val in Y:
  if val== "EAP":
    output_row = list(output_empty)
    output_row[0] = 1
    output.append(output_row)
  elif val== "HPL":
    output_row = list(output_empty)
    output_row[1] = 1
    output.append(output_row)
  elif val== "MWS":
    output_row = list(output_empty)
    output_row[2] = 1
    output.append(output_row)
test_size = 0.33
seed = 7

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, output, test_size=test_size, random_state=seed)
# Fit the model on 33%
print (X_train)
print (X_test.shape)
#T = np.array(X)

batch_size = 32
epochs = 30

input_tensor = (24891,)

# create the base pre-trained model

model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=input_tensor))
model.add(Dense(800, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    
model.fit(X_train, np.array(Y_train),
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_test,np.array( Y_test)))
#scores = model.evaluate_generator((X_train, np.array(Y_train)), val_samples=(X_test, np.array(Y_test)))
#print(model.metrics_names, scores)
#print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
print(model.metrics_names)
print("%s: %%" % (model.metrics_names[1]))

model.save_weights('NLPWeights.h5')
print('model saved')
model_json = model.to_json()
with open("NLPWeights.json", "w") as json_file:
  json_file.write(model_json)
filename2 = "VactorizedWeights.sav"
pickle.dump(count_vectorizer, open(filename2, 'wb'))