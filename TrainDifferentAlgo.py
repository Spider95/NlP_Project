
"""
Created on Wed Apr 25 16:42:22 2018

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


data = pd.read_csv('D:\\SecondTerm\\NLP\\Project\\svm-text-classification-api-master\\Data\\train.csv', encoding='utf8').as_matrix()
count_vectorizer = CountVectorizer(analyzer='word', stop_words=nltk.corpus.stopwords.words('english'), strip_accents='ascii')
tfidf_vectorizer = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'), strip_accents='ascii')

### Data With Count Vectorize
X1 = count_vectorizer.fit_transform(row[0] for row in data)
Y1 = [row[1] for row in data]
filename2 = "CountVactorizedWeights.sav"
pickle.dump(count_vectorizer, open(filename2, 'wb'))

### Data With Count TFidf
X2 = tfidf_vectorizer.fit_transform(row[0] for row in data)
Y2 = [row[1] for row in data]
filename2 = "TFIdfVactorizedWeights.sav"
pickle.dump(tfidf_vectorizer, open(filename2, 'wb'))

test_size = 0.33
seed = 7
### Divide Data With Count Vectorize
X_train1, X_test1, Y_train1, Y_test1 = model_selection.train_test_split(X1, Y1, test_size=test_size, random_state=seed)
### divide Data With Count TFidf
X_train2, X_test2, Y_train2, Y_test2 = model_selection.train_test_split(X2, Y2, test_size=test_size, random_state=seed)

#KNN Neighbor Count
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train1, Y_train1) 
KNeighborsClassifier(...)
filename = 'KNN_Count_model.sav'
pickle.dump(neigh, open(filename, 'wb'))

#KNN Neighbor TFidf
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train2, Y_train2) 
KNeighborsClassifier(...)
filename = 'KNN_Tfidf_model.sav'
pickle.dump(neigh, open(filename, 'wb'))

#NaiveBias Count
clf = MultinomialNB()
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
clf.fit(X_train1, Y_train1)
filename = 'NaiveBias_Count_model.sav'
pickle.dump(clf, open(filename, 'wb'))

#NaiveBias Tfidf
clf = MultinomialNB()
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
clf.fit(X_train2, Y_train2)
filename = 'NaiveBias_Tfidf_model.sav'
pickle.dump(clf, open(filename, 'wb'))


#Logistic regression Count
model = LogisticRegression()
model.fit(X_train1, Y_train1)
filename = 'LogisticRegression_Count_model.sav'
pickle.dump(clf, open(filename, 'wb'))

#Logistic regression Count
model = LogisticRegression()
model.fit(X_train2, Y_train2)
filename = 'LogisticRegression_Tfidf_model.sav'
pickle.dump(clf, open(filename, 'wb'))

