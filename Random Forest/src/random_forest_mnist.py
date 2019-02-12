#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 23:56:24 2018

@author: MyReservoir
"""

import pandas as pd
from keras.datasets import mnist 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

### Load the MNIST dataset
(X_train,y_train),(X_test,y_test)=mnist.load_data()


### Flatten the dataset into 2-D
xtrain,xtest=[],[]

for i in range(len(X_train)):
    xtrain.append(X_train[i].flatten())
    
for i in range(len(X_test)):    
    xtest.append(X_test[i].flatten())
    

clf = RandomForestClassifier()
clf.fit(xtrain, y_train)
train_score = cross_val_score(clf, xtrain, y_train, cv=10, scoring='accuracy').mean()
print("Train Accuracy=", train_score)

results = clf.predict(xtest)
test_score=clf.score(xtest,y_test)
print("Testing Accuracy=",test_score)


### Perform one-hot encoding on predicted labels
encoded=[]
for value in results:
    label=[0 for _ in range(10)]
    label[value]=1
    encoded.append(label)
    
df = pd.DataFrame(encoded)
df.to_csv ('rf.csv', header=False,index=False)