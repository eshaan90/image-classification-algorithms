#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 21:41:49 2018

@author: MyReservoir
"""

from keras.datasets import mnist 
from sklearn.linear_model import LogisticRegression
import pandas as pd


### Load the MNIST dataset
(X_train,y_train),(X_test,y_test)=mnist.load_data()

xtrain,xtest=[],[]

### Flatten the dataset into 2-D
for i in range(len(X_train)):
    xtrain.append(X_train[i].flatten())
    
for i in range(len(X_test)):    
    xtest.append(X_test[i].flatten())

logisticRegr = LogisticRegression(solver='lbfgs')
logisticRegr.fit(xtrain, y_train)

predictions = logisticRegr.predict(xtest)


# Compute accuracy of the model
score = logisticRegr.score(xtest, y_test)
print("Accuracy= ", score)

### Perform one-hot encoding on predicted labels
encoded=[]
for value in predictions:
    label=[0 for _ in range(10)]
    label[value]=1
    encoded.append(label)
    
df = pd.DataFrame(encoded)
df.to_csv ('lr.csv', header=False, index=False)