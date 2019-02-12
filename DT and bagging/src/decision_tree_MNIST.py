#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 16:35:44 2018

@author: MyReservoir
"""


import decision_tree as dt
from sklearn.decomposition import PCA
import numpy as np
import time


##############Variables Initialization###########
"""
Only change test_counts, training_counts and pca comps to check for different values.
test_counts(max)=10000
training_counts(max)=60000

"""
print("Decision Tree implementation for MNIST dataset")

rows = []
rows_total = []
rows_test=[]
rows_test_total=[]
training_labels=[]
testing_labels=[]

dataset_size=1000
print("DataSet Size:",dataset_size)
#test_counts=25 #Change this value to alter the number of images you wish to classify
#training_counts=100 #Change this value to alter the number of images you wish to use to train the model
pca_comps=50 #Change this function to change the number of pca components
path = "/Users/MyReservoir/Library/Mobile Documents/com~apple~CloudDocs/My Program/NCSU ECE Department Docs/Semester 2/ECE 759- Pattern Recognition/Project/Team3/Code/"

Dataset = "training"
train_data = list(dt.read(Dataset, path))
Dataset = "testing"
test_data = list(dt.read(Dataset, path))

##############Training and Testing DataSet Creation############
indices = np.random.permutation(dataset_size)

k=5
testSize=int(len(indices)/k)
trainSize=len(indices)-testSize
final_test_acc=0.0
final_train_acc=0.0
dataset=train_data[:trainSize]
for i in range(testSize):
    dataset.append(test_data[i])

t0=time.time()
for i in range(k):
    print("\nFold Number:", i+1)
    start=testSize*i
    test_idx=indices[start:start+testSize]
    training_idx1=indices[:start]
    training_idx2=indices[start+testSize:]
    training_idx=np.append(training_idx1,training_idx2)
    training_idx = training_idx.tolist()
    test_idx = test_idx.tolist()
    training=[]
    test=[]
    for i in range(len(training_idx)):
        label, pixels = dataset[training_idx[i]]
        record = (pixels.flatten()).tolist()
        training_labels.append(label)
        rows_total.append(record)
        
        
    for i in range(len(test_idx)):
        label, pixels = dataset[test_idx[i]]
        record = (pixels.flatten()).tolist()
        testing_labels.append(label)
        rows_test_total.append(record)
    
    ############# Feature Extraction ##############   
    FinalTrain=[]
    my_model = PCA(n_components= pca_comps, svd_solver='full')
    newSet = my_model.fit_transform(rows_total).tolist()
    newtestSet=my_model.transform(rows_test_total).tolist()
    
    ############# Model Building ##############   
    
    for i in range(len(rows_total)):
        newSet[i].append(training_labels[i])
    b = dt.buildtree(newSet)
    dt.prune(b,0.1)

    ############# Classification of Test Records ##############   
    number = 0
    accuracy=0
    for i in range(testSize):
        a = dt.classify(newtestSet[i], b)
        for key in a.keys():
            if(key == testing_labels[i]):
                number = number + 1
               
    ############# Accuracy Calculations ##############   
    
    accuracy = (number/testSize)* 100
    final_test_acc+=accuracy    
    print('Test accuracy:', accuracy)
    
    ############# Classification of Training Records ##############   
    number = 0
    accuracy=0
    train_label=[]
    for i in range(trainSize):
        train_label.append(newSet[i].pop(-1))
        c = dt.classify(newSet[i], b)
        for key in c.keys():
            if(key == train_label[i]):
                number = number + 1
               
    ############# Accuracy Calculations ##############   
    
    accuracy = (number/trainSize)* 100
    final_train_acc+=accuracy    
    print('Training accuracy:', accuracy)


print("\nFinal Train accuracy= ",final_train_acc/k)
print("Final Test accuracy= ",final_test_acc/k)

t1=time.time()

t=t1-t0
print("Total Time Taken=",t)


#MNIST
#mingain=[0,0.5,1,1.3,2]
#accuracy=[72,73.5,78,100,100]
#sample_size=[250,1000,5000,10000]
#tt=[85.26,1822.11,16443.34,8563.23]
#acc_train_plt=[100,100,100,100]
#acc_test_plt=[83.2,87.2,74.5]


