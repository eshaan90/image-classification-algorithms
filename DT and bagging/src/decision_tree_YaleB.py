#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 16:47:29 2018

@author: MyReservoir
"""

import scipy
import scipy.io
import numpy  as np
from sklearn.decomposition import PCA
import decision_tree as dt
import time


##############Variables Initialization###########
print("\nDecision Tree implementation for YaleB dataset")
#test_counts=100 #Change this value to alter the number of images you wish to classify
#training_counts=500 #Change this value to alter the number of images you wish to use to train the model
pca_comps=80 #Change this function to change the number of pca components

Dataset_size=2414
##############Training and Testing DataSet Creation#############

mat = scipy.io.loadmat("YaleB.mat")
dataset = np.transpose(mat['YaleB'][0][:][1]).tolist()
for x in range(1, mat['YaleB'].shape[0]):
    dataset = dataset + np.transpose(mat['YaleB'][x][:][1]).tolist()
labelInt=[]    
for x in range(0, mat['YaleB'].shape[0]):
    labelInt = labelInt + [x+1]*mat['YaleB'][x][:][1].shape[1]
    
indices = np.random.permutation(2414)

dataset=[dataset[indices[i]] for i in range(Dataset_size)]
labelInt=[labelInt[indices[i]] for i in range(Dataset_size)]
print("Size of Dataset:", Dataset_size)
indices=np.random.permutation(Dataset_size)

k=5
testSize=int(len(indices)/k)
final_test_acc=0.0
final_train_acc=0.0

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
    for x in range(len(test_idx)):
        test.append(dataset[test_idx[x]])   
    for x in range(len(training_idx)):
        training.append(dataset[training_idx[x]])
    
    trainingLabels = [labelInt[i] for i in training_idx] 
    testLabels = [labelInt[i] for i in test_idx] 
    
    
    ############# Feature Extraction ##############   

    my_model = PCA(n_components= pca_comps, svd_solver='full')
    newSet = my_model.fit_transform(training).tolist()
    newTestSet = my_model.transform(test).tolist()
    newTrainSet= my_model.transform(training).tolist()
    
    ############# Model Building ##############   
    for i in range(len(newSet)):
        newSet[i].append(trainingLabels[i])
    passingData = newSet[:]    
    b = dt.buildtree(passingData)
    dt.prune(b,0.1)

    ############# Classification of Train Records ##############   
    count = 0
    for i in range(len(newTrainSet)):
        a = dt.classify(newTrainSet[i], b)    
        for key in a.keys():
            if(key == trainingLabels[i]):
                count = count + 1
            
    ############# Accuracy Calculations for Training DataSet ##############   
    accuracy = (count/len(newTrainSet))* 100
    final_train_acc+=accuracy    
    print('Train accuracy:', accuracy)
    
    
    ############# Classification of Test Records ##############   
    count = 0
    accuracy=0
    for i in range(len(newTestSet)):
        a = dt.classify(newTestSet[i], b)    
        for key in a.keys():
            if(key == testLabels[i]):
                count = count + 1
    ############# Accuracy Calculations ##############   
    accuracy = (count/len(newTestSet))* 100
    final_test_acc+=accuracy
    print('Test accuracy:', accuracy)
   
    
print("\nFinal Train accuracy= ",final_train_acc/k)
print("Final Test accuracy= ",final_test_acc/k)

t1=time.time()

t=t1-t0
print("Total Time Taken=",t)


#mingain=[0,0.5,0.8,1,1.4,1.8,3]
#accuracy=[45.89,48.54,48.96,61.61,100,100,100]
#sample_size=[100,250,500,750,1500,2000,2414]
#tt=[7.39,33.04,141.14,229.27,1528.62,2315.19]
#acc_train_plt=[73.5,79.0,82.3,84.96,87.01,89.07]
#acc_test_plt=[10,20.8,31.4,36.27,49.65,51.04]