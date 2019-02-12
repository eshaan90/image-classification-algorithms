#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 17:54:47 2018

@author: MyReservoir
"""


import scipy
import scipy.io
import numpy  as np
from sklearn.decomposition import PCA
import decision_tree as dt
import time

##############Variables Initialization###########
print("\nDecision Tree implementation with Bagging for YaleB dataset")
test_counts=100 #Change this value to alter the number of images you wish to classify
training_counts=500 #Change this value to alter the number of images you wish to use to train the model
pca_comps=80 #Change this function to change the number of pca components

Dataset_size=2414
print("Size of Dataset:", Dataset_size)
##############Training and Testing DataSet Creation#############

mat = scipy.io.loadmat("YaleB.mat")
dataset = np.transpose(mat['YaleB'][0][:][1]).tolist()
for x in range(1, mat['YaleB'].shape[0]):
    dataset = dataset + np.transpose(mat['YaleB'][x][:][1]).tolist()
labelInt=[]    
for x in range(0, mat['YaleB'].shape[0]):
    labelInt = labelInt + [x+1]*mat['YaleB'][x][:][1].shape[1]
    
indices = np.random.permutation(2414)
no_of_trees=5
stepSize=380
testSize=500
test_classify={}
dataset=[dataset[indices[i]] for i in range(Dataset_size)]
labelInt=[labelInt[indices[i]] for i in range(Dataset_size)]
test=dataset[:500]
testLabels=labelInt[:500]
dataset=dataset[500:]
labelInt=labelInt[500:]

k=5
testSize=int(len(indices)/k)
final_test_acc=0.0
final_train_acc=0.0
models=[]

t0=time.time()

for i in range(no_of_trees):
    print("\nNumber of Trees:", i+1)
    start=stepSize*i
    training=dataset[start:start+stepSize]
    trainingLabels=labelInt[start:start+stepSize]

    
    ############# Feature Extraction ##############   

    my_model = PCA(n_components= pca_comps, svd_solver='full')
    newSet = my_model.fit_transform(training).tolist()
    newTestSet = my_model.transform(test).tolist()
    newTrainSet= my_model.transform(training).tolist()
    
    ############# Model Building ##############   
    for k in range(len(newSet)):
        newSet[k].append(trainingLabels[k])
    passingData = newSet[:]    
    models.append(dt.buildtree(passingData))
#    dt.prune(b,0.1)

    
    ############# Classification of Test Records ##############   

    for j in range(len(newTestSet)):
        if j not in test_classify:
            test_classify[j]=[]
        test_classify[j].append(dt.classify(newTestSet[j],models[i]))
    
    
    ############# Accuracy Calculations ##############   
    
d=[]
f=[]
flat=[]
for l in test_classify.values():
    flat=[]
    d=[]
    for m in l:
        d.append(list(m.keys()))
    flat=[item for sublist in d for item in sublist]
    f.append(flat) 

count = 0
accuracy=0
pred={}
for l in range(len(newTestSet)):
    pred=max(set(f[l]), key=f[l].count)
    if(pred==testLabels[l]):
        count+=1
accuracy = (count/len(newTestSet))* 100
print('Test accuracy:', accuracy)


t1=time.time()

t=t1-t0
print("Total Time Taken=",t)
