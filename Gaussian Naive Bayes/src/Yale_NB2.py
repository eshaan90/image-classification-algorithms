# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 19:18:43 2018

@author: AnupamaKesari
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 13:09:33 2018

@author: AnupamaKesari
"""
import scipy
import scipy.io
import numpy as np
from sklearn.decomposition import PCA
import nbImp as NB
import time
import matplotlib.pyplot as plt

# READING THE DATASET
mat = scipy.io.loadmat("YaleB.mat")  #Change path of YaleB.mat according to system

dataset = np.transpose(mat['YaleB'][0][:][1])
labels = [mat['YaleB'][0][:][0][0]]*mat['YaleB'][0][:][1].shape[1]
for x in range(1, mat['YaleB'].shape[0]):
    dataset=np.concatenate((dataset,np.transpose(mat['YaleB'][x][:][1])))
    labels = labels+ [mat['YaleB'][x][:][0][0]]*mat['YaleB'][x][:][1].shape[1]
labelInt=[]    
for x in range(0, mat['YaleB'].shape[0]):
    labelInt = labelInt + [x+1]*mat['YaleB'][x][:][1].shape[1]
    
# SPLITTING TO TRAINING AND TESTING FILES
indices = np.random.permutation(dataset.shape[0])
dataset=dataset.tolist()
Dataset_size = 2414
dataset=[dataset[indices[i]] for i in range(Dataset_size)]
labelInt=[labelInt[indices[i]] for i in range(Dataset_size)]
print("Size of Dataset:", Dataset_size)
indices=np.random.permutation(Dataset_size)

accuracies = []
accuracies.append(0)
for j in range(1,100):
    t0 = time.time()
    k=5
    testSize=int(len(indices)/k)
    test_acc=0.0
    final_train_acc=0.0
    print("Number:", j)
 
    for i in range(k):
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
        trainingLabels= np.asarray(trainingLabels)
        trainingLabels.shape = (len(trainingLabels),1)
        testLabels = [labelInt[i] for i in test_idx] 
        testLabels= np.asarray(testLabels)
        testLabels.shape = (len(testLabels),1)
        dataset= np.asarray(dataset)
        training= np.asarray(training)
        test= np.asarray(test)
        
        # FEATURE SELECTION
        my_model = PCA(n_components= j, svd_solver='full')
        newSet = my_model.fit_transform(training)
        #my_model.explained_variance_ratio_.cumsum()
        newTestSet = my_model.transform(test)
 
        Xnew = np.hstack((newSet[:,:j],trainingLabels))
        XTestNew = np.hstack((newTestSet[:,:j],testLabels))
        meanSDValues = NB.meanSDofClass(Xnew)
        predictions = NB.predict(meanSDValues, XTestNew)
        acc=NB.accuracy(XTestNew, predictions)
#        print('Best Accuracy: {0}%'.format(acc))
        test_acc+=acc
    print("Average Test Accuracy over 5-Folds= ", test_acc/k)
    accuracies.append(test_acc/k)

