# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 20:16:27 2018

@author: AnupamaKesari
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 16:21:24 2018

@author: AnupamaKesari
"""
import mnist
import scipy.misc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy
import nbImp as NB
import time

# READING THE DATASET
train_images = mnist.train_images()
scipy.misc.toimage(scipy.misc.imresize(train_images[2,:,:] * -1 + 256, 10.))
train_labels = mnist.train_labels()
train_labels.shape = (60000,1)
test_images = mnist.test_images()
test_labels = mnist.test_labels()
test_labels.shape = (10000,1)
x = train_images.reshape((train_images.shape[0], train_images.shape[1] * train_images.shape[2]))
y = test_images.reshape((test_images.shape[0], test_images.shape[1] * test_images.shape[2]))
#train_images = x.reshape((x.shape[0], 28,28))
dataset = numpy.concatenate((x,y)).tolist()
labelInt = numpy.concatenate((train_labels,test_labels)).tolist()
Dataset_size = 70000
# SPLITTING TO TRAINING AND TESTING FILES
print("Size of Dataset:", Dataset_size)
indices=numpy.random.permutation(Dataset_size)


accuracies = []
accuracies.append(0)
for j in range(1,100):
    
    t0 = time.time()
    k=5
    testSize=int(len(indices)/k)
    test_acc=0.0
    final_train_acc=0.0
    
    for i in range(k):
        print("\nFold Number:", i+1)
        start=testSize*i
        test_idx=indices[start:start+testSize]
        training_idx1=indices[:start]
        training_idx2=indices[start+testSize:]
        training_idx=numpy.append(training_idx1,training_idx2)
        training_idx = training_idx.tolist()
        test_idx = test_idx.tolist()
        training=[]
        test=[]
        for x in range(len(test_idx)):
            test.append(dataset[test_idx[x]])   
        for x in range(len(training_idx)):
            training.append(dataset[training_idx[x]])
        
        trainingLabels = [labelInt[i] for i in training_idx] 
        trainingLabels= numpy.asarray(trainingLabels)
        trainingLabels.shape = (len(trainingLabels),1)
        testLabels = [labelInt[i] for i in test_idx] 
        testLabels= numpy.asarray(testLabels)
        testLabels.shape = (len(testLabels),1)
        dataset= numpy.asarray(dataset)
        training= numpy.asarray(training)
        test= numpy.asarray(test)
        
        # FEATURE SELECTION
        my_model = PCA(n_components= j, svd_solver='full')
        newSet = my_model.fit_transform(training)
        newTestSet = my_model.transform(test)
        
        #Best Model
        Xnew = numpy.hstack((newSet[:,:j],trainingLabels))
        XTestNew = numpy.hstack((newTestSet[:,:j],testLabels))
        meanSDValues = NB.meanSDofClass(Xnew)
        predictions = NB.predict(meanSDValues, XTestNew)
        acc=NB.accuracy(XTestNew, predictions)
        print('Best Accuracy: {0}%'.format(acc))
        test_acc+=acc
    print("Average Test Accuracy over 5-Folds= ", test_acc/k)
    accuracies.append(test_acc/k)

