# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
'''
README:
    This program performs two functionalities:
        1. Runs the final designed CNN architecture using the selected hyperparameters.
        2. Performs grid search/babysitting using a range of hyperparamters.
    Default running mode is set to 1  
    
    
    Parameters that can be tweaked for this code are found in the main function block and they are described below:        
        batch_size      -explained in the main block
        activations     -change in the main block and it will take effect throughout.
        alpha           -explained in the main block
        mode            -this should be either 0 or 1. Keep 0 when you want to run the final designed architecture.
                            Keep 1 when you want to perform grid search 

    Please do not make any further changes.
'''


#Dependencies
from keras import models,layers,optimizers
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

def load_dataset():
    '''
    Function to load the MNIST dataset from the keras library. It divides the dataset into training and testing and returns the 
    same.
    '''
    (train_images,train_labels),(test_images,test_labels)=mnist.load_data()
    train_images=train_images.reshape((60000,28,28,1))
    train_images=train_images.astype('float32')/255
    
    test_images=test_images.reshape((10000,28,28,1))
    test_images=test_images.astype('float32')/255
    
    train_labels=to_categorical(train_labels)
    test_labels=to_categorical(test_labels)
    
    return (train_images,train_labels,test_images,test_labels)


def model_arch(activation):
    '''
    This function creates the model used for the application of image classification of MNIST dataset.   
    We use two convolutional and two pooling layers, followed by two dense layers.
    It returns the built model.
    '''
    model=models.Sequential()
    model.add(layers.Conv2D(32,(3,3),activation=activation,input_shape=(28,28,1)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation=activation))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64,activation=activation))
    model.add(layers.Dense(10,activation='softmax'))
    return model


def test_data(cnn_model,test_images,test_labels):
    '''
    Function to test the built model on the testing dataset. Returns the predictions.
    '''
    loss,acc=cnn_model.evaluate(test_images,test_labels)
    print('Loss= {}'.format(loss))
    print('Accuracy= {}'.format(acc))
    
    predictions=cnn_model.predict(test_images)
    return predictions

def one_hot_encoding(predicted_label):
    '''
    This function performs one hot encoding for the predictions. 
    '''
    encoded=[]
    for value in predicted_label:
        label=[0 for _ in range(10)]
        label[value]=1
        encoded.append(label)
    
    return encoded

def run_final_arch(cnn_model,train_images,train_labels,test_images,test_labels,alpha,batch_size,no_of_epochs):
    '''
    This function runs the cnn model that has been finalised after hyper-parameter tuning. 
    Returns the predicted labels after performing one-hot encoding.
    '''
    cnn_model.compile(optimizer=optimizers.RMSprop(lr=alpha),loss='categorical_crossentropy',metrics=['accuracy'])
    cnn_model.fit(train_images,train_labels,epochs=no_of_epochs,batch_size=batch_size)
    
    predictions=test_data(cnn_model,test_images,test_labels)
    predicted_label=np.argmax(predictions,axis=-1)
    encoded=one_hot_encoding(predicted_label)
    
    return encoded


def plot_accuracy_curve(no_of_epochs,train_acc,val_acc,acc_title):
    '''
    Function to plot the accuracy curve during the cross validation phase.
    '''
    fig1=plt.figure()
    epoch=range(1,no_of_epochs+1)
    plt.plot(epoch, train_acc, 'r', label='Training acc')
    plt.plot(epoch, val_acc, 'b', label='Validation acc')
    plt.title(acc_title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy.png')
    
    
def plot_loss_curve(no_of_epochs,train_loss,val_loss,loss_title):
    '''
    Function to plot the loss curve during the cross validation phase.
    '''
    fig2=plt.figure()
    epoch=range(1,no_of_epochs+1)
    plt.plot(epoch,train_loss,'r',label='Training Loss')
    plt.plot(epoch,val_loss,'b',label='Validation Loss')
    plt.title(loss_title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')


def cv_plots(model_history,no_of_epochs,image_directory):
    '''
    This function creates a folder in the home directory and saves all the created accuracy and loss plots in that directory.
    '''
    for key,value in model_history.items():
        os.mkdir(image_directory+'\\'+ key)
        os.chdir(image_directory+'\\'+ key)
        a,b=key.split(sep='_')
        
        train_loss=model_history[key]['loss']
        val_loss=model_history[key]['val_loss']
        loss_title='Loss Curve(alpha={},batch size={})'.format(a,b)
        plot_loss_curve(no_of_epochs,train_loss,val_loss,loss_title)
    
        train_acc = model_history[key]['acc']
        val_acc = model_history[key]['val_acc']
        acc_title='Accuracy Curve(alpha={},batch size={})'.format(a,b)
        plot_accuracy_curve(no_of_epochs,train_acc,val_acc,acc_title)
    os.chdir(image_directory)


def custom_gridsearch(train_images,train_labels,alpha,batch_size,no_of_epochs,activation):
    '''
    This function implements gridsearch for the cross-validation process. It returns a dictionary containing the 
    loss and accuracy results of the grid search. 
    This function can also be used for the babysitting approach for hyper parameter tuning. 
    '''
    hist_of_models={}   
    for a in alpha:
        for b in batch_size:
            a_model=model_arch(activation)
            print('alpha= {}, batch_size={}'.format(a,b))
            a_model.compile(optimizer=optimizers.RMSprop(lr=a),loss='categorical_crossentropy',metrics=['accuracy'])
            history=a_model.fit(train_images,train_labels,epochs=no_of_epochs,batch_size=b,validation_split=0.20)
            historydict=history.history
            
            key=str(a)+'_'+str(b)
            hist_of_models[key]=historydict
        
    return hist_of_models

def main():
    
    image_directory='C:\\Users\\evkirpal\\Desktop\\nn_proj3_test2'
    modes=['run_final_architecture','custom_gridsearch']
    
    #Parameters to tweak
    #Uncomment the below two lines when performing grid search. Also, set mode=1. 
#    batch_size=[64,128,256,512]
#    alpha=[0.01,0.001,0.0001]
    mode=0              #Keep either 0 or 1
    alpha=0.001         #Comment out this line when performing grid search
    batch_size=64       #Comment out this line when performing grid search
    activations='relu'
    no_of_epochs=10
    running_mode=modes[mode]
    (train_images,train_labels,test_images,test_labels)=load_dataset()
    if running_mode=='run_final_architecture':
        final_model=model_arch(activations)
        encoded=run_final_arch(final_model,train_images,train_labels,test_images,test_labels,alpha,batch_size,no_of_epochs)
        
        df=pd.DataFrame(encoded)
        df.to_csv('mnist.csv', header=False, index=False)
    
    if running_mode=='custom_gridsearch':
        model_history=custom_gridsearch(train_images,train_labels,alpha,batch_size,no_of_epochs,activations) 
        cv_plots(model_history,no_of_epochs,image_directory)



if __name__=='__main__':
    main()
