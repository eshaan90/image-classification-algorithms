# Multiclass Image Classification using Keras DL framework

The goal of this project was to implement a CNN architecture using a deep learning framework. 
A convolutional neural network for MNIST classification was designed and implemented using keras with tensorflow as backend. 
Hyperparameter tuning was then performed to decide upon the optimal number of layers, learning rate, and batch size. 
Babysitting and grid search approaches were tried out and the optimal method and corresponding results are discussed in detail. 
Further, once the best set of hyperparameters were decided, different activation functions like Relu, sigmoid and tanh were applied to compare their performances. 
The optimally trained model was then used to predict on the testing dataset.

    This program performs two functionalities:
        1. Runs the final designed CNN architecture using the selected hyperparameters.
        2. Performs grid search/babysitting using a range of hyperparamters.
    Default running mode is set to 1  
    
With two alternating convolution and maxpooling layers and thereatfter two fully connected layers, we obtained an accuracy of 99.19% at 10 epochs with a batch size of 64 and learning rate of 0.001. Activation function choosen was Relu. 
