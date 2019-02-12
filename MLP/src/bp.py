#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 20:27:11 2018

@author: MyReservoir
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: bp.py

import numpy as np
from src.activation import sigmoid, sigmoid_prime

def backprop(x, y, biases, weights, cost, num_layers):
    """ function of backpropagation
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient of all biases and weights.
        Args:
            x, y: input image x and label y
            biases, weights (list): list of biases and weights of entire network
            cost (CrossEntropyCost): object of cost computation
            num_layers (int): number of layers of the network
        Returns:
            (nabla_b, nabla_w): tuple containing the gradient for all the biases
                and weights. nabla_b and nabla_w should be the same shape as 
                input biases and weights
    """
    # initial zero list for store gradient of biases and weights
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_w = [np.zeros(w.shape) for w in weights]
    activations = [np.zeros(w.shape) for w in biases]

    ### Implement here
    # feedforward
    # Here you need to store all the activations of all the units
    # by feedforward pass
    ###
    activations.insert(0,x)
    for layer in range(1,num_layers):
        activations[layer]=sigmoid(np.dot(weights[layer-1],activations[layer-1])+biases[layer-1])

    # compute the gradient of error respect to output
    # activations[-1] is the list of activations of the output layer
    delta = (cost).delta(activations[-1], y) 
    
    ### Implement here
    # backward pass
    # Here you need to implement the backward pass to compute the
    # gradient for each weight and bias
    ###
    for k in range(num_layers,1,-1):
        delta=delta*sigmoid_prime(activations[k-1])
        nabla_b[k-2]=delta
        nabla_w[k-2]=np.dot(delta,activations[k-2].T)
        delta=np.dot(weights[k-2].T,delta)
    

    return (nabla_b, nabla_w)