#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 17:17:45 2018

@author: MyReservoir
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/Users/MyReservoir/Library/Mobile Documents/com~apple~CloudDocs/My Program/NCSU ECE Department Docs/Semester 3/ECE 542 Neural Networks/Programming Asssignments/project 1/')
import src.network2 as network2
import src.mnist_loader as loader
import src.activation as act

DATA_PATH = '/Users/MyReservoir/Downloads/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action='store_true',
                        help='Check data loading.')
    parser.add_argument('--sigmoid', action='store_true',
                        help='Check implementation of sigmoid.')
    parser.add_argument('--gradient', action='store_true',
                        help='Gradient check')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--test', action='store_true',
                        help='Test the model')


    return parser.parse_args()

def load_data():
    train_data, valid_data, test_data = loader.load_data_wrapper(DATA_PATH)
    print('Number of training: {}'.format(len(train_data[0])))
    print('Number of validation: {}'.format(len(valid_data[0])))
    print('Number of testing: {}'.format(len(test_data[0])))
    return train_data, valid_data, test_data

def test_sigmoid():
    z = np.arange(-10, 10, 0.1)
    y = act.sigmoid(z)
    y_p = act.sigmoid_prime(z)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(z, y)
    plt.title('sigmoid')

    plt.subplot(1, 2, 2)
    plt.plot(z, y_p)
    plt.title('derivative sigmoid')
    plt.show()

def gradient_check():
    train_data, valid_data, test_data = load_data()
    model = network2.Network([784, 30, 10])
    model.gradient_check(training_data=train_data, layer_id=1, unit_id=5, weight_id=3)

def main():
    # load train_data, valid_data, test_data
    train_data, valid_data, test_data = load_data()
    # construct the network
    model = network2.Network([784, 110, 10])
    # train the network using SGD
    model.SGD(
        training_data=train_data,
        epochs=100,
        mini_batch_size=128,
        eta=1e-3,
        lmbda = 0.0,
        evaluation_data=valid_data,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)

    model.save("try1")

def test():
    # load train_data, valid_data, test_data
    train_data, valid_data, test_data = load_data()
    n_data = len(test_data[0])
    
    # construct the network
    model = network2.load("try1")
    test_accuracy = model.accuracy(test_data)
    test_accuracy = 100*test_accuracy/ len(test_data[0])
    print("Testing accuracy : {} %".format(
        test_accuracy, n_data))

if __name__ == '__main__':
    FLAGS = get_args()
    if FLAGS.input:
        load_data()
    if FLAGS.sigmoid:
        test_sigmoid()
    if FLAGS.train:
        main()
    if FLAGS.gradient:
        gradient_check()
    if FLAGS.test:
        test()