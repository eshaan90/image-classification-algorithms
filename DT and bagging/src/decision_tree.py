#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 12:34:06 2018

@author: MyReservoir
"""

import os
import struct
import numpy as np

def read(dataset , path):
    """
    This function reads the MNIST data from the path given as an argument.  
    """
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')

    with open(fname_lbl, 'rb') as flbl:
        tempNum, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        tempNum, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])
    # Creates an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)


class decisionnode:
    def __init__(self,col=-1,value=None,results=None,rightB=None,leftB=None):
        """
            self.col= column index of the criteria being tested
            self.value= the value at which splitting of dataset is done
            self.results=dict of results for branch of the nodes, 
                            None for everything except endpoints
            self.rightB= true decision nodes
            self.leftB= false decision nodes
        """
        self.col=col 
        self.value=value 
        self.results=results 
        self.rightB=rightB 
        self.leftB=leftB 



def divideset(rows, col, value):
    """ 
        This function divides the dataset 'rows' into two sets based on a condition.
        col=the column number of the attribute
        value= value at which the test condition is performed
    """
    split_function = None
    split_function = lambda row: row[col] >= value
    oneHalf = [row for row in rows if split_function (row)]  # if split_function(row)
    otherHalf = [row for row in rows if not split_function (row)]
    return (oneHalf, otherHalf)



def distinctValues(rows):
    """
        This function computes the unique counts of the possible classes
    """
    results={}
    for row in rows:
        r=row[len(row)-1]
        if r not in results: results[r]=0
        results[r]+=1
    return results



def entropy(rows):
    """
        This function computes the entropy of the different sets for each node.
    """
    
    from math import log
    logBaseTwo=lambda x:log(x)/log(2)
    results=distinctValues(rows)
    entropyValue=0.0
    for r in results.keys():
        p=float(results[r])/len(rows)
        entropyValue=entropyValue-p*logBaseTwo(p)
    return entropyValue




def buildtree(rows):
    """
        rows= entire dataset
        Tree is built in this function. 
        This function returns an object of class decisionnode 
    """
    if len(rows) == 0: return decisionnode()
    current_score = entropy(rows)

    highest_gain = 0.0
    best_attr = None
    best_sets = None

    attr_count = len(rows[0]) - 1   
    for col in range(0, attr_count):
        column_values = set([row[col] for row in rows])

        for value in column_values:
            oneHalf, otherHalf = divideset(rows, col, value)

            p = float(len(oneHalf)) / len(rows)
            gain = current_score - p*entropy(oneHalf) - (1-p)*entropy(otherHalf)
            if gain > highest_gain and len(oneHalf) > 0 and len(otherHalf) > 0:
                highest_gain = gain
                best_attr = (col, value)
                best_sets = (oneHalf, otherHalf)

    if highest_gain > 0:
        rightBranch = buildtree(best_sets[0])
        leftBranch = buildtree(best_sets[1])
        return decisionnode(col=best_attr[0], value=best_attr[1],
                rightB=rightBranch, leftB=leftBranch)
    else:
        return decisionnode(results=distinctValues(rows))

def prune(tree,mingain):
    # If the branches aren't leaves, then prune them
    if tree.rightB.results==None:
        prune(tree.rightB,mingain)
    if tree.leftB.results==None:
        prune(tree.leftB,mingain)
      
    # If both the subbranches are now leaves, see if they
      # should merged
    if tree.rightB.results!=None and tree.leftB.results!=None:
        # Build a combined dataset 
        rightB,leftB=[],[]
        for v,c in tree.rightB.results.items():
            rightB+=[[v]]*c
        for v,c in tree.leftB.results.items():
            leftB+=[[v]]*c
        # Test the reduction in entropy
        delta=entropy(rightB+leftB)-(entropy(rightB)+entropy(leftB)/2)
        if delta<mingain:
          # Merge the branches
          tree.rightB,tree.leftB=None,None
          tree.results=distinctValues(rightB+leftB)
          
          
def classify(observation,tree):
    """
        observation= one image data
        tree=the object of class decisionnode
        The observation is classified as per the tree model and the dict is returned
    """
    if tree.results!=None:

        return tree.results
    else:
        v=observation[tree.col]
        branch=None
        if v >= tree.value:
            branch = tree.rightB
        else:
            branch = tree.leftB
        return classify(observation, branch)