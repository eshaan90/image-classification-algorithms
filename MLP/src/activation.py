#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 20:26:05 2018

@author: MyReservoir
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: activation.py

import numpy as np


def sigmoid(z):
    """The sigmoid function."""
    f=1/(1+np.exp(-z))
    return f

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    
    a=(1+np.exp(-z))*(1+np.exp(-z))
    f_prime=np.exp(-z)/a
    return f_prime
