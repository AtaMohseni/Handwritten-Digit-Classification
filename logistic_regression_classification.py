# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 13:57:24 2023

@author: ATA
"""


from data_loader import data_loader
import random
import numpy as np
datatrain , datatest = data_loader()
random.shuffle(datatrain)
random.shuffle(datatest)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def log_likelihood(data,w):
    """ function to calculate log-likelihood of entire data using current
    weight vectore w """
    
    Lw= 0
    for index, datapoint in enumerate(data) :
        Lw += datapoint[1]*np.log(sigmoid(np.dot(np.array(datapoint[0]),w))) + (1 - datapoint[1])*np.log(sigmoid(np.dot(np.array(datapoint[0]),-w)))
    return Lw

def gradient(data, w):
    """ function to calculate the gradient vector of log likelihood of data
    using the current weight vector w"""
    
    grad = np.zeros(len(w))
    for index, datapoint in enumerate(data) :
        grad += (datapoint[1] - sigmoid(np.dot(np.array(datapoint[0]),w)))*np.array(datapoint[0])
    return grad  
  
def train(data):
    
    w = np.zeros(len(data[0][0])) #initialize w
    learning_rate = 0.001
    log_likelihood_vec = [log_likelihood(data, w)]
    
    for epoch in range(100):
        
        w = w + learning_rate*(gradient(data, w))
        log_likelihood_vec.append(log_likelihood(data, w))
        
    return w , log_likelihood_vec
        
w , log_likelihood_vec  = train(datatrain)
    
