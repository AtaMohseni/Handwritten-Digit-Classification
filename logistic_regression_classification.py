# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 13:57:24 2023

@author: ATA
"""


from data_loader import data_loader
import random
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def log_likelihood(data,w):
    """ function to calculate log-likelihood of entire data using current
    weight vectore w, and return the log-likelihood value """
    
    Lw= 0
    for index, datapoint in enumerate(data) :
        Lw += datapoint[1]*np.log(sigmoid(np.dot(np.array(datapoint[0]),w))) + (1 - datapoint[1])*np.log(sigmoid(np.dot(np.array(datapoint[0]),-w)))
    return Lw

def gradient(data, w):
    """ function to calculate the gradient vector of log likelihood of data
    using the current weight vector w, and return the gradient vector"""
    
    grad = np.zeros(len(w))
    for index, datapoint in enumerate(data) :
        grad += (datapoint[1] - sigmoid(np.dot(np.array(datapoint[0]),w)))*np.array(datapoint[0])
    return grad  
  
def train(data):
    """ training data to find optimum weight vector w to maximize
    log-likelihood using gradient ascent"""
    
    w = np.zeros(len(data[0][0])) #initialize w
    learning_rate = 0.0001
    log_likelihood_vec = [log_likelihood(data, w)]
    
    for epoch in range(200):
        
        w = w + learning_rate*(gradient(data, w))
        log_likelihood_vec.append(log_likelihood(data, w))
        
    return w , log_likelihood_vec

def error(data,w):
    """ function that predict classes and calculate  and return 
    misclassification percentage """ 
    preds = []
    correct = 0
    for index, datapoint in enumerate (data):
        predict = np.rint(sigmoid(np.dot(np.array(datapoint[0]),w)))
        preds.append(predict)
        if predict == datapoint[1]:
            correct += 1
    percent_error = (1 - correct/len(data))*100
    return percent_error    
   
if __name__ == "__main__" :   
    
    datatrain , datatest = data_loader()
    random.shuffle(datatrain)
    random.shuffle(datatest)     
    w , log_likelihood_vec  = train(datatrain)
    train_error = error(datatrain,w)
    test_error = error(datatest,w)
    plt.plot(list(range(len(log_likelihood_vec))),log_likelihood_vec)
    plt.xlabel('iteration')
    plt.ylabel('log likelihood')
    #plt.axis([0, 100, -1000, 0])
    #plt.show()