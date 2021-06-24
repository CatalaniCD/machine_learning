#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 21:52:43 2021

@author: q

Goal : Create a perceptron algorithm

"""
# =============================================================================
# imports
# =============================================================================

# dataset generator
from sklearn.datasets import make_blobs

# train test split
from sklearn.model_selection import train_test_split

# data hadling
import pandas as pd
import numpy as np

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# program test
# =============================================================================

if __name__ == '__main__':
    
    X, y = make_blobs(n_samples = 500, 
                      centers = 2, 
                      random_state = 0, 
                      cluster_std = 1.0)
    
    # visualize data
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.show()

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        train_size = 0.5, 
                                                        random_state=0)
  
    # perceptron
    
    # initalize Weight Vectors to 0, same size as features
    W = np.zeros(2)
    
    # or
    
    # initialize Weight Vectors to random, same size as features
    W = np.random.normal(loc = 0.0, scale = 1.0, size = len(X_train[0]))
    # W = np.random.uniform(low = 0.0, high = 1.0, size = len(X_train[0]))
    
    # initalize b to 0
    b = 0
    T = 1
    
    # max iteration for convergence 
    for t in range(T):
        
        # iterate the data loop
        L = len(X_train)
        """Stochastic Gradient Descent"""
        rows = [x for x in range(L)]
        stochastic = [np.random.choice(rows, size = None, replace = True) for x in range(L)]
        for i in stochastic:
            
            # select the data point and label
            X_, y_ = np.array([X_train[i][0], X_train[i][1]]), y_train[i]
            
            # evaluate the decision boundary
            if (np.dot(W.T, X_) + b) <= 0:
                
                # update decision boundary
                W = W + (X_ * y_)
                b = b + y_
                
    # calc and plot decision boundary
    max_x1, min_x1 = np.max(X_train[:, 0]), np.min(X_train[:, 0])
    max_x2, min_x2 = np.max(X_train[:, 1]), np.min(X_train[:, 1])
    
    n = 5 
    x1_values = [x for x in np.linspace(start = min_x1, stop = max_x1, num = n)]
    x2_values = [x for x in np.linspace(start = min_x2, stop = max_x2, num = n)]
    
    # calc decision boundary
    decision_boundary = [np.dot(W.T, x) + b for x in zip(x1_values, x2_values)]
    
    # visualize decision boundary 
    plt.plot(x1_values, decision_boundary, c = 'black', linestyle = 'dashed')
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm)
    plt.show()
        
    error = 0
    L = len(X_test)
    for i in range(L):
        
        # select the data point and label
        X_, y_ = np.array([X_test[i][0], X_test[i][1]]), y_test[i]
        
        # evaluate the decision boundary
        if (np.dot(W.T, X_) + b) <= 0:
            
            error += 1

    print('Accuracy Score : ', 1 - (error/ L))
