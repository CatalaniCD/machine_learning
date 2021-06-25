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
                      cluster_std = 0.5)
    
    y = pd.Series(y).replace(0, -1).values

    # visualize data
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.show()

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        train_size = 0.5, 
                                                        random_state=0)
  
    # perceptron
    
    # initalize Weight Vectors to 0, same size as features
    # W = np.zeros(2)
    
    # or
    
    # initialize Weight Vectors to random, same size as features
    # W = np.random.normal(loc = 0.0, scale = 1.0, size = len(X_train[0]))
    W = np.random.uniform(low = 0.0, high = 1.0, size = len(X_train[0]))
    
    # initalize b to 0
    b = 0
    T = 100
    
    # max iteration for convergence 
    for t in range(T):
        
        # iterate the data loop
        L = len(X_train)
        for i in range(L):
            
            # select the data point and label
            X_, y_ = np.array([X_train[i][0], X_train[i][1]]), y_train[i]
            
            # evaluate the decision boundary
            # print((np.dot(W.T, X_) + b) /  np.linalg.norm(W))
            if ( y_ * (np.dot(W.T, X_) + b) /  np.linalg.norm(W)) <= 0:
                
                # update decision boundary
                W = W + (X_ * y_) * 0.01 # learning rate
                b = b + y_
        
    error = 0
    L = len(X_test)
    for i in range(L):
        
        # select the data point and label
        X_, y_ = np.array([X_test[i][0], X_test[i][1]]), y_test[i]
        
        # evaluate the decision boundary
        if ( y_ * (np.dot(W.T, X_) + b) /  np.linalg.norm(W)) <= 0:
            
            error += 1
    

    print('Accuracy Score : ', 1 - (error/ L))

              
    # calc and plot decision boundary
    min_x1, max_x1 = X_test[:, 0].min(), X_test[:, 0].max()
    
    n = 100 
    x1 = [x for x in np.linspace(start = min_x1, stop = max_x1, num = n)] 
    
    # calc decision boundary  
    slope = -W[0] / W[1]
    intercept = -b / W[1]
    decision_boundary = [slope * x + intercept for x in x1]
    
    # visualize decision boundary 
    plt.plot(x1, decision_boundary, c = 'black', linestyle = 'dashed')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.coolwarm)
    plt.show()
   