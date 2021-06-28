#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 21:52:43 2021

@author: q

Goal : Create a support vector machine algorithm algorithm


SVM Algorithm :
    
    - Scale the data using a Standard Scaler
    
    - Objective function : 
        
        L(w) = SUM max(0, y_ * (dot(W.T, X_) + b) - 1 ) + lambda * norm(W) ** 2
        
        Derivatives : 
            
            dD/dW         = y_ * X_
            dlambda / dW  = - 2 * lambda * W 
            
    - Updates :
        
        if y_ * (dot(W.T, X_) + b) <= 1:
            
            W = W + learning_rate * (( y_ * X_ ) - (2 * lambda * W))
        
        else:
             W = W + learning_rate * (- 2 * lambda * W )
        
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
                      cluster_std = 0.33)
    
    y = pd.Series(y).replace(0, -1).values

    # visualize data
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.show()

    # standard scaling
    X = pd.DataFrame(X)
    X = (X - X.mean()) / (X.std())
    X = X.values

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        train_size = 0.5, 
                                                        random_state=0)
  
    # support vector machine
       
    # initialize Weight Vectors to random, same size as features
    W = np.random.uniform(low = -1.0, high = 1.0, size = len(X_train[0]))
    
    # initalize b to 0
    b = 0
    T = 500
    
    C = 5
    reg_lambda = 1 / C
    learning_rate = 0.01
    
    # max iteration for convergence 
    for t in range(1, T):
        
        # iterate the data loop
        L = len(X_train)
        loss = []
        for i in range(L):
            
            # select the data point and label
            X_, y_ = np.array([X_train[i][0], X_train[i][1]]), y_train[i]
            
            # store distance values
            loss  += [( y_ * (np.dot(W.T, X_) + b))]
            
        
        # evaluate the decision boundary
        support_distance = np.mean(loss) + reg_lambda * np.linalg.norm(W) ** 2
            
        if support_distance <= 1:
            
            # update decision boundary
            W = W + learning_rate * ( ( y_ * X_) - (2 * reg_lambda * W ))
        
        else:
             
            # update decision boundary
            W = W + learning_rate * (- 2 * reg_lambda * W )
        
        
    error = 0
    L = len(X_test)
    for i in range(L):
        
        # select the data point and label
        X_, y_ = np.array([X_test[i][0], X_test[i][1]]), y_test[i]
        
        # evaluate the decision boundary
        if ( y_ * (np.dot(W.T, X_) + b) ) <= 1:
            
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
   
