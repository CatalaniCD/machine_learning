#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 12:23:58 2021

@author: q

GOAL : Develop a Linear Regression Algorithm with Numpy

/// Concept

    Linear Equation : 
        
        :: y = a + b * x
        
    From Algebra : Linear Transformation
        
        :: A linear transformation, X data, b result
    
        :: Ax = b , the solution x = bA_inverse
        
/// A Linear Regression may have a Closed Form Solution 
        or an Iterative Solution

/// Backpropagating the Loss

    > Linear Estimation
    
        y = w*x + b | z = W*x + b

    > Squared Loss

        L = (true - pred) ** 2
    
    > Chain Rule
    
      W :  dL/dw = dL/dz * dz/dw
        
      b : dL/db = dL/dz * dz/db
      
      dL/dz = (u)**2 * du/dz = 2 * (true - pred) * 1
      
      dz/dw = x
      
      dz/db = 1
      
      W : dL/dw = 2 * (y_pred - y_true) * 1 * x
      
      b : dL/db = 2 * (y_pred - y_true) * 1 * 1
      
    > Gradient Updates

      W = W - alpha * (dLdz * dzdw)
      
      b = b - alpha * (dLdz * dzdb)

"""

# =============================================================================
# imports
# =============================================================================

# create random dataset
from sklearn.datasets import make_regression

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# scientific computation
import numpy as np

# =============================================================================
# funcions
# =============================================================================

def plot_2d_scatter(data: tuple):
    if data[0].shape[1] == 1:
       plt.scatter(data[0], data[1])
       plt.axhline(y = 0.0, c = 'k', ls = '--')
       plt.axvline(x = 0.0, c = 'k', ls = '--')
       plt.title('Linear Regression DataSet')
       plt.show()
    else:
        raise ValueError('More than 1 Feature in the dataset')

def plot_linear_model(X, y, W, b):
    if X.shape[1] == 1:
       x_values = np.linspace(X.min(), X.max(), 100)
       y_values = [linear_estimate(x, W, b) for x in x_values]
       
       plt.scatter(X, y)
       plt.axhline(y = 0.0, c = 'k', ls = '--')
       plt.axvline(x = 0.0, c = 'k', ls = '--')
       
       plt.title('Fitting Linear Regression')
       plt.plot(x_values, y_values, c = 'r')
       plt.show()
    else:
       raise ValueError('More than 1 Feature in the dataset')

# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    
    
    dataset = make_regression(n_samples = 100, 
                              n_features=1, 
                              n_targets=1,
                              noise= 30.0,
                              tail_strength = 1.0,
                              )
    try:
        plot_2d_scatter(data = dataset)
    except Exception as e:
        print(e)
    
    # create an iterative algorithm to fix this... 
    X, y = dataset
     
    y = y + 50
    
    # define weights
    alpha = 0.001
    # W = np.random.normal(size = X.shape[1])
    # W = np.ones(shape = X.shape[1])
    W = np.zeros(shape = X.shape[1])
    b = 0
    
    # loss function
    def sqloss(true, pred):
        return 0.5 * ((pred - true) ** 2)
    
    # partial derivatives
    
    def partial_derivatives(true, pred, x):
        return {'dLdz' : (pred - true) * 1,
                'dzdw' : x,
                'dzdb' : 1}
    
    # linear estimate
    def linear_estimate(X, W, b):
        return np.dot(W.T, X) + b
    
    def batch_loss(X, y, W, b):
        loss = 0
        for i in range(X.shape[0]): 
            x = X[i, :]
            y_ = y[i]
            y_pred = linear_estimate(x, W, b)
            loss += sqloss(true = y_, pred = y_pred)
        return loss / X.shape[0]
  
    for j in range(100):
        
        for i in range(X.shape[0]):
    
            x = X[i, :]
            y_ = y[i]
    
            y_pred = linear_estimate(x, W, b)
        
            pder = partial_derivatives(true = y_, pred = y_pred, x = x)
            
            W = W - alpha * (pder['dLdz'] * pder['dzdw'])
            b = b - alpha * (pder['dLdz'] * pder['dzdb'])
        
            print(batch_loss(X, y, W, b), W, b)

    plot_linear_model(X, y, W, b)   
    
    print(batch_loss(X, y, W, b), W, b)

   