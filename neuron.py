#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 08:58:58 2021

@author: q

Goal : develop a neuron class

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

import time


# =============================================================================
# Neuron Class
# =============================================================================

class Neuron():
       
    """ Neuron Classifier Model """
    
    def __init__(self):
        self.weights = np.random.uniform(low = -1, high = 1, size = X.shape[1]) 
        self.bias = np.random.uniform(low = -1, high = 1, size = None)   
        self.learning_rate = 0.1
        self.epsilon = 0.1
    
    def loss_function(self, y, y_hat):
        """ Loss Function : Squared Loss """
        return 0.5 * (y - y_hat) ** 2
    
    def loss_derivative(self, y, y_hat):
        return 0.5 * 2 *  (y - y_hat) * (-1)
    
    def sigmoid(self, z):
        """ Activation Function """
        # sigmoid function domain {0, 1}, S shaped function
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        """ Activation Derivative """
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def forward_prop(self, X_):
        self.z = self.weights @ X_ + self.bias
        return self.sigmoid(self.z)
   
    def back_prop(self, X_, y_, activation):
        # derivatives
        loss_derivative = self.loss_derivative(y_, activation)
        activation_derivative = self.sigmoid_derivative(self.z)
        # updates                                        
        weights_update =  loss_derivative * activation_derivative * X_
        bias_update = ( loss_derivative * activation_derivative * 1 ).sum( axis = 0 , keepdims = True)
        return weights_update, bias_update
        
          
    def training(self, X, y, T = 100, VISUALIZE = False):
        """ Model Training """
        # initialize random weights and bias
        L = X.shape[0]
        update = 0
        for t in range(T):
            
            for i in range(L):
                
                X_ =  X[i, :]
                y_ = y[i]
                
                # neuron activation            
                activation = self.forward_prop(X_)
                
                # loss function
                loss = self.loss_function(y_, activation)
              
                # check for loss magnitude
                if loss > self.epsilon:
                    
                    """
                    
                    # derivatives
                    loss_derivative
                    activation_derivative
                    
                    # updates                                        
                    weights_update = loss_derivative * activation_derivative * X_
                    bias_update = loss_derivative *  activation_derivative
                    
                    # new wights, bias
                    weigths = weights - ( learning_rate * weights_update )
                    bias = bias - (learning_rate * bias_update.sum( axis = 0 ) )
                    
                
                    """
    
                    weights_update, bias_update = self.back_prop(X_, y_, activation)
                    
                    # new wights / bias
                    self.weights = self.weights - ( self.learning_rate * weights_update )
                    self.bias = self.bias - (self.learning_rate * bias_update )
                    
                    update += 1
                    
                    W = self.weights
                    b = self.bias
                    
                    # print(weights_update, W, b, t)
                    if VISUALIZE:
                    
                         # calc and plot decision boundary
                        min_x1, max_x1 = X_test[:, 0].min(), X_test[:, 0].max()
                        
                        n = 10
                        x1 = [x for x in np.linspace(start = min_x1, stop = max_x1, num = n)] 
                        
                        # calc decision boundary  
                        slope = -W[0] / W[1]
                        intercept = -b / W[1]
                        decision_boundary = [slope * x + intercept for x in x1]
                        
                        # visualize decision boundary 
                        plt.plot(x1, decision_boundary, c = 'black', linestyle = 'dashed')
                        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.coolwarm)
                        plt.show()
                        
                        time.sleep(0.01)
                        
                                    
        print(f'""" Training Complete with {update} updates """')
        pass           
                        
                
    def predict(self, X_):
        """ Model Prediction """            
        # neuron activation                
        activation = self.forward_prop(X_)
        return (activation > 0.5).astype(int)
    
    def accuracy(self, X, y):
        """ Model Accuracy """
        error = 0
        L = X.shape[0]
        for i in range(L):
            
            X_ =  X[i, :]
            y_ = y[i]
            
            # neuron activation / prediction                
            pred = self.predict(X_)
            
            # error, y_ != predicted
            error += (y_ != pred).astype(int)
            
        accuracy = (1 - (error / L))
        baseline = len([x for x in (y == 1) if x == True]) / len(y)
        
        print(f'"""Model Baseline : { baseline }"""')
        print(f'"""Model Accuracy : { accuracy }"""')
        score = (accuracy - baseline) / baseline
        print(f'"""Model Score    : { str(score)[:5] }"""')
        pass

# =============================================================================
# program test
# =============================================================================

if __name__ == '__main__':
    
    X, y = make_blobs(n_samples = 500, 
                      centers = 2, 
                      random_state = 0, 
                      cluster_std = 0.5)
    
    # y = pd.Series(y).replace(0, -1).values
    
    
    # standard scaling
    # X = pd.DataFrame(X)
    # X = (X - X.mean()) / (X.std())
    # X = X.values
    
    # min max scaling
    X = pd.DataFrame(X)
    rg = (X.max() - X.min())
    X = (X - X.min()) / rg
    X = X.values
    
    
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        train_size = 0.5, 
                                                        random_state=0)


    # visualize train set
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm)
    plt.show()

    neuron = Neuron()
    
    neuron.training(X_train, y_train, VISUALIZE = False)
    
    neuron.accuracy(X_test, y_test)
    
    W = neuron.weights
    b = neuron.bias
    
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
    
