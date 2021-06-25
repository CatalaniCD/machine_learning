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


# =============================================================================
# Neuron Class
# =============================================================================

class Neuron():
       
    """ Neuron Classifier Model """
    
    def __init__(self):
        self.weights = np.array([])
        self.bias = 0
        self.learning_rate = 1e-03
        self.epsilon = 1e-03
        
    def scalar_product(self, X):
        """ Dot Product / Matmul """
        return np.dot(self.weights.T, X) + self.bias
    
    def sigmoid(self, z):
        """ Activation Function """
        # sigmoid function domain {-1, 1}, S shaped function
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
      """ Activation Derivative """
      return self.sigmoid(z) * (1 - self.sigmoid(z))
  
    def loss_function(self, y, y_hat):
        """ Loss Function : Squared Loss """
        return 0.5 * (y - y_hat) ** 2
    
    def loss_derivative(self, y, y_hat):
        """ Loss derivative """
        return - (y - y_hat)
          
    def training(self, X, y, T = 25):
        """ Model Training """
        # initialize random weights and bias
        self.weights = np.random.normal(loc = 0.0, scale = 5.0, size = X.shape[1])
        self.bias = np.random.randint(1)      
        L = X.shape[0]
        update = 0
        for t in range(T):
            
            for i in range(L):
                
                X_ =  X[i, :]
                y_ = y[i]
                
                # neuron activation            
                z = self.scalar_product(X_)
                activation = self.sigmoid(z)
                
                # loss function
                loss = self.loss_function(y_, activation)
    
                # print(loss)
    
                # check for loss magnitude
                if loss > self.epsilon:
                    
                    """
                    # derivatives
                    loss_derivative
                    activation_derivative
                    
                    # updates                                        
                    weights_update = loss_derivative * activation_derivative * activation
                    bias_update = loss_derivative *  activation_derivative
                    
                    # new wights, bias
                    weigths = weights - ( learning_rate * weights_update )
                    bias = bias - (learning_rate * bias_update.sum( axis = 0 ) )
                    """
                    
                    # derivatives
                    loss_derivative = self.loss_derivative(y_, activation)
                    activation_derivative = self.sigmoid_derivative(z)
                    
                    # updates                                        
                    weights_update = loss_derivative * activation_derivative * activation
                    bias_update = ( loss_derivative * activation_derivative * 1 ).sum( axis = 0 )
                    
                    # new wights / bias
                    self.weigths = self.weights - ( self.learning_rate * weights_update )
                    self.bias = self.bias - (self.learning_rate * bias_update )
                    
                    update += 1
                    
        print(f'""" Training Complete with {update} updates """')
        pass           
                        
                
    def predict(self, X_):
        """ Model Prediction """            
        # neuron activation                
        activation = self.sigmoid(self.scalar_product(X_))
        # activation < epsilon for classification
        return (activation < self.epsilon).astype(int)
    
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
        print(f'"""Model Baseline : { len([x for x in (y == 1) if x == True]) / len(y) }"""')
        print(f'"""Model Accuracy : {accuracy}"""')
        pass

# =============================================================================
# program test
# =============================================================================

if __name__ == '__main__':
    
    X, y = make_blobs(n_samples = 500, 
                      centers = 2, 
                      random_state = 0, 
                      cluster_std = 1.0)
    
    
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        train_size = 0.5, 
                                                        random_state=0)


    # visualize train set
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm)
    plt.show()

    neuron = Neuron()
    neuron.training(X_train, y_train)
    neuron.accuracy(X_test, y_test)
    
    plt.scatter(X_test[:, 0], X_test[:, 1], c=[neuron.predict(x) for x in X_test], cmap=plt.cm.coolwarm)
    plt.show()
