#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 12:01:13 2021

@author: q

Goal : Develop a Basic Regression & Classification Model with Keras

"""

# =============================================================================
# imports
# =============================================================================

# scientific computing
import numpy as np
# data handling
import pandas as pd

# keras library
import keras

# keras utils
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# load model 
from keras.models import load_model

# =============================================================================
# functions
# =============================================================================

def regression_model(n_cols):
    """ Define Regression Model """
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def classification_model(n_classes, n_cols):
    """ Define Classification Model """
    model = Sequential()
    model.add(Dense(n_cols, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def plot(training):
    """ Plot Training Loss - Val_Loss """
    for i in ['loss', 'val_loss']:
        pd.Series(training.history[i], index = training.epoch, name = i).plot(grid = True, legend = True)
    pass
   

if __name__ == '__main__':
    
   
    REGRESSION = False
    CLASSIFICATION = True
   
    """ Regression Model """
    if REGRESSION:
    
        """ Create a DataSet """
        n = 100
        X_train = pd.DataFrame(np.random.normal(loc = 0, scale = 1, size = (n, 3)))
        y_train = pd.Series(np.random.uniform(low = 0, high = 1.0, size = n))
    
        X_test = pd.DataFrame(np.random.normal(loc = 0, scale = 1, size = (n, 3)))
    
        """ Build a model """
        model = regression_model(X_train.shape[1])
        
        """ Fit Model """    
        training = model.fit(X_train, y_train, validation_split=0.5, epochs=100, verbose=2)
        
        """ Training Plot """
        plot(training)
        
        """ Predict """
        y_pred = pd.Series(model.predict(X_test).reshape(-1), index = X_test.index)
        
        print(y_pred[:10])

        """ Save Model """
        # model.save('regression_model.h5')

    """ Classification Model """
    if CLASSIFICATION:
        
        """ Create a DataSet """
        n = 100
        X_train = pd.DataFrame(np.random.normal(loc = 0, scale = 1, size = (n, 3)))
        y_train = pd.Series(np.random.binomial(n = 1, p = 0.5, size = n))
    
        X_test = pd.DataFrame(np.random.normal(loc = 0, scale = 1, size = (n, 3)))
        y_test = pd.Series(np.random.binomial(n = 1, p = 0.5, size = n))
    
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        """ Build a model """
        model = classification_model(n_classes = y_train.shape[1] , n_cols = X_train.shape[1])
        
        """ Fit Model """
        training = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, verbose=2)
        
        """ Training Plot """
        plot(training)
        
        training.__dict__
        
        """ Evaluate Model """
        scores = model.evaluate(X_test, y_test, verbose=0)
        
        print('Accuracy: {}% \nError: {}'.format(scores[1], 1 - scores[1]))   
        
        """ Save Model """
        # model.save('classification_model.h5')
        
        """ Load Pretrained Model """
        # pretrained_model = load_model('classification_model.h5')