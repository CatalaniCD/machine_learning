#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 15:18:34 2021

@author: q

Goal : Create a Function Approximation with a Neuron

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor


def function_plus_noise(n = 100, f = np.sqrt):
    """ Apply a Function plus Gaussian Noise """
    values = np.linspace(0, 10, 100)
    return pd.Series([f(x) * (1+np.random.normal(loc = 0.0, scale = 0.1)) for x in values], name = str(f)) 

def plot(series):
    plt.scatter(series.index, series.values)
    plt.title(series.name)
    plt.grid()
    plt.show()   
    pass

def plot_model(series, model):
    y_pred = pd.Series(model.predict(np.array(series.index).reshape(-1, 1)), index = series.index)
    plt.scatter(series.index, series.values)
    plt.plot(y_pred.index, y_pred.values, color = 'red', ls = 'dashed')
    plt.title(series.name)
    plt.grid()
    plt.show()   
    pass

def function_mapping():
    """ Visual Evaluation for Indicators Mapping """
    values = np.linspace(0, 10, 1000)
    plt.plot(values, values, label = 'linear mapping')
    plt.plot(values, pd.Series(values).apply(np.log), label = 'ln mapping')
    plt.plot(values, pd.Series(values).apply(np.sqrt), label = 'sqrt mapping') 
    plt.legend()
    plt.grid()
    plt.show()
    pass

if __name__ == '__main__':
    
    function_mapping()

    n = 100    

    values = function_plus_noise(n = n, f = np.sqrt)
    
    plot(values)

    S = values.iloc[sorted(np.random.randint(n, size = n//2))]
    S.name = 'Sample : ' + values.name
        
    plot(S)

    model = MLPRegressor(hidden_layer_sizes = (15, 15), activation = 'tanh',
                         solver = 'adam', max_iter= 5000)
    
    model.fit(np.array(S.index).reshape(-1, 1), S.values.reshape(-1, 1))
    
    plot_model(values, model)
    


    values = function_plus_noise(n = 100, f = np.sin)
    
    plt.scatter(values.index, values)
    plt.title(values.name)
    plt.grid()
    plt.show()   

    S = values.iloc[sorted(np.random.randint(n, size = n//2))]
    S.name = 'Sample' + values.name    
    
    plot(S)
    
    model = MLPRegressor(hidden_layer_sizes = (125, 125, 125, 125, 125), 
                         activation = 'tanh', solver = 'adam', max_iter= 1000,
                         verbose = True, tol = 1e-4, validation_fraction= 0.3,
                         n_iter_no_change = 100)
    
    model.fit(np.array(S.index).reshape(-1, 1), S.values.reshape(-1, 1))
    
    plot_model(values, model)


    """
    
    Log Returns Approximation
    
    Approximate S Shaped function using a Neural Network and Input Data
    
    - n_log_returns
    - mean log_returns
    - diff_mean_log_returns
    - std_log_returns
    - diff_std_log_returns

    """
    n = 300
    
    lnr = pd.Series([np.random.normal() for x in range(n)], name = 'ln_returns')
    
    plt.plot(lnr.cumsum())
    plt.title(lnr.name)
    plt.grid()
    plt.show()   
   
    plt.scatter(lnr.index, lnr.cumsum(), marker = '.')
    plt.title(lnr.name)
    plt.grid()
    plt.show()   
    
    plt.scatter(lnr.index, sorted(lnr.values))
    plt.title(lnr.name + ' Sorted')
    plt.grid()
    plt.show()    

    S = lnr.iloc[sorted(np.random.randint(n, size = n//2))]
    S.name = 'Sample' + lnr.name    
    
    plt.scatter(S.index, sorted(S.values))
    plt.title(S.name)
    plt.grid()
    plt.show() 
    
    model = MLPRegressor(hidden_layer_sizes = (128 for i in range(10)), 
                         activation = 'tanh', solver = 'adam', max_iter= 1000,
                         verbose = True, tol = 1e-6, validation_fraction= 0.5,
                         n_iter_no_change = 100)
    
    plt.plot(np.log(model.loss_curve_))
    plt.title('Loss Curve')
    plt.grid()
    plt.show()
    
    model.fit(np.array(S.index).reshape(-1, 1), np.array(sorted(S)).reshape(-1, 1))
    
    y_pred = pd.Series(model.predict(np.array(lnr.index).reshape(-1, 1)), index = lnr.index)
    plt.scatter(lnr.index, sorted(lnr.values))
    plt.plot(y_pred.index, y_pred.values, color = 'red', ls = 'dashed')
    plt.title(lnr.name)
    plt.grid()
    plt.show()  


