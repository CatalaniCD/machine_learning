#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 21:32:32 2021

@author: q

Goal : Create a series of scalers

"""
# =============================================================================
# imports
# =============================================================================

# data handling
import numpy as np
import pandas as pd

# data visualization
import matplotlib.pyplot as plt

# =============================================================================
# functions
# =============================================================================

def gaussian_series(n = 500):
    """ Create Synthetic TimeSeries"""
    data = []
    for i in range(n):
        data += [np.random.normal(loc = 0.0, scale = 1.0, size = None)]
    return pd.Series(data).cumsum()


def standard_scaler(series):
    """ Standard Scaler """
    assert isinstance(series, pd.Series), ' Input a pd.Series'
    u = series.mean()
    std = series.std()
    return (series - u) / std

def min_max_scaler(series):
    """ MinMax Scaler """
    assert isinstance(series, pd.Series), ' Input a pd.Series'
    mn = series.min()
    mx = series.max()
    rg = mx - mn
    return (series - mn) / rg


if __name__ == '__main__':
    
    series = gaussian_series()
    
    # plot series
    series.plot()
    plt.show()
                 
    # transform and plot
    standard_scaled_series = standard_scaler(series)
    standard_scaled_series.plot()    
    plt.show()
    
    # transform and plot
    min_max_scaled_series = min_max_scaler(series)
    min_max_scaled_series.plot()    
    plt.show()
    
    
    