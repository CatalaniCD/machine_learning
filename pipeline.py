#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 21:52:43 2021

@author: q

Goal : Create a pipeline

"""
# =============================================================================
# imports
# =============================================================================

# standard scaler
from sklearn.preprocessing import StandardScaler

# dimensionality reduction
from sklearn.decomposition import PCA

# support vector classifier
from sklearn.svm import SVC

# pipeline
from sklearn.pipeline import Pipeline

# dataset generator
from sklearn.datasets import make_blobs

# train test split
from sklearn.model_selection import train_test_split

# data hadling
import pandas as pd
import numpy as np

# data visualization
import matplotlib.pyplot as plt

# =============================================================================
# program test
# =============================================================================

if __name__ == '__main__':
    
    X, y = make_blobs(n_samples=500, 
                      centers=2, 
                      random_state=0, 
                      cluster_std=1.2)
    
    # visualize data
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.show()

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    """ Pipeline """
    steps = (('scaler', StandardScaler()), ('dim_reduction', PCA()), ('estimator', SVC()))
    
    # instantiate pipeline   
    pipeline = Pipeline(steps, verbose = True)
    
    # fit pipeline
    pipeline.fit(X_train, y_train)
    
    # predict
    y_preds = pipeline.predict(X_test)    

    # score
    score = pipeline.score(X_test, y_test) 
    
    print(f'\nPipeline score : {score}\n')
