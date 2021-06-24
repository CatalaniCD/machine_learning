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

# support vector classifier
from sklearn.svm import SVC

# dataset generator
from sklearn.datasets import make_blobs

# train test split
from sklearn.model_selection import train_test_split, GridSearchCV

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
    
    X, y = make_blobs(n_samples=500, 
                      centers=2, 
                      random_state=0, 
                      cluster_std=1.2)
    
    # visualize data
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.show()

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # parameter space    
    gammas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    Cs = [1, 10, 100, 1e3, 1e4, 1e5]
    param_space = {'gamma': gammas, 'C': Cs}
    
    # SVC model
    svc = SVC(random_state=0)

    # grid search
    grid_search = GridSearchCV(estimator=svc, param_grid=param_space, verbose = True)
    
    # fit grid search
    grid_search.fit(X_train, y_train)
    
    # create results dataframe
    results = pd.DataFrame.from_dict(grid_search.cv_results_)
   
    # create a scores matrix
    scores_matrix = results.pivot(index='param_gamma', columns='param_C',
                                      values='mean_test_score')

    # plot param_space heatmap
    sns.heatmap(data = scores_matrix, cmap = 'coolwarm')
 