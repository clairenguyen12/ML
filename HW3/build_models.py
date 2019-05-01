'''
Chi Nguyen
Improving Machine Learning Pipeline
This file contains functions to build
and evaluate models using different 
classification methods
The functions are inspired by Rayid Ghani's
codes from the following Github repo:
https://github.com/rayidghani/magicloops/blob/master/magicloop.py
'''

import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pylab as pl
from datetime import timedelta
from datetime import datetime
from __future__ import division
import random
from scipy import optimize
import time
import seaborn as sns
import csv
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import roc_auc_score 

from mlhelperfunctions import *


RANDOM_STATE = 100

def clf_loop(models_to_run, clfs, grid, test_size):
    """Runs the loop using models_to_run, clfs, gridm and the data
    """
    results_df =  pd.DataFrame(columns=('model_type', 'clf', 'parameters', 
    									'p_at_1', 'p_at_2', 'p_at_5', 
    									'p_at_10', 'p_at_20', 'p_at_30', 'p_at_50',
    									'recall_at_1', 'recall_at_2', 'recall_at_5',
    									'recall_at_10', 'recall_at_20', 'recall_at_30', 'recall_at_50'	
    									))
    for n in range(1, 2):
        # create training and valdation sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)
        for index, clf in enumerate([clfs[x] for x in models_to_run]):
            print(models_to_run[index])
            parameter_values = grid[models_to_run[index]]
            for p in ParameterGrid(parameter_values):
                try:
                    clf.set_params(**p)
                    y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
                    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                    results_df.loc[len(results_df)] = [models_to_run[index], clf, p,
                                                       baseline(X_train, X_test, y_train, y_test),
                                                       precision_at_k(y_test_sorted, y_pred_probs_sorted, 1.0),
                                                       precision_at_k(y_test_sorted, y_pred_probs_sorted, 2.0),
                                                       precision_at_k(y_test_sorted, y_pred_probs_sorted, 5.0),
                                                       precision_at_k(y_test_sorted, y_pred_probs_sorted, 10.0),
                                                       precision_at_k(y_test_sorted, y_pred_probs_sorted, 20.0),
                                                       precision_at_k(y_test_sorted, y_pred_probs_sorted, 30.0),
                                                       precision_at_k(y_test_sorted, y_pred_probs_sorted, 50.0),
                                                       recall_at_k(y_test_sorted, y_pred_probs_sorted, 1.0),
                                                       recall_at_k(y_test_sorted, y_pred_probs_sorted, 2.0),
                                                       recall_at_k(y_test_sorted, y_pred_probs_sorted, 5.0),
                                                       recall_at_k(y_test_sorted, y_pred_probs_sorted, 10.0),
                                                       recall_at_k(y_test_sorted, y_pred_probs_sorted, 20.0),
                                                       recall_at_k(y_test_sorted, y_pred_probs_sorted, 30.0),
                                                       recall_at_k(y_test_sorted, y_pred_probs_sorted, 50.0)
                                                       roc_auc_score(y_test, y_pred_probs)
                                                       ]
                except IndexError as e:
                    print('Error:',e)
                    continue
    return results_df
