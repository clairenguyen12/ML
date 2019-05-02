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

from __future__ import division
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pylab as pl
from datetime import timedelta
from datetime import datetime
import random
from scipy import optimize
import time
import seaborn as sns
import csv
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import roc_auc_score 
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

from mlhelperfunctions import *


RANDOM_STATE = 100


def clf_loop_cross_validation(models_to_run, clfs, grid, df, predictors, outcome,
                              date_col, prediction_windows, start_time, end_time, test_size=0.2):
    '''
    '''
    rv = temporal_validation(df, date_col, prediction_windows, start_time, end_time)
    results_df =  pd.DataFrame(columns=('model_type', 'clf', 'parameters', 'split_date', 
                                        'p_at_1', 'p_at_2', 'p_at_5',
                                        'p_at_10', 'p_at_20', 'p_at_30', 'p_at_50',
                                        'recall_at_1', 'recall_at_2', 'recall_at_5',
                                        'recall_at_10', 'recall_at_20', 'recall_at_30', 'recall_at_50',
                                        'f1_at_5', 'f1_at_20', 'f1_at_50',
                                        'auc-roc', 'target_threshold_top_5_percent', 
                                        'precision_at_target', 'recall_at_target', 'f1_at_target'))
    i = 0
    for n in range(1, 2):
        for split_date, data in rv.items():
            train_set = data[0]
            test_set = data[1]
            X_train = train_set[predictors]
            X_test = test_set[predictors]
            y_train = train_set[outcome]
            y_test = test_set[outcome]
            for index, clf in enumerate([clfs[x] for x in models_to_run]):
                model_name = models_to_run[index]
                print(model_name)
                parameter_values = grid[models_to_run[index]]
                for p in ParameterGrid(parameter_values):
                    try:
                        clf.set_params(**p)
                        clf.fit(X_train, y_train)
                        if model_name == 'SVM':
                            y_pred_probs = clf.decision_function(X_test)
                        else:
                            y_pred_probs = clf.predict_proba(X_test)[:,1]                        
                        y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                        target_index = int(0.05*len(y_pred_probs_sorted))
                        target_threshold = y_pred_probs_sorted[target_index]
                        row = [models_to_run[index], clf, p, split_date,
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
                               recall_at_k(y_test_sorted, y_pred_probs_sorted, 50.0),
                               f1_at_k(y_test_sorted, y_pred_probs_sorted, 5.0),
                               f1_at_k(y_test_sorted, y_pred_probs_sorted, 20.0),
                               f1_at_k(y_test_sorted, y_pred_probs_sorted, 50.0),
                               roc_auc_score(y_test, y_pred_probs),
                               target_threshold,
                               precision_at_k(y_test_sorted, y_pred_probs_sorted, target_threshold),
                               recall_at_k(y_test_sorted, y_pred_probs_sorted, target_threshold),
                               f1_at_k(y_test_sorted, y_pred_probs_sorted, target_threshold) 
                               ]
                        results_df.loc[len(results_df)] = row
                        i +=1
                        print("Added row {}".format(i))
                    except IndexError as e:
                        print('Error:',e)
                        continue
    return results_df 


def normal_clf_loop(models_to_run, clfs, grid, X, y, test_size=0.2):
    '''
    Runs the loop using models_to_run, clfs, gridm and the data
    Unless test_size is specified, default value 0.2 will be used
    '''
    results_df =  pd.DataFrame(columns=('model_type', 'clf', 'parameters', 'baseline', 
                                        'p_at_1', 'p_at_2', 'p_at_5', 
                                        'p_at_10', 'p_at_20', 'p_at_30', 'p_at_50',
                                        'recall_at_1', 'recall_at_2', 'recall_at_5',
                                        'recall_at_10', 'recall_at_20', 'recall_at_30', 'recall_at_50',
                                        'auc-roc'))
    for n in range(1, 2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)
        for index, clf in enumerate([clfs[x] for x in models_to_run]):
            model_name = models_to_run[index]
            print(model_name)
            parameter_values = grid[models_to_run[index]]
            for p in ParameterGrid(parameter_values):
                try:
                    clf.set_params(**p)
                    clf.fit(X_train, y_train)
                    if model_name == 'SVM':
                        y_pred_probs = clf.decision_function(X_test)
                    else:
                        y_pred_probs = clf.predict_proba(X_test)[:,1]
                    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                    row = [models_to_run[index], clf, p,
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
                           recall_at_k(y_test_sorted, y_pred_probs_sorted, 50.0),
                           roc_auc_score(y_test, y_pred_probs)]
                    results_df.loc[len(results_df)] = row
                    #print(row)
                except IndexError as e:
                    print('Error:',e)
                    continue
    return results_df


def temporal_validation(df, date_col, prediction_windows, start_time, end_time):
    '''
    Create a dictionary that maps a key that is the validation date with a list
    of train set and test set that correspond to that validation date.
    Train set will contain records before validation date
    Test set will contain records after validation date

    Inputs:
        - df: a dataframe
        - date_col: the date column
        - prediction_windows: a list that contains all prediction windows in months
        - start_time: string, earliest datetime in the data
        - end_time: string, latest datetime in the data
    '''
    start_time_date = datetime.strptime(start_time, '%Y-%m-%d')
    end_time_date = datetime.strptime(end_time, '%Y-%m-%d')
    train_start_time = start_time_date
    rv = {}
    for prediction_window in prediction_windows:
        test_end_time = end_time_date
        while test_end_time >= start_time_date + relativedelta(months=prediction_window):
            test_start_time = test_end_time - relativedelta(months=prediction_window)
            train_end_time = test_start_time - relativedelta(days=1)
            train_set = df[(df[date_col] >= train_start_time) &
                           (df[date_col] <= train_end_time)]
            test_set = df[(df[date_col] >= test_start_time) &
                           (df[date_col] <= test_end_time)]
            rv[test_start_time] = [train_set, test_set]
            test_end_time -= relativedelta(months=prediction_window)
    return rv
