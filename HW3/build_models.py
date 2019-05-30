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

from mlhelper import *


RANDOM_STATE = 100


def find_nuls(df):
    '''
    This function finds all the null columns in the dataframe
    and return a list of such columns
    '''
    print(df.isnull().sum().sort_values(ascending=False))
    null_col_list = df.columns[df.isna().any()].tolist()
    return null_col_list


def fill_null_cols(df, null_col_list):
    '''
    '''
    for col in null_col_list:
        try:
            df[col].fillna(df[col].median(), inplace=True)
        except:
            print("Can't fill missing values for non-numeric column {}".format(col))
            continue


def discretize_cols(df, old_col, num_bins=3, labels=False):
    '''
    Inputs:
        - dataframe (pandas dataframe)
        - feature (string): label of column to discretize
        - num_bins (int): number of bins to discretize into
        - labels
    Returns a pandas dataframe
    '''
    new_col = old_col + '_group'
    df[new_col] = pd.cut(df[old_col], 
                         bins=num_bins, 
                         labels=labels, 
                         right=True, 
                         include_lowest=True)
    return df


def convert_to_binary(df, cols_to_transform):
    '''
    This function converts a categorical variable into binary

    Inputs:
        - df (dataframe)
        - cols_to_transform (list)
    '''
    df = pd.get_dummies(df, dummy_na=True, columns=cols_to_transform)
    return df


def convert_to_datetime(df, cols_to_transform):
    '''
    '''
    for col in cols_to_transform:
        df[col] = pd.to_datetime(df[col])


def process_train_data(rv, cols_to_discretize, num_bins, labels, cols_to_binary):
    '''
    '''
    processed_data = []
    processed_rv = {}
    for split_date, data in rv.items():
        for df in data:
            fill_null_cols(df, find_nuls(df))
            for col in cols_to_discretize:
                processed_df = discretize_cols(df, col, num_bins, labels)
            processed_df = convert_to_binary(processed_df, cols_to_binary)
            processed_data.append(processed_df)
        processed_rv[split_date] = processed_data
    return processed_rv


def clf_loop_cross_validation(models_to_run, clfs, grid, processed_rv, 
                              predictors, outcome, thresholds, test_size=0.2,
                              target=0.05):
    '''
    '''
    metrics = ['p1_at_', 'recall_at_', 'f1_at_']
    metric_cols = []
    for thres in thresholds:
        for metric in metrics:
            metric_cols.append(metric + str(thres))
    COLS = ['model_type', 'clf', 'parameters', 'split_date', 'baseline'] + \
           metric_cols + \
           ['auc-roc', 'target_threshold_top_5_percent', 'precision_at_target', 
           'recall_at_target', 'f1_at_target']
    
    results_df =  pd.DataFrame(columns=COLS)
    i = 0
    for n in range(1, 2):
        for split_date, data in processed_rv.items():
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
                        target_index = int(target*len(y_pred_probs_sorted))
                        target_threshold = y_pred_probs_sorted[target_index]

                        metrics_stats = []
                        for thres in thresholds:
                            pres = precision_at_k(y_test_sorted, y_pred_probs_sorted, thres)
                            rec = recall_at_k(y_test_sorted, y_pred_probs_sorted, thres)
                            f1 = f1_at_k(y_test_sorted, y_pred_probs_sorted, thres)
                            metrics_stats.extend([pres, rec, f1])

                        row = [models_to_run[index], clf, p, split_date] + \
                              [precision_at_k(y_test_sorted, y_pred_probs_sorted, 100)] + \
                              metrics_stats + \
                              [roc_auc_score(y_test, y_pred_probs),
                               target_threshold,
                               precision_at_k(y_test_sorted, y_pred_probs_sorted, target_threshold),
                               recall_at_k(y_test_sorted, y_pred_probs_sorted, target_threshold),
                               f1_at_k(y_test_sorted, y_pred_probs_sorted, target_threshold)]
                        results_df.loc[len(results_df)] = row
                        i +=1
                        #Plot the precision recall curves
                        plot_precision_recall_n(y_test, y_pred_probs, clf)
                    except IndexError as e:
                        print('Error:',e)
                        continue
    return results_df


def temporal_validation(df, date_col, prediction_windows, gap, start_time, end_time):
    '''
    Create a dictionary that maps a key that is the validation date with a list
    of train set and test set that correspond to that validation date.
    Train set will contain records before validation date
    Test set will contain records after validation date

    Inputs:
        - df: a dataframe
        - date_col: the date column
        - prediction_windows: a list that contains all prediction windows in months
        - gap: the number of days between train end date and test start date 
        - start_time: string, earliest datetime in the data
        - end_time: string, latest datetime in the data

    Outputs:
        a dictionary that maps the validation date to a list that contains the
        corresponding train set and test set for that date
    '''
    start_time_date = datetime.strptime(start_time, '%Y-%m-%d')
    end_time_date = datetime.strptime(end_time, '%Y-%m-%d')
    train_start_time = start_time_date
    rv = {}
    for prediction_window in prediction_windows:
        test_end_time = end_time_date
        while test_end_time >= start_time_date + relativedelta(months=prediction_window):
            test_start_time = test_end_time - relativedelta(months=prediction_window)
            #leaving a gap between the train set and the test set
            train_end_time = test_start_time - relativedelta(days=gap)
            train_set = df[(df[date_col] >= train_start_time) &
                           (df[date_col] <= train_end_time)]
            test_set = df[(df[date_col] >= test_start_time) &
                           (df[date_col] <= test_end_time)]
            rv[test_start_time] = [train_set, test_set]
            test_end_time -= relativedelta(months=prediction_window)
    return rv
