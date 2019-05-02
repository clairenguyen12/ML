'''
Chi Nguyen
Improving Machine Learning Pipeline
This file contains helper functions that will be used 
throughout the pipeline. These helper functions are 
adapted from Rayid Ghani's Github at
https://github.com/rayidghani/magicloops/blob/master/mlfunctions.py
'''


from __future__ import division
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pylab as pl
from datetime import timedelta
import random
from scipy import optimize
import time
import seaborn as sns
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn import preprocessing, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import itertools


# modeling helper functions

def define_clfs_params(grid_size):

    '''
    This functions defines parameter grid for all the classifiers
    Inputs:
        grid_size: how big of a grid do you want. it can be test, small, or large
    Returns:
        a set of model and parameters
    Raises:
        KeyError: Raises an exception.
    '''

    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
            'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
            'LR': LogisticRegression(penalty='l1', C=1e5),
            'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
            'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
            'DT': DecisionTreeClassifier(),
            'KNN': KNeighborsClassifier(n_neighbors=3) 
            }

    large_grid = { 
    'RF': {'n_estimators': [1,10,100,1000,10000], 
           'max_depth': [1,5,10,20,50,100], 
           'max_features': ['sqrt','log2'],
           'min_samples_split': [2,5,10], 
           'n_jobs': [-1]},
    'AB': {'algorithm': ['SAMME', 'SAMME.R'], 
           'n_estimators': [1,10,100,1000,10000]},
    'LR': {'penalty': ['l1','l2'], 
           'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'SVM': {'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'GB': {'n_estimators': [1,10,100,1000,10000], 
           'learning_rate' : [0.001,0.01,0.05,0.1,0.5],
           'subsample' : [0.1,0.5,1.0], 
           'max_depth': [1,3,5,10,20,50,100]},
    'DT': {'criterion': ['gini', 'entropy'], 
           'max_depth': [1,5,10,20,50,100], 
           'max_features': [None, 'sqrt','log2'],
           'min_samples_split': [2,5,10]},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],
            'weights': ['uniform','distance'],
            'algorithm': ['auto','ball_tree','kd_tree']}
           }

    small_grid = { 
    'RF':{'n_estimators': [100, 10000], 
          'max_depth': [5,50], 
          'max_features': ['sqrt','log2'],
          'min_samples_split': [2,10], 
          'n_jobs':[-1]},
    'AB': {'algorithm': ['SAMME', 'SAMME.R'], 
           'n_estimators': [1,10,100,1000,10000]},
    'LR': {'penalty': ['l1','l2'], 
           'C': [0.00001,0.001,0.1,1,10]},
    'SVM' :{'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'GB': {'n_estimators': [100, 10000], 
           'learning_rate' : [0.001,0.1,0.5],
           'subsample' : [0.1,0.5,1.0], 
           'max_depth': [5,50]},
    'DT': {'criterion': ['gini', 'entropy'], 
           'max_depth': [1,5,10,20,50,100], 
           'max_features': [None,'sqrt','log2'],
           'min_samples_split': [2,5,10]},
    'KNN': {'n_neighbors': [1,5,10,25,50,100],
            'weights': ['uniform','distance'],
            'algorithm': ['auto','ball_tree','kd_tree']}
           }
    
    test_grid = { 
    'RF': {'n_estimators': [1], 
           'max_depth': [1], 
           'max_features': ['sqrt'],
           'min_samples_split': [10], 
           'n_jobs': [-1]},
    'AB': {'algorithm': ['SAMME'], 
           'n_estimators': [1]},
    'LR': {'penalty': ['l1'], 
           'C': [0.01]},
    'SVM': {'C' :[0.01]},
    'GB': {'n_estimators': [1], 
           'learning_rate' : [0.1],
           'subsample' : [0.5], 
           'max_depth': [1]},
    'DT': {'criterion': ['gini'], 
           'max_depth': [1], 
           'max_features': [None],
           'min_samples_split': [10]},
    'KNN': {'n_neighbors': [5],
            'weights': ['uniform'],
            'algorithm': ['auto']}
           }

    small_grid = { 
    'RF':{'n_estimators': [100, 10000], 
          'max_depth': [5,50], 
          'max_features': ['sqrt','log2'],
          'min_samples_split': [2,10], 
          'n_jobs':[-1]},
    'AB': {'algorithm': ['SAMME', 'SAMME.R'], 
           'n_estimators': [1,10,100]},
    'LR': {'penalty': ['l1','l2'], 
           'C': [0.001,0.1,1,10]},
    'SVM' :{'C': [0.01,0.1,1,10]},
    'GB': {'n_estimators': [100, 10000], 
           'learning_rate' : [0.1,0.5],
           'subsample' : [0.1,0.5,1.0], 
           'max_depth': [5,20]},
    'DT': {'criterion': ['gini', 'entropy'], 
           'max_depth': [1,5,10,20], 
           'max_features': [None,'sqrt','log2'],
           'min_samples_split': [2,5,10]},
    'KNN': {'n_neighbors': [1,5,50],
            'weights': ['uniform','distance'],
            'algorithm': ['auto','ball_tree','kd_tree']}
           }
    
    if (grid_size == 'large'):
        return clfs, large_grid
    elif (grid_size == 'small'):
        return clfs, small_grid
    elif (grid_size == 'test'):
        return clfs, test_grid
    elif (grid_size == 'mini'):
        return clfs, mini_grid
    else:
        return 0, 0


# Evaluation functions
# calculate precision, recall and auc metrics

def baseline(X_train, X_test, y_train, y_test):
    '''
    '''
    clf = DummyClassifier(strategy='most_frequent', random_state=0)
    clf.fit(X_train, y_train)
    baseline = clf.score(X_test, y_test)
    return baseline


def precision_at_k(y_true, y_scores, k):
    '''
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    precision = precision_score(y_true_sorted, preds_at_k)
    return precision


def recall_at_k(y_true, y_scores, k):
    '''
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    recall = recall_score(y_true_sorted, preds_at_k)
    return recall


def f1_at_k(y_true, y_scores, k):
    '''
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    f1 = f1_score(y_true_sorted, preds_at_k)
    return f1


def plot_precision_recall_n(y_true, y_prob, model_name):
    '''
    '''
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score >= value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    
    name = model_name
    plt.title(name)
    plt.show()


def joint_sort_descending(l1, l2):
    '''
    Inputs:
        - l1: numpy array
        - l2: numpy array
    '''
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]


def generate_binary_at_k(y_scores, k):
    '''
    '''
    cutoff_index = int(len(y_scores) * (k / 100.0))
    predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return predictions_binary

