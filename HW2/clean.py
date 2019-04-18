'''
Chi Nguyen
Machine Learning Pipeline
'''

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as accuracy
import graphviz
import os.path


def import_csv(csv_file):
    '''
    '''
    if not os.path.exists(csv_file):
        print("File path does not exist")
        return None
    df = pd.read_csv(csv_file)
    return df


def explore_data(df):
    '''
    '''
    print("Let's take a look at the first 10 lines of the dataframe!")
    print(df.head(10))
    print()
    print("Dataframe's shape: {}".format(df.shape))
    print()
    print("Data types:")
    print(df.dtypes)
    print()
    print("Distribution of the variables in the dataframe:")
    print(df.describe())


def create_boxplot(df, column_name, by_variable, x_lab, y_lab, title):
    '''
    '''
    df.boxplot(column=column_name, 
               by=by_variable).set(xlabel=x_lab, 
                                   ylabel=y_lab)
    plt.title(title)
    plt.suptitle('')
    plt.show()


def find_outliers(df, column_name):
    '''
    '''
    outliers = df[np.abs((df[column_name] - 
                  df[column_name].mean()) / df[column_name].std()) > 3]
    print("Outliers' shape:")
    print(outliers.shape)
    return outliers


def create_corr_heatmap(df):
    '''
    '''
    corr = df.corr()
    sns.heatmap(corr, 
                xticklabels=corr.columns, 
                yticklabels=corr.columns, 
                cmap=sns.diverging_palette(220, 20, n=7, as_cmap=True))
    plt.show()





