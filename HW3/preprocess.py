'''
Chi Nguyen
Improving Machine Learning Pipeline
'''

import os.path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def import_csv(csv_file):
    '''
    This function reads in a csv file and returns a dataframe
    '''
    if not os.path.exists(csv_file):
        print("File path does not exist")
        return None
    df = pd.read_csv(csv_file)
    return df


def explore_data(df):
    '''
    this function will take in a dataframe and print
    the first 10 lines of that dataframe, its shape,
    data types of each column, and the data distribution
    '''
    print("Let's take a look at the first 10 lines of the dataframe!")
    print()
    print(df.head(10))
    print('\n' * 2)
    print("Dataframe's shape: {}".format(df.shape))
    print('\n' * 2)
    print("Data types:")
    print()
    print(df.dtypes)
    print('\n' * 2)
    print("Distribution of the variables in the dataframe:")
    print()
    print(df.describe())


def create_boxplot(df, column_name, by_variable, x_lab, y_lab, title):
    '''
    This function will print a boxplot that shows the
    distribution of data

    Inputs:
        - df: a dataframe
        - column_name: string, the column to show distribution
        - by_variable: string, the column to create comparison groups
        - x_lab: string, the label for the x axis
        - y_lab: string, the label for the y axis
        - title: string, the title of the boxplot
    '''
    df.boxplot(column=column_name,
               by=by_variable).set(xlabel=x_lab,
                                   ylabel=y_lab)
    plt.title(title)
    plt.suptitle('')
    plt.show()


def find_outliers(df, column_name):
    '''
    This function will return a dataframe that contains
    the outliers for a given column

    Inputs:
        - df: a dataframe
        - column_name: string
    '''
    outliers = df[np.abs((df[column_name] -
                          df[column_name].mean()) / df[column_name].std()) > 3]
    print("Outliers' shape:")
    print(outliers.shape)
    return outliers


def create_corr_heatmap(df):
    '''
    This function will take in a dataframe and produce a
    correlation heatmap
    '''
    corr = df.corr()
    sns.heatmap(corr,
                xticklabels=corr.columns,
                yticklabels=corr.columns,
                cmap=sns.diverging_palette(220, 20, n=7, as_cmap=True))
    plt.show()


def pre_process(df, col_to_upcode=None):
    '''
    This function will fill in missing values and upcode
    values that seem illogical

    Inputs:
        - df: a dataframe
        - col_to_upcode: string, the column that needs to be upcoded
    '''
    print("Brief overview of the number of missing"
          "observations for each column:")
    print()
    print(df.isnull().sum().sort_values(ascending=False))
    print('\n'*2)
    null_col_list = df.columns[df.isna().any()].tolist()
    print()
    print("List of columns that contain missing data:")
    print()
    print(null_col_list)
    print('\n'*2)
    for col in null_col_list:
        df[col].fillna(df[col].median(), inplace=True)
    if col_to_upcode:
        df.loc[df[col_to_upcode] > 1, [col_to_upcode]] = 1
    print("Brief overview of the data distribution after pre-processing:")
    print()
    print(df.describe())


def convert_to_categorical_using_qcut(df, old_col, new_col, num_bin, label_list):
    '''
    This function converts a continuous variable into categorical
    using the pd.qcut method. The dataframe is transformed, function
    returns nothing.

    Inputs:
        - df: a dataframe
        - olc_col: (string) the name of the column to be converted
        - new_col: (string) the name of the new column that contains the
        categorical variable
        - num_bin: (int) number of bins
        - label_list: (list) list of labels for the categories
    '''
    df[new_col] = pd.qcut(df[old_col],
                          num_bin,
                          labels=label_list)


def convert_to_categorical_using_cut(df, old_col, new_col, bins, labels):
    '''
    This function converts a continuous variable into categorical
    using the pd.cut method. The dataframe is transformed, function
    returns nothing.

    Inputs:
        - df: a dataframe
        - olc_col: (string) the name of the column to be converted
        - new_col: (string) the name of the new column that contains the
        categorical variable
        - bins: (int) number of bins
        - labels: (list) list of labels for the categories
    '''
    df[new_col] = pd.cut(df[old_col],
                         bins=bins,
                         labels=labels,
                         include_lowest=True,
                         right=True)


def convert_to_binary(df, cols_to_transform):
    '''
    This function converts a categorical variable into binary

    Inputs:
        - df (dataframe)
        - cols_to_transform (list)
    '''
    df = pd.get_dummies(df, columns=cols_to_transform)
    return df
