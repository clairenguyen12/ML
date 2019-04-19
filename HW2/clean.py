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
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as accuracy
import graphviz
import os.path
from sklearn.externals.six import StringIO 
import pydotplus
from IPython.display import Image


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
    print()
    print(df.head(10))
    print()
    print()
    print("Dataframe's shape: {}".format(df.shape))
    print()
    print()
    print("Data types:")
    print()
    print(df.dtypes)
    print()
    print()
    print("Distribution of the variables in the dataframe:")
    print()
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


def pre_process(df, col_to_upcode=None):
    '''
    '''
    print("Brief overview of the number of missing" 
          "observations for each column:")
    print()
    print(df.isnull().sum().sort_values(ascending=False))
    print()
    print()
    null_col_list = df.columns[df.isna().any()].tolist()
    print()
    print("List of columns that contain missing data:")
    print()
    print(null_col_list)
    print()
    print()
    for col in null_col_list:
        df[col].fillna(df[col].median(), inplace=True)
    if col_to_upcode:
        df.loc[df[col_to_upcode] > 1, [col_to_upcode]] = 1
    print("Brief overview of the data distribution after pre-processing:")
    print()
    print(df.describe())


def convert_to_categorical_using_qcut(df, old_col, new_col, num_bin, label_list):
    '''
    '''
    df[new_col] = pd.qcut(df[old_col], 
                          num_bin, 
                          labels=label_list)


def convert_to_categorical_using_cut(df, old_col, new_col, bins, labels):
    '''
    '''
    df[new_col] = pd.cut(df[old_col],
                         bins=bins,
                         labels=labels,
                         include_lowest=True,
                         right=True)


def convert_to_binary(df, cols_to_transform):
    '''
    '''
    df = pd.get_dummies(df, columns=cols_to_transform)
    return df


def build_classifiers(selected_features, outcome_col, 
                      test_size, df, random_state):
    '''
    '''
    x = df[selected_features]
    y = df[outcome_col]
    x_train, x_test, y_train, y_test = train_test_split(x, 
                                                        y, 
                                                        test_size=test_size, 
                                                        random_state=random_state)
    return (x_train, x_test, y_train, y_test)


def build_tree(max_depth, x_train, y_train):
    '''
    '''
    dec_tree = DecisionTreeClassifier(max_depth=max_depth)
    dec_tree.fit(x_train, y_train)
    return dec_tree


def evaluate_classifier(dec_tree, x_train, y_train, x_test, y_test, threshold):
    '''
    '''
    print("Histogram to show predicted scores when performed on test data:")
    predicted_scores_test = dec_tree.predict_proba(x_test)[:,1]
    plt.hist(predicted_scores_test) 
    plt.show()
    print()
    calc_threshold = lambda x,y: 0 if x < y else 1 
    predicted_test = np.array([calc_threshold(score, threshold) 
                              for score in predicted_scores_test])
    test_acc = accuracy(predicted_test, y_test)
    print("With the chosen threshold {}, chosen max_depth {} and criterion {}"
          " the accuracy of the model is {}".format(threshold, 
                                                   dec_tree.max_depth,
                                                   dec_tree.criterion, 
                                                   test_acc))
    return test_acc


def rank_features(dec_tree, x_train):
    '''
    '''
    df = pd.DataFrame()
    for name, importance in zip(x_train.columns, dec_tree.feature_importances_):
        rv = {'Feature': name, 'Importance Weight': importance}
        df = df.append(rv, ignore_index=True)
        df = df.sort_values(by='Importance Weight', ascending=False)
    return df


def evaluate_accuracy_by_max_depth(threshold, depth_list, 
                                     x_train, y_train, x_test, y_test):
    '''
    '''
    calc_threshold = lambda x,y: 0 if x < y else 1
    for d in depth_list:
        dec_tree = DecisionTreeClassifier(max_depth=d)
        dec_tree.fit(x_train, y_train)
        train_scores = dec_tree.predict_proba(x_train)[:,1]
        test_scores = dec_tree.predict_proba(x_test)[:,1]
        predicted_train = np.array([calc_threshold(score, threshold) 
                                    for score in train_scores])
        train_acc = accuracy(predicted_train, y_train)
        predicted_test = np.array([calc_threshold(score, threshold) 
                                   for score in test_scores])
        test_acc = accuracy(predicted_test, y_test) 
        print("Depth: {} | Train acc: {:.2f} | Test acc: {:.2f}".format(
            d, train_acc, test_acc))


def evaluate_accuracy_by_threshold(max_depth, threshold_list,
                                   x_train, y_train, x_test, y_test):
    '''
    '''
    calc_threshold = lambda x,y: 0 if x < y else 1
    for threshold in threshold_list:
        dec_tree = DecisionTreeClassifier(max_depth=max_depth)
        dec_tree.fit(x_train, y_train)
        train_scores = dec_tree.predict_proba(x_train)[:,1]
        test_scores = dec_tree.predict_proba(x_test)[:,1]
        predicted_train = np.array([calc_threshold(score, threshold) 
                                    for score in train_scores])
        train_acc = accuracy(predicted_train, y_train)
        predicted_test = np.array([calc_threshold(score, threshold) 
                                   for score in test_scores])
        test_acc = accuracy(predicted_test, y_test) 
        print("Threshold: {} | Train acc: {:.2f} | Test acc: {:.2f}".format(
            threshold, train_acc, test_acc))


def visualize_dec_tree(dec_tree):
    '''
    '''
    dot_data = StringIO()
    export_graphviz(dec_tree, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    Image(graph.create_png())
