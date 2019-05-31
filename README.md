# Machine Learning for Public Policy
Student Name: Chi Nguyen

Folder: HW3

Machine Learning Pipeline - Spring 2019


**Goal**

Create a machine learning pipeline with functions that do the following tasks:
1) Load data
2) Explore data
3) Split data into train and test sets and process data
4) Generate label and features
5) Build different classifiers
6) Generate an outcome report with different performance metrics


**Application**

Apply the machine learning pipeline to predict whether a project listed in Donors Choose will fail to receive fundings within 60 days of posting

I performed analysis and predictive modeling using Python pandas and scik-itlearn on a Donors Choose dataset, including school location, school type, teacher program, project topic, number of students reached and requested funding amount.

I trained multiple classifiers (Decision Trees, Bagging, Boosting, Random Forest, Logistic Regression, Support Vector Machine, K-nn Neighbors) each tuned to a different set of parameters, and evaluated them based on performance metrics such as precision, recall, f1, auc-roc to choose the best model.
