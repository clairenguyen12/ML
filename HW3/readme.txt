Machine Learning Pipeline
Author: Chi Nguyen

Apply the machine learning pipeline to predict whether a project listed in Donors Choose will fail to receive fundings within 60 days of posting

I performed analysis and predictive modeling using Python pandas and scik-itlearn on a Donors Choose dataset, including school location, school type, teacher program, project topic, number of students reached and requested funding amount.

I trained multiple classifiers (Decision Trees, Bagging, Boosting, Random Forest, Logistic Regression, Support Vector Machine, K-nn Neighbors) each tuned to a different set of parameters, and evaluated them based on performance metrics such as precision, recall, f1, auc-roc to choose the best model.


The Pipeline contains 2 files:
	- mlhelper.py contains helper functions that will be
	used to calculate evaluation metrics (baseline, precision, recall, f1).
	It also contains the classifiers (Logistic Regression, K-Nearest Neighbor, 
	Decision Trees, SVM, Random Forests, Boosting, and Baggingand) 
	and their parameters.
	- build_models.py contains a function that leverages the for loop
	to create and evaluation models and a temporal_validation
	function to split the dataset into test and train sets over time. 
	The end result is a dataframe that stores all models created and 
	their performance metrics.


Analysis will be done in the Jupyter Notebook. File name is donors.ipynb

The report will be a pdf file that will be uploaded both to my personal
github and on Canvas.

