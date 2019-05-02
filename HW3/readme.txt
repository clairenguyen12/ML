Machine Learning Pipeline
Author: Chi Nguyen


The Pipeline contains 3 files:
	- preprocess.py contains functions to load, clean data, 
	create boxplots and histograms. It also contains functions
	to convert continuous variables to categorical, and categorical
	variables to binary.
	- mlhelperfunctions.py contains helper functions that will be
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

