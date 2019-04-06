'''
Chi Nguyen
Machine Learning for Public Policy
Homework 1 - Diagnostic
'''

import csv
import pandas as pd
from sodapy import Socrata
import matplotlib.pyplot as plt
import seaborn as sns


def get_crime_data(year, filename):
    '''
    '''
    client = Socrata("data.cityofchicago.org", None)
    results = client.get("6zsd-86xi", 
                          content_type="csv", 
                          year=year, 
                          limit=500000)
    
    csv_file = open(filename, "w")
    writer = csv.writer(csv_file, delimiter='|')
    for row in results:
        writer.writerow(row)
    csv_file.close()


def analyze_crime_data():
    '''
    '''
    get_crime_data(2017, "crimes_2017.csv")
    get_crime_data(2018, "crimes_2018.csv")
    df_2017 = pd.read_csv("crimes_2017.csv", delimiter='|')
    df_2018 = pd.read_csv("crimes_2018.csv", delimiter='|')
    df = pd.concat([df_2017, df_2018])
    #crime in 2017
    df_2017['primary_type'].value_counts(normalize=True)
    #crime in 2018
    df_2018['primary_type'].value_counts(normalize=True)
    #comparing total crimes by type and year using pivot table
    table = pd.pivot_table(df,
                           index=['primary_type'],
                           columns=['year','arrest'],
                           values='id',
                           aggfunc='count',
                           margins=True,
                           fill_value=0) 
    print(table)
    #plot the change in incidents of crime by year
    table_for_plot = pd.pivot_table(df, 
                                    index='primary_type',
                                    columns='year',
                                    values='id',
                                    aggfunc='count',
                                    fill_value=0) 
    sns.set()
    table_for_plot.plot(kind='bar')
    plt.ylabel("Total incidents of crime")
    plt.show()
    #explore crime patter by ward







