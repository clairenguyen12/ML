'''
Chi Nguyen
Machine Learning for Public Policy
Homework 1 - Diagnostic
'''

import pandas as pd
from sodapy import Socrata

def get_crime_data(year):
    '''
    '''
    client = Socrata("data.cityofchicago.org", None)
    results = client.get("6zsd-86xi", 
                          content_type="csv", 
                          year=year, 
                          limit=500000)
    return results

