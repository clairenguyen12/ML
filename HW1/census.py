'''
Chi Nguyen
Machine Learning for Public Policy
Homework 1 - Diagnostic
'''

import requests
import censusdata
import geopandas as gpd
import pandas as pd


def get_tract_spatial_data(): 
     ''' 
     ''' 
     client = Socrata("data.cityofchicago.org", None) 
     results = client.get("bt9m-d2mf", limit=100000) 
     results_df = pd.DataFrame.from_records(results) 
     return results_df

def get_polygon(row):
    '''
    '''
    coordinates = row['the_geom']['coordinates'][0][0] 
    polygon = Polygon(coordinates) 
    return polygon


def transform_tract_spatial_data():
    '''
    '''
    df = get_tract_spatial_data()
    df['geometry'] = df.apply(get_polygon, axis=1)
    return df 


def get_census_data():
    '''
    '''
    tables = ("B25010_001E,B19013_001E,B03002_012E,B03002_003E,"
              "B03002_004E,B03002_005E,B03002_006E,B25013_002E,"
              "B25013_007E,B25013_003E,B25013_008E")
    col_dict = {}
    for col in tables.split(","):
        concept = censusdata.censustable('acs5',2017,col[:6])[col]['concept']
        label = censusdata.censustable('acs5',2017,col[:6])[col]['label']
        col_dict[col] = label + " " + concept

    census_api_key = '3fe9e22eeba4c4df8dec801a8308938e3de723bd'
    url = ("https://api.census.gov/data/2017/acs/acs5?"
           "get=") + tables + (",NAME&"
           "for=block%20group:*&in=state:17&in=county:031&in=tract:*&"
           "key=") + census_api_key
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame(data)
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    df = df.rename(index=str, columns=col_dict)
    return df
    




