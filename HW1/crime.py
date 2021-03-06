'''
Chi Nguyen
Machine Learning for Public Policy
Homework 1 - Diagnostic
'''

import csv
import pandas as pd
from sodapy import Socrata
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.geometry import Point
import geopandas as gpd
import rtree
import requests
import seaborn


def get_crime_data(year, filename):
    '''
    Download data from Chicago Open Data Portal using the API
    Inputs:
    year: int
    filename: string
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


def process_crime_data():
    '''
    Process crime data downloaded and return a dataframe
    that contains all crime reports in year 2017 and 2018
    '''
    get_crime_data(2017, "crimes_2017.csv")
    get_crime_data(2018, "crimes_2018.csv")
    df_2017 = pd.read_csv("crimes_2017.csv", delimiter='|')
    df_2018 = pd.read_csv("crimes_2018.csv", delimiter='|')
    df = pd.concat([df_2017, df_2018])
    return df


def get_tract_spatial_data():
    '''
    Download tract spatial data using the API from
    Chicago Open Data Portal
    '''
    client = Socrata("data.cityofchicago.org", None)
    results = client.get("74p9-q2aq", limit=100000)
    results_df = pd.DataFrame.from_records(results)
    return results_df


def get_polygon(row):
    '''
    Turn a list of coordinates into a polygon object
    '''
    coordinates = row['the_geom']['coordinates'][0][0]
    polygon = Polygon(coordinates)
    return polygon


def get_point(row):
    '''
    Turn a tuple of lon and lat into a point object
    '''
    point = Point(row['longitude'], row['latitude'])
    return point


def merge_crime_geodata():
    '''
    This function merges the crime dataframe and
    the tract spatial data and returns a geo dataframe
    '''
    crime_df = process_crime_data()
    crime_df = crime_df.dropna(subset=['longitude', 'latitude'])
    crime_df['geometry'] = crime_df.apply(get_point, axis=1)
    crime_geodf = gpd.GeoDataFrame(crime_df)
    tract_df = get_tract_spatial_data()
    tract_df['geometry'] = tract_df.apply(get_polygon, axis=1)
    tract_geodf = gpd.GeoDataFrame(tract_df)
    merged_geodf = gpd.sjoin(crime_geodf, tract_geodf, how="left", op='intersects')
    return merged_geodf


def analyze_crime_data():
    '''
    This function analyzes the crime dataframe that is
    collected from the Chicago Open Data Portal
    '''
    crime_df = process_crime_data()
    #crime in 2017
    df_2017 = pd.read_csv("crimes_2017.csv", delimiter='|')
    df_2018 = pd.read_csv("crimes_2018.csv", delimiter='|')
    print("Total crimes by type in 2017")
    print(df_2017['primary_type'].value_counts(normalize=True))
    #crime in 2018
    print("Total crimes by type in 2018")
    print(df_2018['primary_type'].value_counts(normalize=True))
    #total crime incidents by year
    print("Total crimes in 2017 and 2018")
    print(crime_df['year'].value_counts())
    print("Percent change in crime in 2018 compared to 2017")
    print((len(df_2018) - len(df_2017)) / len(df_2017))
    #plot the change in incidents of crime by year
    table_for_plot = pd.pivot_table(crime_df,
                                    index='primary_type',
                                    columns='year',
                                    values='id',
                                    aggfunc='count',
                                    fill_value=0)
    seaborn.set()
    table_for_plot.plot(kind='barh', fontsize=6, title='Crime type by year')
    plt.ylabel("Total incidents of crime")
    plt.show()
    #community areas with the most crimes in both 2017 and 2018:
    print("Top 10 community areas with the"
          "most total crime reports in 2017 and 2018")
    print(crime_df['community_area'].value_counts(normalize=True).head(10))
    #crime trends over time: crimes increase in the summer and decrease in the winter
    #correlation between crime and weather
    crime_df['date'] = pd.to_datetime(crime_df['date'])
    over_time = crime_df.groupby(pd.Grouper(key='date', freq='MS')).size().reset_index()
    over_time.columns = ['date', 'Total crimes']
    over_time.set_index('date', inplace=True)
    over_time.plot()
    plt.show()
    #exploring crime by month and type
    print("Testifying against Ward 43' alderman's statement")
    ward43 = crime_df[crime_df['ward'] == 43]
    ward43['date'] = pd.to_datetime(ward43['date'])
    ward43['month_year'] = ward43['date'].dt.to_period('M')
    crime_by_month = pd.pivot_table(ward43,
                                    index='primary_type',
                                    columns='month_year',
                                    values='id',
                                    aggfunc='count',
                                    fill_value=0)
    crime_by_month.loc['TOTAL MONTHLY CRIMES'] = crime_by_month.sum()
    crime_by_month = crime_by_month.reset_index()
    crime_by_month.columns.values[7] = 'Jul 2017'
    crime_by_month.columns.values[19] = 'Jul 2018'
    cols_to_keep = ['primary_type', 'Jul 2017', 'Jul 2018']
    crime_july = crime_by_month[cols_to_keep]
    some_crime_july = crime_july[(crime_july['primary_type'] == 'ROBBERY') |
                                 (crime_july['primary_type'] == 'BATTERY') |
                                 (crime_july['primary_type'] == 'BURGLARY') |
                                 (crime_july['primary_type'] == 'MOTOR VEHICLE THEFT') |
                                 (crime_july['primary_type'] == 'TOTAL MONTHLY CRIMES')]
    some_crime_july['% Increase'] = ((some_crime_july['Jul 2018'] -
                                      some_crime_july['Jul 2017']
                                      ) / some_crime_july['Jul 2017']) * 100
    print(some_crime_july)


def get_census_data():
    '''
    This function downloads census data using an API
    '''
    tables = ("B25010_001E,B19013_001E,B03002_001E,B03002_018E,"
              "B03002_003E,B03002_004E,B03002_005E,B03002_006E,"
              "B23006_001E,B23006_002E")
    table_names = ['Household Size',
                   'Median Household Income',
                   'Total population',
                   'Hispanic',
                   'White',
                   'Black',
                   'Native American',
                   'Asian',
                   'Total age 25 to 64',
                   'Total age 25 to 64 less than HS']
    col_dict = {}
    table_lst = tables.split(",")

    for i in range(len(table_lst)):
        col_dict[table_lst[i]] = table_names[i]

    census_api_key = '3fe9e22eeba4c4df8dec801a8308938e3de723bd'
    url = ("https://api.census.gov/data/2017/acs/acs5?"
           "get=") + tables + (",NAME&"
                               "for=tract:*&in=state:17&in=county:031&"
                               "key=") + census_api_key
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame(data)
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    df = df.rename(index=str, columns=col_dict)
    for col in table_names:
        df = df.astype({col:float})

    df['% Hispanic'] = df['Hispanic'] / df['Total population']
    df['% White'] = df['White'] / df['Total population']
    df['% Black'] = df['Black'] / df['Total population']
    df['% Native American'] = df['Native American'] / df['Total population']
    df['% Asian'] = df['Asian'] / df['Total population']
    df['% Age 25 to 64 less than HS'] = df[
        'Total age 25 to 64 less than HS'] / df['Total age 25 to 64']
    df = df[df['Household Size'] != -666666666]
    df = df[df['Median Household Income'] != -666666666]
    return df


def merge_crime_census_data():
    '''
    This function merges the crime geodataframe and
    the census data
    '''
    crime_df = merge_crime_geodata()
    census_df = get_census_data()
    final_df = crime_df.merge(census_df, how='left', left_on='tractce10', right_on='tract')
    return final_df


def get_aggregated_data():
    '''
    This function merges crime geodataframe and census data,
    and produce an aggregated version of total crime by types
    by tracts
    '''
    crime_df = merge_crime_geodata()
    census_df = get_census_data()
    df = pd.DataFrame(crime_df)
    agg = crime_df.groupby([
        'tractce10', 'year', 'primary_type']).size().unstack().reset_index()
    agg = agg.fillna(0)
    agg_census = agg.merge(census_df, how='left', left_on='tractce10', right_on='tract')
    return agg_census


def analyze_aggregated_data():
    '''
    This function analyzes the aggregated dataframe
    that is returned from the get_aggregated_data()
    '''
    agg_census = get_aggregated_data()
    #Batteries tend to happen in poor neighborhoods
    seaborn.scatterplot(x='BATTERY', y='Median Household Income', data=agg_census)
    plt.show()
    #Homicides - '% Black' & 'Median Household Income'
    seaborn.scatterplot(x='HOMICIDE', y='Median Household Income', data=agg_census)
    plt.show()
    seaborn.scatterplot(x='HOMICIDE', y='% Black', data=agg_census)
    plt.show()
    #Crime over time
    seaborn.scatterplot(x='BATTERY', y='Median Household Income', data=agg_census, hue='year')
    plt.show()
    seaborn.scatterplot(x='HOMICIDE', y='Median Household Income', data=agg_census, hue='year')
    plt.show()
    #Checking correlation
    print("Correlation between Theft and Median Household Income")
    print(agg_census['THEFT'].corr(agg_census['Median Household Income']))
    print("Correlation between Theft and % working age people without than high school degree")
    print(agg_census['THEFT'].corr(agg_census['% Age 25 to 64 less than HS']))

    print("Correlation between Battery and Median Household Income")
    print(agg_census['BATTERY'].corr(agg_census['Median Household Income']))

    print("Correlation between Sex Offense and Median Household Income")
    print(agg_census['SEX OFFENSE'].corr(agg_census['Median Household Income']))
    print("Correlation between Deceptive Practice and Median Household Income")
    print(agg_census['DECEPTIVE PRACTICE'].corr(agg_census['Median Household Income']))

    print("Correlation between Sex Offense and Household Size")
    print(agg_census['DECEPTIVE PRACTICE'].corr(agg_census['Household Size']))
    print("Correlation between Deceptive Practice and Household Size")
    print(agg_census['SEX OFFENSE'].corr(agg_census['Household Size']))


def analyze_prob_911_calls():
    '''
    This function will be used to answer question 4 in hw1
    '''
    chi_crime = process_crime_data()
    block_of_interest = chi_crime[chi_crime['block'] == '021XX S MICHIGAN AVE']
    print(block_of_interest['primary_type'].value_counts(normalize=True))
    #Community area number for Garfield Park is 26(East) and 27(West)
    theft = chi_crime[chi_crime['primary_type'] == 'THEFT']
    total_thefts = len(theft)
    w_garfield = chi_crime[(chi_crime['primary_type'] == 'THEFT') &
                           (chi_crime['community_area'] == 26)]
    e_garfield = chi_crime[(chi_crime['primary_type'] == 'THEFT') &
                           (chi_crime['community_area'] == 27)]
    uptown = chi_crime[(chi_crime['primary_type'] == 'THEFT') &
                       (chi_crime['community_area'] == 3)]
    percent_uptown = len(uptown) / total_thefts
    percent_garfield = (len(w_garfield) + len(e_garfield)) / total_thefts
    diff = percent_garfield - percent_uptown
    print("Given a call is about theft, probability that a call is from Uptown")
    print(percent_uptown)
    print("Given a call is about theft, probability that a call is from Garfield")
    print(percent_garfield)
    print("Probability difference")
    print(diff)
