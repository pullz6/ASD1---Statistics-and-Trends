#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 08:57:02 2022

@author: Pulsara
"""

import pandas as pd 
import numpy as np 


def return_two_Dataframes(filename): 
    df = pd.read_csv(filename)
    df_t = pd.DataFrame.transpose(df)
    header = df_t.iloc[0].values.tolist()
    df_t.columns = header
    return df_t 

def return_population(df_all): 
    df_pop = df_all[df_all['Indicator Name']=='Population, total']
    df_pop = df_pop.reset_index()
    return df_pop

def return_Co2Emissions_liquid(df_all): 
    df_pop = df_all[df_all['Indicator Name']=='CO2 emissions from liquid fuel consumption (kt)']
    df_pop = df_pop.reset_index()
    return df_pop

def return_Co2Emissions_solid(df_all): 
    df_pop = df_all[df_all['Indicator Name']=='CO2 emissions from solid fuel consumption (kt)']
    df_pop = df_pop.reset_index()
    return df_pop

def return_arable_lands(df_all): 
    df_pop = df_all[df_all['Indicator Name']=='Arable land (% of land area)']
    df_pop = df_pop.reset_index()
    return df_pop

def return_agricultural_lands(df_all): 
    df_pop = df_all[df_all['Indicator Name']=='Agricultural land (sq. km)']
    df_pop = df_pop.reset_index()
    return df_pop



df = pd.read_csv('API_19_DS2_en_csv_v2_4700503.csv')
print(return_Co2Emissions_liquid(df))
print(return_Co2Emissions_solid(df))
print(return_arable_lands(df))
print(return_agricultural_lands(df))
