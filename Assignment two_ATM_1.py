#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 08:57:02 2022

@author: Pulsara
"""

import pandas as pd 
import numpy as np 
from scipy import stats as scp


def return_two_Dataframes(filename): 
    df = pd.read_csv(filename)
    df = df.dropna()
    df_population = return_population(df)
    df_Co2_liquid = return_Co2Emissions_liquid(df)
    df_Co2_solid = return_Co2Emissions_solid(df)
    df_lands = return_arable_lands(df)
    df_agri_lands = return_agricultural_lands(df)
    
    df_population_t = return_transposed_df(df_population)
    df_Co2_liquid_t = return_transposed_df(df_Co2_liquid)
    df_Co3_solid_t = return_transposed_df(df_Co2_solid)
    df_lands_t = return_transposed_df(df_lands)
    df_agri_lands_t = return_transposed_df(df_agri_lands)
    
    return df_population, df_Co2_liquid, df_Co2_solid, df_lands, df_population_t, df_Co2_liquid_t, df_Co3_solid_t, df_lands_t , df_agri_lands_t 

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

def return_transposed_df(df):
    df_t = pd.DataFrame.transpose(df)
    header = df_t.iloc[1].values.tolist()
    df_t.columns = header
    df_t = df_t.drop(df_t.index[0:6])
    return df_t

df = pd.read_csv('API_19_DS2_en_csv_v2_4700503.csv')
df_population, df_Co2_liquid, df_Co2_solid, df_lands, df_population_t, df_Co2_liquid_t, df_Co3_solid_t, df_lands_t , df_agri_lands_t = return_two_Dataframes('API_19_DS2_en_csv_v2_4700503.csv')
print(df_population)
print(df_population_t)

print(scp.pearsonr(df_population_t['Aruba'], df_population_t['South Africa']))
