#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 08:57:02 2022

@author: Pulsara
"""

import pandas as pd 
import numpy as np 


def return_two_Dataframes(filename): 
    df = pd.read_csv(filename, skiprows=(1500))
    df_t = pd.DataFrame.transpose(df)
    header = df_t.iloc[0].values.tolist()
    df_t.columns = header
    return df_t

print(return_two_Dataframes('API_19_DS2_en_csv_v2_4700503.csv'))