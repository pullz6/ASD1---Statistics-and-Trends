#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 08:57:02 2022

@author: Pulsara
"""

import pandas as pd 
import numpy as np 
from scipy import stats as scp
import matplotlib.pyplot as plt

def return_Dataframe_two(filename):
    df = pd.read_csv(filename)
    df_t = return_transposed_df(df)
    return df,df_t
    

def return_Dataframes(filename): 
    """This function is to provide dataframes with transposed....."""
    df = pd.read_csv(filename)
    #df = df.dropna(axis=0m inplace = False,)
    df_population = return_population(df)
    df_pop_growth = return_population_growth(df)
    df_Co2_liquid = return_Co2Emissions_liquid(df)
    df_Co2_solid = return_Co2Emissions_solid(df)
    df_lands = return_arable_lands(df)
    df_agri_lands = return_agricultural_lands(df)
    df_urban_pop = return_urban_pop(df)
    df_urban_growth = return_urban_growth(df)
    df_urban_percent = return_urban_percent(df)
    
    df_population_t = return_transposed_df(df_population)
    df_pop_growth_t = return_transposed_df(df_pop_growth)
    df_Co2_liquid_t = return_transposed_df(df_Co2_liquid)
    df_Co3_solid_t = return_transposed_df(df_Co2_solid)
    df_lands_t = return_transposed_df(df_lands)
    df_agri_lands_t = return_transposed_df(df_agri_lands)
    df_urban_pop_t = return_transposed_df(df_urban_pop)
    df_urban_growth_t = return_transposed_df(df_urban_growth)
    df_urban_percent_t = return_transposed_df(df_urban_percent)
    
    return df_population, df_pop_growth, df_Co2_liquid, df_Co2_solid, df_lands, df_agri_lands, df_population_t,df_pop_growth_t, df_Co2_liquid_t, df_Co3_solid_t, df_lands_t , df_agri_lands_t, df_urban_growth, df_urban_pop, df_urban_percent, df_urban_growth_t, df_urban_percent_t, df_urban_pop_t

def return_population(df_all): 
    df_pop = df_all[df_all['Indicator Name']=='Population, total']
    df_pop = df_pop.reset_index()
    return df_pop

def return_population_growth(df_all): 
    df_pop = df_all[df_all['Indicator Name']=='Population growth (annual %)']
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

def return_urban_pop(df_all): 
    df_pop = df_all[df_all['Indicator Name']=='Urban population']
    df_pop = df_pop.reset_index()
    return df_pop

def return_urban_growth(df_all): 
    df_pop = df_all[df_all['Indicator Name']=='Urban population growth (annual %)']
    df_pop = df_pop.reset_index()
    return df_pop

def return_urban_percent(df_all): 
    df_pop = df_all[df_all['Indicator Name']=='Urban population (% of total population)']
    df_pop = df_pop.reset_index()
    return df_pop

def return_transposed_df(df):
    df_t = pd.DataFrame.transpose(df)
    header = df_t.iloc[1].values.tolist()
    df_t.columns = header
    df_t = df_t.drop(df_t.index[0:6])
    df_t = df_t.reset_index()
    df_t = df_t.replace(np.nan,0)
    #df_pop = df_pop.dropna()
    df_t.columns.values[0] = "Year"
    return df_t

def countrywise_dataframe (country, df_pop_total, df_Co2_liq, df_Co2_solid, df_agri_land, df_urban_population): 
    print(country)       
    df = pd.DataFrame()
    if country == "China" : 
        df['Population'] = df_pop_total['China']
        df['CO2_Liquid'] = df_Co2_liq['China']
        df['CO2_Solid'] = df_Co2_solid['China']
        df['Agri_land'] = df_agri_land['China']
        df['Urban Population'] = df_urban_population['China']
        if country == "India" :
            df['Population'] = df_pop_total['India']
            df['CO2_Liquid'] = df_Co2_liq['India']
            df['CO2_Solid'] = df_Co2_solid['India']
            df['Agri_land'] = df_agri_land['India']
            df['Urban Population'] = df_urban_population['India']
            if country == "Saudi Arabia" : 
                df['Population'] = df_pop_total['Saudi Arabia']
                df['CO2_Liquid'] = df_Co2_liq['Saudi Arabia']
                df['CO2_Solid'] = df_Co2_solid['Saudi Arabia']
                df['Agri_land'] = df_agri_land['Saudi Arabia']
                df['Urban Population'] = df_urban_population['Saudi Arabia']
                if country == "United States" : 
                    df['Population'] = df_pop_total['United States']
                    df['CO2_Liquid'] = df_Co2_liq['United States']
                    df['CO2_Solid'] = df_Co2_solid['United States']
                    df['Agri_land'] = df_agri_land['United States']
                    df['Urban Population'] = df_urban_population['United States']
                    if country == "United Kingdom" : 
                        df['Population'] = df_pop_total['United Kingdom']
                        df['CO2_Liquid'] = df_Co2_liq['United Kingdom']
                        df['CO2_Solid'] = df_Co2_solid['United Kingdom']
                        df['Agri_land'] = df_agri_land['United Kingdom']
                        df['Urban Population'] = df_urban_population['United Kingdom']
    df = df[(df[['Population','CO2_Liquid','CO2_Solid','Agri_land','Urban Population']] != 0).all(axis=1)]
    return df

    

df = pd.read_csv('API_19_DS2_en_csv_v2_4700503.csv')
df_population, df_pop_growth, df_Co2_liquid, df_Co2_solid, df_lands, df_agri_lands, df_population_t,df_pop_growth_t, df_Co2_liquid_t, df_Co3_solid_t, df_lands_t , df_agri_lands_t, df_urban_growth, df_urban_pop, df_urban_percent, df_urban_growth_t, df_urban_percent_t, df_urban_pop_t  = return_Dataframes('API_19_DS2_en_csv_v2_4700503.csv')
df, df_t = return_Dataframe_two('API_19_DS2_en_csv_v2_4700503.csv')

print(df_t)
print(df_Co2_liquid_t)


#print(df_population_t)
# print(df_Co2_liquid_t.max())
# print(df_population_t.max())

# print(scp.pearsonr(df_population_t['Aruba'], df_agri_lands_t['Aruba']))

# print("Stat Values")
# print("Population VS Lands:")
# print("China")
# print(scp.pearsonr(df_pop_growth_t['China'],df_lands_t["China"]))
# print("India")
# print(scp.pearsonr(df_pop_growth_t['India'],df_lands_t["India"]))
# print("Saudi Arabia")
# print(scp.pearsonr(df_pop_growth_t['Saudi Arabia'],df_lands_t["Saudi Arabia"]))
# print("")
# print("Population VS Co emissions liquid:")
# print("China")
# print(scp.pearsonr(df_pop_growth_t['China'],df_Co2_liquid_t["China"]))
# print("India")
# print(scp.pearsonr(df_pop_growth_t['India'],df_Co2_liquid_t["India"]))
# print("Saudi Arabia")
# print(scp.pearsonr(df_pop_growth_t['Saudi Arabia'],df_Co2_liquid_t["Saudi Arabia"]))
# print("")
# print("Population VS Co emissions Solid:")
# print("China")
# print(scp.pearsonr(df_pop_growth_t['China'],df_Co3_solid_t["China"]))
# print("India")
# print(scp.pearsonr(df_pop_growth_t['India'],df_Co3_solid_t["India"]))
# print("Saudi Arabia")
# print(scp.pearsonr(df_pop_growth_t['Saudi Arabia'],df_Co3_solid_t["Saudi Arabia"]))
# print("")
# print("Population VS Agricultural lands:")
# print("China")
# print(scp.pearsonr(df_pop_growth_t['China'],df_agri_lands_t["China"]))
# print("India")
# print(scp.pearsonr(df_pop_growth_t['India'],df_agri_lands_t["India"]))
# print("Saudi Arabia")
# print(scp.pearsonr(df_pop_growth_t['Saudi Arabia'],df_agri_lands_t["Saudi Arabia"]))
# print("")
# print("Agricultural Lands VS Co-emissions liquid :")
# print("China")
# print(scp.pearsonr(df_Co2_liquid_t['China'],df_agri_lands_t["China"]))
# print("India")
# print(scp.pearsonr(df_Co2_liquid_t['India'],df_agri_lands_t["India"]))
# print("Saudi Arabia")
# print(scp.pearsonr(df_Co2_liquid_t['Saudi Arabia'],df_agri_lands_t["Saudi Arabia"]))
# print("")
# print("Agricultural Lands VS Co-emissions solid :")
# print("China")
# print(scp.pearsonr(df_Co3_solid_t['China'],df_agri_lands_t["China"]))
# print("India")
# print(scp.pearsonr(df_Co3_solid_t['India'],df_agri_lands_t["India"]))
# print("Saudi Arabia")
# print(scp.pearsonr(df_Co3_solid_t['Saudi Arabia'],df_agri_lands_t["Saudi Arabia"]))
# print("")
# print("Co-emissions liquid VS Co-emissions solid :")
# print("China")
# print(scp.pearsonr(df_Co3_solid_t['China'],df_Co2_liquid_t["China"]))
# print("India")
# print(scp.pearsonr(df_Co3_solid_t['India'],df_Co2_liquid_t["India"]))
# print("Saudi Arabia")
# print(scp.pearsonr(df_Co3_solid_t['Saudi Arabia'],df_Co2_liquid_t["Saudi Arabia"]))
# print("")



#Plotting a boxplot 
data = [df_pop_growth_t['United States'], df_pop_growth_t['China'], df_pop_growth_t['United Kingdom'], df_pop_growth_t['India'], df_pop_growth_t['Saudi Arabia'],df_pop_growth_t['World']]
#Including labels for the boxplot
labels_of_data = ["United States","China","United Kindgom","India", "Saudi Arabia", "World"]
plt.figure()
#Creating a boxplot for each countries population growth 
plt.boxplot(data, labels = labels_of_data)
#Including a title for the graph
plt.title("Annual Population Growth")
plt.ylabel("Growth (annual %)")
#Saving the boxplot as an image
plt.show()

#Plotting a line graph to see the coreelation between united states 
plt.figure()
years = pd.to_datetime(df_agri_lands_t['Year'])
#plt.plot(years, df_population_t['United States'])
plt.plot(years, df_agri_lands_t['United States'], color="blue")
#plt.plot(years, df_population_t['China'])
plt.plot(years, df_agri_lands_t['China'], color="red")
plt.plot(years, df_agri_lands_t['United Kingdom'], color="pink")
plt.plot(years, df_agri_lands_t['India'], color="green")
plt.plot(years, df_agri_lands_t['Saudi Arabia'], color="yellow")
#plt.xticks(df_population_t.iloc[:0])
plt.show()

#Dropping years 
#df_Co2_liquid_t = df_Co2_liquid_t.drop(df_Co2_liquid_t.index[0:31], inplace = True)

#Plotting the graph with co-emissions per year
years = pd.to_datetime(df_t['Year'])
years = pd.to_datetime(df_Co2_liquid_t['Year'])
plt.subplot(2, 3, 1)
plt.scatter(years, df_Co2_liquid_t['United States'], color = "blue")
plt.scatter(years, df_Co3_solid_t['United States'], color = "green")
plt.scatter(years, df_agri_lands_t['United States'], color = "red")

plt.subplot(2, 3, 2)
plt.scatter(years, df_Co2_liquid_t['China'], color = "blue")
plt.scatter(years, df_Co3_solid_t['China'], color = "green")
plt.scatter(years, df_agri_lands_t['China'], color = "red")

plt.subplot(2, 3, 3)
plt.scatter(years, df_Co2_liquid_t['Bangladesh'], color = "blue")
plt.scatter(years, df_Co3_solid_t['Bangladesh'], color = "green")
plt.scatter(years, df_agri_lands_t['Bangladesh'], color = "red")


plt.subplot(2, 3, 4)
plt.scatter(years, df_Co2_liquid_t['India'], color = "blue")
plt.scatter(years, df_Co3_solid_t['India'], color = "green")
plt.scatter(years, df_agri_lands_t['India'], color = "red")

plt.subplot(2, 3, 5)
plt.scatter(years, df_Co2_liquid_t['Saudi Arabia'], color = "blue")
plt.scatter(years, df_Co3_solid_t['Saudi Arabia'], color = "green")
plt.scatter(years, df_agri_lands_t['Saudi Arabia'], color = "red")

plt.subplot(2, 3, 6)
plt.scatter(years, df_Co2_liquid_t['World'], color = "blue")
plt.scatter(years, df_Co3_solid_t['World'], color = "green")
plt.scatter(years, df_agri_lands_t['Saudi Arabia'], color = "red")



plt.show()

df_china = countrywise_dataframe("China", df_population_t, df_Co2_liquid_t, df_Co3_solid_t, df_agri_lands_t, df_urban_pop_t)
print(df_china)
pear_corr=df_china.corr(method='pearson')
print(pear_corr)

fig, ax = plt.subplots(figsize=(8,8))
im = ax.imshow(pear_corr, interpolation='nearest')
fig.colorbar(im, orientation='vertical', fraction = 0.05)
# Show all ticks and label them with the dataframe column name
ax.set_xticklabels(df_china.columns, rotation=65, fontsize=15)
ax.set_yticklabels(df_china.columns, rotation=0, fontsize=15)

# Loop over data dimensions and create text annotations
for i in range(len(df_china.columns)):
    for j in range(len(df_china.columns)):
        text = ax.text(j, i, round(pear_corr.to_numpy()[i, j], 2),
                       ha="center", va="center", color="black")

plt.show()
