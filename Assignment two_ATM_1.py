#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 08:57:02 2022

@author: Pulsara
#Importing modules

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt


#Defining the functions to be used

def return_Dataframe_two(filename):
    """"This function returns the twp dataframes from the entire dataset, one with years as columns and another with countries as columns"""
    #Reading the CSV file and adding it to a dataframe, this dataframe will have the years as columns
    df = pd.read_csv(filename)
    #Creating the transposed dataframe with countries as columns using a function, we are also passing the required parameters
    df_t = return_transposed_df(df,4,0)
    #Returning the dataframes
    return df,df_t
    
def return_dataframes(df, Indicator):
    """"This function returns the dataframe by filtering a specific indicator or a factor"""
    #If statement to filter the factor
    df_temp = df[df['Indicator Name']== Indicator]
    #Resetting our index
    df_temp = df_temp.reset_index()
    #Creating the transposed dataframe with countries as columns using a function, we are also passing the required parameters
    df_temp_t = return_transposed_df(df_temp,5,1)
    #Returning the dataframes
    return df_temp, df_temp_t


def return_transposed_df(df, i, j):
    """"This function returns the transposed dataframe of a passed dataframe along with index and header parameters"""
    #Transposing the passed datafram
    df_t = pd.DataFrame.transpose(df)
    #Selecting the header column
    header = df_t.iloc[j].values.tolist()
    #Including the header column
    df_t.columns = header
    #Dropping transposed rows that are not neccessary for our calculations
    df_t = df_t.drop(df_t.index[0:i])
    #Resetting our index
    df_t = df_t.reset_index()
    df_t = df_t.replace(np.nan,0)
    #Renaming the year column
    df_t.columns.values[0] = "Year"
    return df_t

def return_year_df(year, df):
    """"This function returns a list for values for a specific factor for particular year and for 5 countries"""
    #Create empty list
    year_list = []
    #Select all the values of that factor in one year
    df_temp = df[df["Year"]==year]
    #Include the values for the selected countries in the list
    year_list.append(df_temp.iloc[0]['United States'])
    year_list.append(df_temp.iloc[0]['China'])
    year_list.append(df_temp.iloc[0]['United Kingdom'])
    year_list.append(df_temp.iloc[0]['India'])
    year_list.append(df_temp.iloc[0]['Saudi Arabia'])
    #Return the list 
    return year_list

def return_mean_per_factor(df):
    """"This function returns a mean of every year in the dataset for a particular factor's dataframe"""
    #Create empty dataframe
    df_mean = pd.DataFrame()
    #Drop the columns that are not years
    df.drop(df.iloc[:, 0:5], inplace=True, axis=1)
    #Extract the mean from the .describe function's output
    df_mean = df.describe(include='all').loc['mean']
    #Return means
    return df_mean

def countrywise_dataframe (country, df_pop_total, df_Co2_liq, df_Co2_solid, df_agri_land, df_urban_population, df_Co2_total, df_forest_area, df_arable_lands): 
    """"This function returns a dataframe consisting of all the yearly values for each selected factor related to a particular country"""
    #Create empty dataframe       
    df = pd.DataFrame()
    #If statement to filter if the particular country and add it to the new dataframe under accurate column names. 
    if country == "China" : 
        df['Population'] = df_pop_total['China']
        df['CO2_Liquid'] = df_Co2_liq['China']
        df['CO2_Solid'] = df_Co2_solid['China']
        df['Agri_land'] = df_agri_land['China']
        df['Co2_Total'] = df_Co2_total['China']
        df['Urban Population'] = df_urban_population['China']
        df['Forest Area'] = df_forest_area['China']
        df['Arable Lands'] = df_arable_lands['China']
    if country == "India" :
        df['Population'] = df_pop_total['India']
        df['CO2_Liquid'] = df_Co2_liq['India']
        df['CO2_Solid'] = df_Co2_solid['India']
        df['Agri_land'] = df_agri_land['India']
        df['Urban Population'] = df_urban_population['India']
        df['Co2_Total'] = df_Co2_total['India']
        df['Forest Area'] = df_forest_area['India']
        df['Arable Lands'] = df_arable_lands['India']
    if country == "Saudi Arabia" : 
        df['Population'] = df_pop_total['Saudi Arabia']
        df['CO2_Liquid'] = df_Co2_liq['Saudi Arabia']
        df['CO2_Solid'] = df_Co2_solid['Saudi Arabia']
        df['Agri_land'] = df_agri_land['Saudi Arabia']
        df['Urban Population'] = df_urban_population['Saudi Arabia']
        df['Co2_Total'] = df_Co2_total['Saudi Arabia']
        df['Forest Area'] = df_forest_area['Saudi Arabia']
        df['Arable Lands'] = df_arable_lands['Saudi Arabia']
    if country == "United States" : 
        df['Population'] = df_pop_total['United States']
        df['CO2_Liquid'] = df_Co2_liq['United States']
        df['CO2_Solid'] = df_Co2_solid['United States']
        df['Agri_land'] = df_agri_land['United States']
        df['Urban Population'] = df_urban_population['United States']
        df['Co2_Total'] = df_Co2_total['United States']
        df['Forest Area'] = df_forest_area['United States']
        df['Arable Lands'] = df_arable_lands['United States']
    if country == "United Kingdom" : 
        df['Population'] = df_pop_total['United Kingdom']
        df['CO2_Liquid'] = df_Co2_liq['United Kingdom']
        df['CO2_Solid'] = df_Co2_solid['United Kingdom']
        df['Agri_land'] = df_agri_land['United Kingdom']
        df['Urban Population'] = df_urban_population['United Kingdom']
        df['Co2_Total'] = df_Co2_total['United Kingdom']
        df['Forest Area'] = df_forest_area['United Kingdom']
        df['Arable Lands'] = df_arable_lands['United Kingdom']
    #Delete any zero values in each of the columns
    df = df[(df[['Population','CO2_Liquid','CO2_Solid','Agri_land','Urban Population']] != 0).all(axis=1)]
    #Return the dataframe
    return df

#Main Programme

#Calling our function to get the a dataframe with years as columns and a dataframe with countries as columns
df, df_t = return_Dataframe_two('API_19_DS2_en_csv_v2_4700503.csv')

#Calling the function to get two dataframes for particular factor or climate indicator
df_population, df_population_t = return_dataframes(df, "Population, total")
df_pop_growth, df_pop_growth_t = return_dataframes(df, "Population growth (annual %)") 
df_Co2_liquid, df_Co2_liquid_t = return_dataframes(df, "CO2 emissions from liquid fuel consumption (kt)")
df_Co2_solid, df_Co2_solid_t = return_dataframes(df, "CO2 emissions from solid fuel consumption (kt)")
df_Co2_total, df_Co2_total_t = return_dataframes(df, "CO2 emissions (kt)")
df_Co2_gas, df_Co2_fas_t = return_dataframes(df,"CO2 emissions from gaseous fuel consumption (kt)")
df_ara_lands, df_ara_lands_t = return_dataframes(df, "Arable land (% of land area)")
df_agri_lands, df_agri_lands_t = return_dataframes(df, "Agricultural land (sq. km)")
df_urban_growth, df_urban_growth_t = return_dataframes(df, "Urban population growth (annual %)")
df_urban_pop, df_urban_pop_t = return_dataframes(df, "Urban population")
df_urban_percent, df_urban_percent_t = return_dataframes(df, "Urban population (% of total population)")
df_forest_area, df_forest_area_t = return_dataframes(df, "Forest area (sq. km)") 

#==============================================================================
#Plotting the histogram for Total Co2 Emissions over the year
# set width of bars
barWidth = 0.25
 
#Calling a function to return data for each year for our selected countries 
year_1990 = return_year_df("1990", df_Co2_total_t)
year_2010 = return_year_df("2010", df_Co2_total_t)
year_2019 = return_year_df("2019", df_Co2_total_t)
 
# Set position of bar on X axis
r1 = np.arange(len(year_1990))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
 
# Make the plot
plt.figure(figsize=(8, 8))
plt.bar(r1, year_1990, color='#7f6d5f', width=barWidth, edgecolor='white', label='1990')
plt.bar(r2, year_2010, color='#557f2d', width=barWidth, edgecolor='white', label='2010')
plt.bar(r3, year_2019, color='#2d7f5e', width=barWidth, edgecolor='white', label='2019')
 
#Add title and xticks on the middle of the group bars
plt.title("Total Co2 Emissions for Countries over years",fontweight='bold')
plt.xlabel('Countries', fontweight='bold')
plt.ylabel('Total Co2 Emission (kt)', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(year_1990))], ["United States","China","United Kingdom","India","Saudi Arabia"])
 
#Create legend
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
#Get current figure
figure = plt.gcf()
#Set the size to 800 and 600 for easy inclusion for the report
figure.set_size_inches(8, 6)
plt.savefig("Total Co2 Emissions for Countries over years.png", format='png', dpi=100)
plt.show()
#==============================================================================

#Creating a heatmap for countries

#CHINA
#Calling our function to get a all factors for China into one dataframe
df_china = countrywise_dataframe("China", df_population_t, df_Co2_liquid_t, df_Co2_solid_t, df_agri_lands_t, df_urban_pop_t, df_Co2_total_t, df_forest_area_t, df_ara_lands_t)

#Creating a labels list
labels = ["","Population","Co2 Liquid","Co2 Solid","Agricultural Lands","Urban Population","Co2 Total","Forest Area","Arable Lands"]
pear_corr=df_china.corr(method='pearson')
fig, ax = plt.subplots(figsize=(8,8))
im = ax.imshow(pear_corr, interpolation='nearest')
fig.colorbar(im, orientation='vertical', fraction = 0.05)
# Show all ticks and label them with the dataframe column name
ax.set_xticklabels(labels, rotation=90, fontsize=10)
ax.set_yticklabels(labels, rotation=0, fontsize=10)

# Loop over data dimensions and create text annotations
for i in range(len(df_china.columns)):
    for j in range(len(df_china.columns)):
        text = ax.text(j, i, round(pear_corr.to_numpy()[i, j], 2),
                        ha="center", va="center", color="black")
        
plt.title("China", fontweight='bold')
#Get current figure
figure = plt.gcf()
#Set the size to 800 and 600 for easy inclusion for the report
figure.set_size_inches(8, 6)
plt.savefig("China.png", format='png', dpi=100)
plt.show()

#UNITED STATES
#Calling our function to get a all factors for United States into one dataframe
df_united_states = countrywise_dataframe("United States", df_population_t, df_Co2_liquid_t, df_Co2_solid_t, df_agri_lands_t, df_urban_pop_t, df_Co2_total_t, df_forest_area_t, df_ara_lands_t)

#Creating a labels list
labels = ["","Population","Co2 Liquid","Co2 Solid","Agricultural Lands","Urban Population","Co2 Total","Forest Area","Arable Lands"]
#Caculating the pearson correlation for all the factors
pear_corr=df_united_states.corr(method='pearson')
#Creating a figure
fig, ax = plt.subplots(figsize=(8,8))
#Creating the heatmap
im = ax.imshow(pear_corr, interpolation='nearest', cmap="PuBu")
fig.colorbar(im, orientation='vertical', fraction = 0.05)
# Show all ticks and label them with the dataframe column name
ax.set_xticklabels(labels, rotation=90, fontsize=10)
ax.set_yticklabels(labels, rotation=0, fontsize=10)

#Loop over data dimensions and create text annotations
for i in range(len(df_united_states.columns)):
    for j in range(len(df_united_states.columns)):
        text = ax.text(j, i, round(pear_corr.to_numpy()[i, j], 2),
                        ha="center", va="center", color="black")
        
plt.title("United States", fontweight='bold')
#Get current figure
figure = plt.gcf()
#Set the size to 800 and 600 for easy inclusion for the report
figure.set_size_inches(8, 6)
plt.savefig("United States.png", format='png', dpi=100)
plt.show()

#==============================================================================

#Plotting a line graph to view the agriculural lands over the years 
plt.figure()
#Getting the x value from the years converting it to datetime
years = pd.to_datetime(df_agri_lands_t['Year'])
#Plotting the agricultural lands for each of the 5 countries by calling our function
plt.plot(years, df_agri_lands_t['United States'], color="blue", label = "United States")
plt.plot(years, df_agri_lands_t['China'], color="green", label = "China")
plt.plot(years, df_agri_lands_t['United Kingdom'], color="pink" , label = "United Kingdom")
plt.plot(years, df_agri_lands_t['India'], color="red", label = "India")
plt.plot(years, df_agri_lands_t['Saudi Arabia'], color="yellow", label = "Saudi Arabia")
#Name the figure and axis
plt.title("Agricultural Lands over the years", fontweight='bold')
plt.ylabel("Agricultural Lands (square meters)", fontweight='bold')
plt.xlabel("Years", fontweight='bold')
plt.xlim([years[30], years[60]])
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
figure = plt.gcf()
plt.savefig("Agricultural Lands over the years.png", format='png', dpi=100)
plt.show()
#==============================================================================

#Plotting a line graph to view the forest lands over the years  
plt.figure()
#Getting the x value from the years converting it to datetime
years = pd.to_datetime(df_forest_area_t['Year'])
#Plotting the forest lands for each of the 5 countries by calling our function
plt.plot(years, df_forest_area_t['United States'], color="blue", label = "United States")
plt.plot(years, df_forest_area_t['China'], color="green", label = "China")
plt.plot(years, df_forest_area_t['United Kingdom'], color="purple", label = "United Kingdom")
plt.plot(years, df_forest_area_t['India'], color="red", label = "India")
plt.plot(years, df_forest_area_t['Saudi Arabia'], color="yellow", label = "Saudi Arabia", alpha=0.3)
#Name the figure and axis
plt.title("Forest Area over the years",fontweight='bold')
plt.xlabel("Years", fontweight='bold')
plt.ylabel("Number of Forest area (Square Meters)", fontweight='bold')
plt.xlim([years[30], years[60]])
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
figure = plt.gcf()
plt.savefig("Forest Lands over the years.png", format='png', dpi=100)
plt.show()
#==============================================================================
#Plotting a boxplot 
data = [df_urban_growth_t['United States'], df_urban_growth_t['China'], df_urban_growth_t['United Kingdom'], df_urban_growth_t['India'], df_urban_growth_t['Saudi Arabia']]
#Including labels for the boxplot
labels_of_data = ["United States","China","United Kindgom","India", "Saudi Arabia"]
plt.figure()
#Creating a boxplot for each countries population growth 
box=plt.boxplot(data, labels = labels_of_data, patch_artist=True)
#Adding colors into the boxplot
colors = ['blue', 'lime', 'violet', 'tomato', 'gold']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
#Name the figure and axis
plt.title("Annual Urban population Growth",fontweight='bold')
plt.ylabel("Urban Growth (annual %)",fontweight='bold')
plt.xlabel("Countries",fontweight='bold')
plt.savefig("Annual Urban population growth.png", format='png')
plt.show()
#==============================================================================
#Plotting a scatter graph for the mean of all factors over a time period  by calling our function. 
Co2_emissions = return_mean_per_factor(df_Co2_total)
Co2_Solid = return_mean_per_factor(df_Co2_solid)
Co2_liquid = return_mean_per_factor(df_Co2_liquid)
agricultural_lands = return_mean_per_factor(df_agri_lands)
population = return_mean_per_factor(df_population)
forest_area = return_mean_per_factor(df_forest_area)

plt.figure()
#Plotting the mean for each factor. 
plt.plot(years, Co2_emissions, marker = 'p', label='Mean of Co2 Emissions')
plt.plot(years, forest_area, marker = 'X', label='Mean of forest area')
plt.plot(years, agricultural_lands, marker = 'D', label='Mean of agricultural lands')
plt.plot(years, Co2_Solid, marker = 'D', label='Mean of Co2 Solid')
plt.plot(years, Co2_liquid, marker = 's', label='Mean of Co2 liquid')
plt.xlim([years[30], years[60]])
plt.title("Mean of Factors over the years",fontweight='bold')
plt.ylabel("Mean",fontweight='bold')
plt.xlabel("Year",fontweight='bold')
plt.savefig("ean of Factors over the years.png", format='png')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.show()
#==============================================================================

