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
    df_t = pd.DataFrame.transpose(df)
    header = df_t.iloc[j].values.tolist()
    df_t.columns = header
    df_t = df_t.drop(df_t.index[0:i])
    df_t = df_t.reset_index()
    df_t = df_t.replace(np.nan,0)
    df_t.columns.values[0] = "Year"
    return df_t

def return_year_Co2(year, df):
    year_list = []
    df_temp = df[df["Year"]==year]
    year_list.append(df_temp.iloc[0]['United States'])
    year_list.append(df_temp.iloc[0]['China'])
    year_list.append(df_temp.iloc[0]['United Kingdom'])
    year_list.append(df_temp.iloc[0]['India'])
    year_list.append(df_temp.iloc[0]['Saudi Arabia'])
    return year_list

def return_mean_per_factor(df):
    df_mean = pd.DataFrame()
    df.drop(df.iloc[:, 0:5], inplace=True, axis=1)
    df_mean = df.describe(include='all').loc['mean']
    return df_mean

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

#Main Programme to ru
df, df_t = return_Dataframe_two('API_19_DS2_en_csv_v2_4700503.csv')

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

#Plotting the histogram 
# set width of bars
barWidth = 0.25
 
# set heights of bars
year_1990 = return_year_Co2("1990", df_Co2_total_t)
year_2000 = return_year_Co2("2000", df_Co2_total_t)
year_2010 = return_year_Co2("2010", df_Co2_total_t)
 
# Set position of bar on X axis
r1 = np.arange(len(year_1990))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
 
# Make the plot
plt.bar(r1, year_1990, color='#7f6d5f', width=barWidth, edgecolor='white', label='1990')
plt.bar(r2, year_2000, color='#557f2d', width=barWidth, edgecolor='white', label='2000')
plt.bar(r3, year_2010, color='#2d7f5e', width=barWidth, edgecolor='white', label='2010')
 
# Add xticks on the middle of the group bars
plt.xlabel('', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(year_1990))], ["United States","China","United Kingdom","India","Saudi Arabia"])
 
# Create legend & Show graphic
plt.legend()
plt.show()

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
plt.plot(years, df_agri_lands_t['United States'], color="blue")
plt.plot(years, df_agri_lands_t['China'], color="red")
plt.plot(years, df_agri_lands_t['United Kingdom'], color="pink")
plt.plot(years, df_agri_lands_t['India'], color="green")
plt.plot(years, df_agri_lands_t['Saudi Arabia'], color="yellow")
plt.title("Agricultural Lands over the years")
plt.ylabel("Number of lands")
plt.xlim([years[0], years[61]])
plt.show()

#Plotting a line graph to see the coreelation between united states 
plt.figure()
years = pd.to_datetime(df_ara_lands_t['Year'])
#plt.plot(years, df_population_t['United States'])
plt.plot(years, df_ara_lands_t['United States'], color="blue")
#plt.plot(years, df_population_t['China'])
plt.plot(years, df_ara_lands_t['China'], color="red")
plt.plot(years, df_ara_lands_t['United Kingdom'], color="pink")
plt.plot(years, df_ara_lands_t['India'], color="green")
plt.plot(years, df_ara_lands_t['Saudi Arabia'], color="yellow")
plt.title("Arable Lands over the years")
plt.ylabel("Number of lands")
plt.xlim([years[0], years[61]])
plt.show()

#Dropping years 
#df_Co2_liquid_t = df_Co2_liquid_t.drop(df_Co2_liquid_t.index[0:31], inplace = True)

#Plotting the graph with co-emissions per year
years = pd.to_datetime(df_t['Year'])
plt.figure(figsize=(19, 10))
#years = pd.to_datetime(df_Co2_liquid_t['Year'])
plt.subplot(2, 3, 1)
plt.scatter(years, df_Co2_liquid_t['United States'], color = "blue")
plt.scatter(years, df_Co2_solid_t['United States'], color = "green")
plt.scatter(years, df_Co2_total_t['United States'], color = "black")
plt.scatter(years, df_agri_lands_t['United States'], color = "red")

plt.subplot(2, 3, 2)
plt.scatter(years, df_Co2_liquid_t['China'], color = "blue")
plt.scatter(years, df_Co2_solid_t['China'], color = "green")
plt.scatter(years, df_Co2_total_t['China'], color = "black")
plt.scatter(years, df_agri_lands_t['China'], color = "red")

plt.subplot(2, 3, 3)
plt.scatter(years, df_Co2_liquid_t['Bangladesh'], color = "blue")
plt.scatter(years, df_Co2_solid_t['Bangladesh'], color = "green")
plt.scatter(years, df_agri_lands_t['Bangladesh'], color = "red")


plt.subplot(2, 3, 4)
plt.scatter(years, df_Co2_liquid_t['India'], color = "blue")
plt.scatter(years, df_Co2_solid_t['India'], color = "green")
plt.scatter(years, df_agri_lands_t['India'], color = "red")

plt.subplot(2, 3, 5)
plt.scatter(years, df_Co2_liquid_t['Saudi Arabia'], color = "blue")
plt.scatter(years, df_Co2_solid_t['Saudi Arabia'], color = "green")
plt.scatter(years, df_agri_lands_t['Saudi Arabia'], color = "red")

plt.subplot(2, 3, 6)
plt.scatter(years, df_Co2_liquid_t['World'], color = "blue")
plt.scatter(years, df_Co2_solid_t['World'], color = "green")
plt.scatter(years, df_agri_lands_t['World'], color = "red")

plt.show()

#Drawing a violin plot
plt.figure()
plt.figure(figsize=(19, 10))
plt.subplot(2, 3, 1)
plt.violinplot(df_Co2_liquid_t['United States'])
plt.violinplot(df_Co2_solid_t['United States'])
plt.violinplot(df_Co2_total_t['United States'])

plt.subplot(2, 3, 2)
plt.violinplot(df_Co2_liquid_t['China'])
plt.violinplot(df_Co2_solid_t['China'])
plt.violinplot(df_Co2_total_t['China'])

plt.subplot(2, 3, 3)
plt.violinplot(df_Co2_liquid_t['United Kingdom'])
plt.violinplot(df_Co2_solid_t['United Kingdom'])
plt.violinplot(df_Co2_total_t['United Kingdom'])


plt.subplot(2, 3, 4)
plt.violinplot(df_Co2_liquid_t['India'])
plt.violinplot(df_Co2_solid_t['India'])
plt.violinplot(df_Co2_total_t['India'])

plt.subplot(2, 3, 5)
plt.violinplot(df_Co2_liquid_t['Saudi Arabia'])
plt.violinplot(df_Co2_solid_t['Saudi Arabia'])
plt.violinplot(df_Co2_total_t['Saudi Arabia'])

plt.subplot(2, 3, 6)
plt.violinplot(df_Co2_liquid_t['World'])
plt.violinplot(df_Co2_solid_t['World'])
plt.violinplot(df_Co2_total_t['World'])
plt.show()



#creating a heatmap
df_china = countrywise_dataframe("China", df_population_t, df_Co2_liquid_t, df_Co2_solid_t, df_agri_lands_t, df_urban_pop_t)
labels = ["","Population","Co2 Liquid","Co2 Solid","Agricultural Lands","Urban Population"]
#print(df_china)
pear_corr=df_china.corr(method='pearson')

fig, ax = plt.subplots(figsize=(8,8))
im = ax.imshow(pear_corr, interpolation='nearest')
fig.colorbar(im, orientation='vertical', fraction = 0.05)
# Show all ticks and label them with the dataframe column name
ax.set_xticklabels(labels, rotation=65, fontsize=15)
ax.set_yticklabels(labels, rotation=0, fontsize=15)

# Loop over data dimensions and create text annotations
for i in range(len(df_china.columns)):
    for j in range(len(df_china.columns)):
        text = ax.text(j, i, round(pear_corr.to_numpy()[i, j], 2),
                        ha="center", va="center", color="black")

plt.show()

#Plotting a scatter graph for the mean of all factors over a time period 
Co2_emissions = return_mean_per_factor(df_Co2_total)
arable_lands = return_mean_per_factor(df_ara_lands)
agricultural_lands = return_mean_per_factor(df_agri_lands)
population = return_mean_per_factor(df_population)
urban_population = return_mean_per_factor(df_urban_pop)
forest_area = return_mean_per_factor(df_forest_area)


plt.figure()
plt.plot(years, Co2_emissions, 'ro', label='Mean of Co2 Emissions')
plt.plot(years, forest_area, 'bx', label='Mean of forest area')
plt.legend()
plt.show()

