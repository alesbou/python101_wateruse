# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 02:23:29 2021

@author: escriva
"""

#1. IMPORT PACKAGES
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import geopandas as gpd
from matplotlib import rcParams, ticker
from cycler import cycler


#2. READING DATA
data_water_use = pd.read_excel('uw_supplier_data072021.xlsx')
data_agencies = pd.read_csv('UrbanLatLon.csv')

#3. MANAGING DATA
#Change field/column names
data_water_use = data_water_use.rename(columns={
    'CALCULATED Total Potable Water Production Gallons (Ag Excluded)':'Water_Use'})
data_water_use = data_water_use.rename(columns={
    'CALCULATED Total Potable Water Production Gallons 2013 (Ag Excluded)':'Water_Use_2013'})
data_water_use = data_water_use.rename(columns={
    'CALCULATED R-GPCD':'gpcd'})

#Subset only one agency
data_EBMUD = data_water_use.loc[data_water_use['Supplier Name']==
                                data_water_use.iloc[0,0]]

#Subset only a few columns of the whole dataset
data_selected = data_water_use[['Supplier Name' , 'Reporting Month' ,
                                'Hydrologic Region', 'Total Population Served',
                                'Water_Use' , 'Water_Use_2013']]

#Subset based on conditions: Data for 2016
data_2016 = data_water_use.loc[data_water_use['Reporting Month']<'2017-01-01']
data_2016 = data_2016.loc[data_2016['Reporting Month']>'2015-12-31']
#or
data_2016_v02 = data_water_use.loc[(data_water_use['Reporting Month']<'2017-01-01') 
                                   & (data_water_use['Reporting Month']>'2015-12-31')]

#Grouping
data_2016_agency_sum = data_2016.groupby('Supplier Name').sum()
data_2016_HR = data_2016.groupby('Hydrologic Region').mean()
#To include index and have the field as a column you have to reset index (but
#first we double check that agencies are reporting all months)
data_2016['agency_reporting'] = 1
data_2016_agency_sum_v02 = data_2016.groupby('Supplier Name').sum().reset_index()

#Merging data
data_2016_agency_sum_latlon = data_2016_agency_sum_v02.merge(data_agencies, left_on='Supplier Name',
                                                  right_on = 'AgencyName')

#Rolling operations
data_EBMUD = data_EBMUD.sort_values('Reporting Month')
data_EBMUD_v02 = data_EBMUD.sort_values('Reporting Month').reset_index()
data_EBMUD_v02['Annual_Water_Use'] = data_EBMUD_v02.Water_Use.rolling(12).sum()

#Other operations
data_water_use['water_conservation'] = 100 * (1 - (data_water_use.Water_Use / data_water_use.Water_Use_2013))

#Functions and if sentences
def newhr(series):
    series.HR5=''
    if series['Hydrologic Region'] == 'Tulare Lake' or series['Hydrologic Region'] == 'San Joaquin River':
        series.HR5='SJ Valley'
    if series['Hydrologic Region'] == 'South Lahontan' or series['Hydrologic Region']=='Colorado River' or series['Hydrologic Region']=='South Coast':
        series.HR5='Southern California'
    if series['Hydrologic Region'] == 'Sacramento River' or series['Hydrologic Region']=='North Coast' or series['Hydrologic Region'] == 'North Lahontan':
        series.HR5='Northern California'
    if series['Hydrologic Region'] == 'Central Coast':
        series.HR5='Central Coast'
    if series['Hydrologic Region'] == 'San Francisco Bay':
        series.HR5='Bay Area'
    return series.HR5

data_water_use['HR5']=data_water_use.apply(newhr,axis=1)

#For loops
data_NorthCoast = data_water_use.loc[data_water_use['Hydrologic Region'] =='North Coast']
fig1 = plt.subplots()
for agency in np.unique(data_NorthCoast['Supplier Name']):
    df_agency = data_NorthCoast.loc[data_NorthCoast['Supplier Name'] == agency]
    plt.plot(df_agency['Reporting Month'], df_agency.Water_Use)
    
#Actually seaborn can do this in one line, but I wanted to show you the for loop
fig2 = plt.subplots()
sns.lineplot(x='Reporting Month' , y = 'Water_Use' , hue='Supplier Name' , data=data_NorthCoast)

#4. ANALYSIS
#We'll use the data_2016_agency_sum_latlon that has data for one year for all agencies, including some regressors
#First we take out data that does not have reporting for all months
data_analysis = data_2016_agency_sum_latlon.loc[data_2016_agency_sum_latlon.agency_reporting>11]
print(data_analysis.est_med_income.describe())
#Regression
y = data_analysis['gpcd']
X = data_analysis[[ 'Total Population Served', 'est_med_income', 'est_perc_u10', 'est_perc_o75']] 
X = sm.add_constant(X)
model = sm.OLS(y,X)
results = model.fit()
print(results.summary())

#5. VISUALIZATION
#Line plot of water use by HR5
data_water_use_hr5 = data_water_use.groupby(['HR5','Reporting Month']).sum().reset_index()
data_water_use_hr5['daysinmonth'] = data_water_use_hr5['Reporting Month'].dt.daysinmonth
data_water_use_hr5['gpcd_calc'] = data_water_use_hr5.Water_Use/(data_water_use_hr5.daysinmonth * 
                                                                data_water_use_hr5['Total Population Served'])

fig3 = plt.subplots(figsize=(7,5))
sns.lineplot(x = 'Reporting Month', y = 'gpcd_calc', hue = 'HR5', data = data_water_use_hr5)

#Map with residential per capita water use
#read shapefile
ca_outline = gpd.read_file('CA_State_TIGER2016.shp')
fig4 , ax4 = plt.subplots()
ca_outline.plot(ax = ax4)
#Create geodataframe from lat and long
water_use_cities = gpd.GeoDataFrame(data_2016_agency_sum_latlon, geometry=gpd.points_from_xy(
    data_2016_agency_sum_latlon.longitude,data_2016_agency_sum_latlon.latitude))
#To plot them together both projections have to be the same
water_use_cities = water_use_cities.set_crs("EPSG:4326")
ca_outline_newcrs = ca_outline.to_crs(water_use_cities.crs)
#Plotting
fig5, ax5 = plt.subplots(figsize = [7,5])
water_use_cities.plot(column = water_use_cities['gpcd'], markersize = 0.00001*water_use_cities['Total Population Served'] , ax = ax5 , alpha = 0.5 , cmap = 'Spectral_r', zorder=2 )
ca_outline_newcrs.plot(color='lightgrey', ax = ax5, zorder=1)
plt.legend()

#PPIC production ready
#General parameters for all images
plt.style.use('seaborn-whitegrid')

ppic_coltext = '#333333'
ppic_colgrid = '#898989'
ppic_colors = ['#e98426','#649ea5','#0d828a','#776972','#004a80','#3e7aa8','#b44b27','#905a78','#d2aa1d','#73a57a','#4fb3ce']


params = {
   'axes.labelsize': 9,
   'axes.labelweight': "bold",
   'font.size': 9,
   'legend.fontsize': 8,
   'xtick.labelsize': 8,
   'ytick.labelsize': 8,
   'text.usetex': False,
   'font.family': "Arial",
   'text.color' : ppic_coltext,
   'axes.labelcolor' : ppic_coltext,
   'xtick.color' : ppic_coltext,
   'ytick.color': ppic_coltext,
   'grid.color' : ppic_colgrid,
   'figure.figsize': [7, 5],
   'axes.prop_cycle' : cycler(color=ppic_colors)
   }
rcParams.update(params)

fig6 , ax6 = plt.subplots(figsize=(7,5))
ax6 = sns.lineplot(x = 'Reporting Month', y = 'gpcd_calc', hue = 'HR5', data = data_water_use_hr5)
plt.grid(b = False)
ax6.set_xlabel('Date')
ax6.set_ylabel('Per capita water use (gallons)')
plt.legend(loc='best')
plt.savefig('wateruse.pdf')


#6. EXPORT
#We already know how to export figures (see above)
#To export output dataframes
data_water_use_hr5.to_csv('data_water_use_hr5.csv')

