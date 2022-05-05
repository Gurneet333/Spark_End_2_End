# Databricks notebook source
# MAGIC %md
# MAGIC ## Exploritory Data Analysis on Bike Inventory Changes
# MAGIC [ref](https://nbviewer.jupyter.org/github/nicolasfauchereau/Auckland_Cycling/blob/master/notebooks/Auckland_cycling_and_weather.ipynb)

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

# MAGIC %run ./includes/utilities

# COMMAND ----------

!date

# COMMAND ----------

import logging
import warnings

import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from sklearn.metrics import mean_absolute_error as MAE
from scipy.stats import skew
from calendar import day_abbr, month_abbr, mdays
from fbprophet import Prophet

%matplotlib inline
logging.getLogger('fbprophet').setLevel(logging.ERROR)
logging.getLogger('py4j').setLevel(logging.ERROR)
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning, )
np.random.seed(42)

# COMMAND ----------

# Generate the hive metastore data for the bronze and silver data tables
drop_make('bronze_bike_trips',DELTA_TABLE_BIKE_TRIPS)
drop_make('bronze_nyc_weather',DELTA_TABLE_NYC_WEATHER)
drop_make('silver_modeling_data',DELTA_TABLE_MODELING_DATA)

# COMMAND ----------

# grab the station information (system wide)
stationDF=get_bike_stations()[['name','station_id','lat','lon']]
stationDF.head()

# COMMAND ----------

m = folium.Map(
    location=[40.7128, -74.0060],
    zoom_start=14,
    tiles='OpenStreetMap', 
    width='80%', 
)

m.add_child(folium.LatLngPopup())

marker_cluster = MarkerCluster().add_to(m)

for i, row in stationDF.iterrows():
    name = row['name']
    lat = row.lat
    lon = row.lon
    sid = row.station_id
    
    # HTML here in the pop up 
    popup = '<b>{}</b></br><i>station id = {}</i>'.format(name, sid)
    
    folium.Marker([lat, lon], popup=popup, tooltip=name).add_to(marker_cluster)

# COMMAND ----------

# Display the map of stations
m

# COMMAND ----------

STATION_NAME=stationDF[stationDF.station_id==f"{STATION_IDENTIFIER}"]['name'].values[0]
dfc=spark.read.format('delta').table("silver_modeling_data").toPandas()
dfc.head()

# COMMAND ----------

dfc.plot(x="time", y="inventory_change", title=f"{STATION_NAME} Inventory Change", figsize=(11,8))

# COMMAND ----------

dfc.index=dfc['time']
dfc[['inventory_change','inventory_change_filtered']].plot(figsize=(11,8))

# COMMAND ----------

dfc = dfc.dropna(subset=['inventory_change_filtered']).sort_index()
dfc[['inventory_change','inventory_change_filtered']].isnull().sum()

# COMMAND ----------

# DBTITLE 1,Profile a sample of the dataset
import pandas_profiling
from pandas_profiling.utils.cache import cache_file
profileDF = dfc.drop("time", axis=1).sample(1000)
displayHTML(pandas_profiling.ProfileReport(profileDF).html)

# COMMAND ----------

seas_cycl = dfc.loc[:,'inventory_change_filtered'].rolling(window=7*24, center=True, min_periods=4).mean().groupby(dfc.index.dayofyear).mean()
q25 = dfc.loc[:,'inventory_change_filtered'].rolling(window=7*24, center=True, min_periods=4).mean().groupby(dfc.index.dayofyear).quantile(0.25)
q75 = dfc.loc[:,'inventory_change_filtered'].rolling(window=7*24, center=True, min_periods=4).mean().groupby(dfc.index.dayofyear).quantile(0.75)

# COMMAND ----------

# DBTITLE 1,Plot the average inventory change over the season with IQR
f, ax = plt.subplots(figsize=(11,8)) 

seas_cycl.plot(ax=ax, lw=2, color='k', legend=False)

ax.fill_between(seas_cycl.index, q25.values.ravel(), q75.values.ravel(), color='0.8')

ax.set_xticks(np.cumsum(mdays))
#ax.set_xticklabels(month_abbr[1:])

ax.grid(ls=':')

ax.set_xlabel('', fontsize=15)

ax.set_ylabel('Average Change in Inventory\npositive indicates bikes accumulating\nnegative indicates bikes depleting', fontsize=15);

[l.set_fontsize(13) for l in ax.xaxis.get_ticklabels()]
[l.set_fontsize(13) for l in ax.yaxis.get_ticklabels()]

ax.set_title(f"Station: {STATION_NAME}\n7 days running average hourly inventory change", fontsize=15)

f.show()
#for ext in ['png','jpeg','pdf']: 
    #f.savefig(f'../figures/paper/seasonal_cycle.{ext}', dpi=200)

# COMMAND ----------

# DBTITLE 1,Plot the inventory change per day and over the week
hour_week = dfc.loc[:,['inventory_change_filtered']].copy()
hour_week.loc[:,'day_of_week'] = hour_week.index.dayofweek
hour_week.loc[:,'hour'] = hour_week.index.hour
hour_week = hour_week.groupby(['day_of_week','hour']).mean().unstack()
hour_week.columns = hour_week.columns.droplevel(0)

# COMMAND ----------

hour_week

# COMMAND ----------

f, ax = plt.subplots(figsize=(12,6))

sns.heatmap(hour_week, ax = ax, cmap=plt.cm.gray_r, vmax=5, cbar_kws={'boundaries':np.arange(-5,5,0.5)})

cbax = f.axes[1]
[l.set_fontsize(13) for l in cbax.yaxis.get_ticklabels()]
cbax.set_ylabel('Average Change in Inventory', fontsize=13)

[ax.axhline(x, ls=':', lw=0.5, color='0.8') for x in np.arange(1, 7)]
[ax.axvline(x, ls=':', lw=0.5, color='0.8') for x in np.arange(1, 24)];

ax.set_title(f"Station: {STATION_NAME}\nchange in inventory per day of week and hour of the day\ndark is accumulating\nlight is depleting", fontsize=16)

[l.set_fontsize(13) for l in ax.xaxis.get_ticklabels()]
[l.set_fontsize(13) for l in ax.yaxis.get_ticklabels()]

ax.set_xlabel('hour of the day', fontsize=15)
ax.set_ylabel('day of the week', fontsize=15)
ax.set_yticklabels(day_abbr[0:7]);

f.show()
#for ext in ['png','jpeg','pdf']: 
#    f.savefig(f'../figures/paper/cyclists_dayofweek_hourofday.{ext}', dpi=200)

# COMMAND ----------

# DBTITLE 1,Weekday vs. Weekend?
weekdays = dfc.loc[dfc.index.day_name().isin(['Monday','Tuesday','Wednesday','Thursday','Friday']), 'inventory_change_filtered']
weekends = dfc.loc[dfc.index.day_name().isin(['Sunday','Saturday']), 'inventory_change_filtered']
summary_hour_weekdays = weekdays.groupby(weekdays.index.hour).describe()
summary_hour_weekends = weekends.groupby(weekends.index.hour).describe()

# COMMAND ----------

f, ax = plt.subplots(figsize=(11,8))

ax.plot(summary_hour_weekends.index, summary_hour_weekends.loc[:,'mean'], color='k', label='week ends', ls='--', lw=3)

ax.fill_between(summary_hour_weekends.index, summary_hour_weekends.loc[:,'25%'], \
                summary_hour_weekends.loc[:,'75%'], hatch='///', facecolor='0.8', alpha=0.1)

ax.set_xticks(range(24));

ax.grid(ls=':', color='0.8')

# ax.set_title('week-ends', fontsize=16)

ax.plot(summary_hour_weekdays.index, summary_hour_weekdays.loc[:,'mean'], color='k', label='week days', lw=3)

ax.fill_between(summary_hour_weekdays.index, summary_hour_weekdays.loc[:,'25%'], \
                summary_hour_weekdays.loc[:,'75%'], hatch='\\\\\\', facecolor='0.8', alpha=0.1)

ax.legend(loc=1 , fontsize=15)

ax.set_xticks(range(24));

ax.grid(ls=':', color='0.8')

ax.set_ylim([-20, 20])

ax.set_xlabel('hour of the day', fontsize=15)

ax.set_ylabel('inventory change', fontsize=15);

[l.set_fontsize(13) for l in ax.xaxis.get_ticklabels()]
[l.set_fontsize(13) for l in ax.yaxis.get_ticklabels()]

ax.set_title(f"station: {STATION_NAME} inventory change per hour of the day", fontsize=16)

f.show()
#for ext in ['png','jpeg','pdf']: 
#    f.savefig(f'../figures/paper/daily_cycle.{ext}', dpi=200)

# COMMAND ----------

# DBTITLE 1,Daily inventory change
data = dfc.loc['2017':,['inventory_change_filtered']].resample('1D').sum()

# COMMAND ----------

f, ax = plt.subplots(figsize=(14,8))

data.plot(ax=ax, color='0.2')

data.rolling(window=7, center=True).mean().plot(ax=ax, ls='-', lw=3, color='0.6')

ax.grid(ls=':')
ax.legend(['daily values','7 days running average'], frameon=False, fontsize=14)

[l.set_fontsize(13) for l in ax.xaxis.get_ticklabels()]
[l.set_fontsize(13) for l in ax.yaxis.get_ticklabels()]

ax.set_xlabel('date', fontsize=15)

ax.set_ylabel('inventory change', fontsize=15);

ax.axvline('2017', color='0.8', lw=8, zorder=-1)

f.show()

#for ext in ['png','jpeg','pdf']: 
#    f.savefig(f'../figures/paper/cycling_counts_Tamaki_drive.{ext}', dpi=200)
