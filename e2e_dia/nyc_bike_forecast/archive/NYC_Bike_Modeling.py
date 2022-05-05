# Databricks notebook source
# MAGIC %md
# MAGIC ## Modeling Bike Inventory Changes
# MAGIC [ref](https://nbviewer.jupyter.org/github/nicolasfauchereau/Auckland_Cycling/blob/master/notebooks/Auckland_cycling_and_weather.ipynb)

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

# MAGIC %run ./includes/utilities

# COMMAND ----------

!date

# COMMAND ----------

import logging
logging.getLogger('fbprophet').setLevel(logging.ERROR)
logging.getLogger('py4j').setLevel(logging.ERROR)

# COMMAND ----------

import warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning, )

# COMMAND ----------

# MAGIC %matplotlib inline

# COMMAND ----------

import os
import sys
from glob import glob
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error as MAE
from scipy.stats import skew
from calendar import day_abbr, month_abbr, mdays
from fbprophet import Prophet

np.random.seed(42)

# COMMAND ----------

dfc=spark.read.format('delta').table("silver_modeling_data").toPandas()
dfc.index = dfc['time']
dfc.tail()

# COMMAND ----------

# DBTITLE 1,Preparing training set from the silver data source
# Grab the data from 2017 onward
data = dfc.loc['2017':,['inventory_change_filtered']].rename({'inventory_change_filtered':'y'}, axis=1)
data.index.rename("ds",inplace=True)
data.head()

# COMMAND ----------

data.plot(figsize=(11,8))

# COMMAND ----------

?prepare_data

# COMMAND ----------

# Create a training set 2017 & 2018
# Create a test set 2019 
data_train, data_test = prepare_data(data, 2019)
# remove the nan values
data_train.dropna(inplace=True)
data_test.dropna(inplace=True)
print(data_train.isna().sum(),data_test.isna().sum())

# COMMAND ----------

data_train.tail()

# COMMAND ----------

data_test.tail()

# COMMAND ----------

# DBTITLE 1,Instantiate, then fit the model to the training data
# MAGIC %md
# MAGIC The first step in fbprophet is to instantiate the model, it is there that you can set the prior scales for each component of your time-series, as well as the number of Fourier series to use to model the cyclic components.
# MAGIC 
# MAGIC A general rule is that larger prior scales and larger number of Fourier series will make the model more flexible, but at the potential cost of generalisation: i.e. the model might overfit, learning the noise (rather than the signal) in the training data, but giving poor results when applied to yet unseen data (the test data)... setting these hyperparameters) can be more an art than a science ...

# COMMAND ----------

import holidays

holidays_us = pd.DataFrame([], columns = ['ds','holiday'])

ldates = []
lnames = []
for date, name in sorted(holidays.US(state='NY', years=np.arange(2017, 2022)).items()):
    ldates.append(date)
    lnames.append(name)
    
ldates = np.array(ldates)
lnames = np.array(lnames)

holidays_us.loc[:,'ds'] = ldates
holidays_us.loc[:,'holiday'] = lnames
holidays_us.loc[:,'holiday'] = holidays_us.loc[:,'holiday'].apply(lambda x : x.replace(' (Observed)',''))
holidays_us.head()

# COMMAND ----------

holidays_us.tail()

# COMMAND ----------

model = Prophet(holidays=holidays_us, holidays_prior_scale=0.25, seasonality_prior_scale=10.0, changepoint_prior_scale=0.01, seasonality_mode='multiplicative', yearly_seasonality=10, \
            weekly_seasonality=True, \
            daily_seasonality=True)
#model = Prophet(holidays=holidays_us, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)


# COMMAND ----------

model.fit(data_train)

# COMMAND ----------

# create a "future" dataframe that picks up where the training set left off and generates forecasts for a number of periods and grain
future = model.make_future_dataframe(periods=365*24, freq='1H')
future.tail()

# COMMAND ----------

forecast = model.predict(future)
forecast.head()

# COMMAND ----------

forecast.tail()

# COMMAND ----------

f = model.plot_components(forecast)

# COMMAND ----------

verif = make_verif(forecast, data_train, data_test)
verif.tail()

# COMMAND ----------

f = plot_verif(verif, year=2019)

# COMMAND ----------

plot_joint_plot(verif.loc[:'2018',:].dropna(), title="training data results")

# COMMAND ----------

plot_joint_plot(verif.loc['2019':,:].dropna(), title="test data results")

# COMMAND ----------


