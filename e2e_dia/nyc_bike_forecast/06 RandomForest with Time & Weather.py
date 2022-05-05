# Databricks notebook source
# MAGIC %md ##Forecasting Using Random Forests with Temperature & Weather Features
# MAGIC In this notebook, we will build regression models to forecast rentals using some basic weather data. As before, we will start by installing the libraries needed for this work.  Notice we're installing a relatively new version of SciKit-Learn to gain access to some expanded functionality for the RandomForestRegressor:

# COMMAND ----------

spark.conf.set("spark.sql.shuffle.partitions", "32")  # Configure the size of shuffles the same as core count
spark.conf.set("spark.sql.adaptive.enabled", "true")  # Spark 3.0 AQE - coalescing post-shuffle partitions, converting sort-merge join to broadcast join, and skew join optimization
dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')

# COMMAND ----------

from datetime import datetime as dt
from datetime import timedelta

dbutils.widgets.removeAll()

dbutils.widgets.text('01.start_date', "2018-01-01")
dbutils.widgets.text('02.end_date', "2020-01-01")

start_date = str(dbutils.widgets.get('01.start_date'))
end_date = str(dbutils.widgets.get('02.end_date'))
print(start_date,end_date)

# COMMAND ----------

# MAGIC %md Now we define our function.  

# COMMAND ----------

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

from sklearn.metrics import mean_absolute_error

import shutil

from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor

# structure of the dataset returned by the function
result_schema =StructType([
  StructField('model_version',StringType()),
  StructField('station_id',IntegerType()),
  StructField('ds',TimestampType()),
  StructField('y', FloatType()),
  StructField('yhat', FloatType()),
  StructField('yhat_lower', FloatType()),
  StructField('yhat_upper', FloatType())
  ])

# training function
#@pandas_udf( result_schema, PandasUDFType.GROUPED_MAP )
def train_model(group_pd):
  
  # DATA PREP
  # ---------------------------------
  # identify station id and hours to forecast
  station_id = group_pd['station_id'].iloc[0]
 
  # fill records (rows) with nan values
  data_pd = group_pd.fillna(method='ffill').fillna(method='bfill')
  
  # separate features and labels
  X = data_pd.drop(['station_id','y','ds'], axis=1).values
  y = data_pd[['y']].values
  # ---------------------------------  
  
  with mlflow.start_run(run_name="citibike-randomforest-with-regressors-{0}".format(station_id)) as run:

    # TRAIN MODEL
    # ---------------------------------  
    # train model
    model = RandomForestRegressor( max_samples=0.60 )
    model.fit(X, y)

    # capture the model signature
    signature = infer_signature(data_pd.drop(['station_id','y','ds'], axis=1), data_pd[['y']])
    
    # score the model
    mae = mean_absolute_error(y, model.predict(X))

    # save models for potential later use including the signature
    mlflow.sklearn.log_model(model, 
                             artifact_path="{0}_rf_model".format(station_id),
                             registered_model_name="{}-reg-rf-model".format(station_id),
                             signature = signature
                            )
    
    # tag model with training range
    tags = {"start_of_training": start_date, "end_of_training": end_date}
    mlflow.set_tags(tags)
    
    # Log params
    mlflow.log_params({'max_samples':0.60})
    
    # log performance
    mlflow.log_metric("mae", mae)
    
    # move this latest version of the model to the Staging if there is a production version
    # else register it as the production version
    client = MlflowClient()
    model_name = "{}-reg-rf-model".format(station_id)
    model_version = dict(client.search_model_versions(f"name='{model_name}'")[0])['version']
    
    model_stage = "Production"
    for mv in client.search_model_versions(f"name='{model_name}'"):
      if dict(mv)['current_stage'] == 'Staging':
          # Archive the currently staged model
          client.transition_model_version_stage(
              name=dict(mv)['name'],
              version=dict(mv)['version'],
              stage="Archived"
          )
          model_stage = "Staging"
      elif dict(mv)['current_stage'] == 'Production':
          model_stage = "Staging"
    # move the model to the appropriate stage.
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage=model_stage
    )
    
  # ---------------------------------
  # Performance
  # ---------------------------------  
  # generate performance
  yhat = model.predict(X)
  yhat_lower, yhat_upper = pred_ints(model, X, interval_width=0.80)
     
  preds_np = np.concatenate(
    (
      yhat.reshape(-1,1), 
      yhat_lower.reshape(-1,1), 
      yhat_upper.reshape(-1,1)
      ), axis=1
    )
  preds_pd = pd.DataFrame(preds_np, columns=['yhat', 'yhat_lower', 'yhat_upper'])
  
  # ---------------------------------
  # PREPARE RESULTS
  # ---------------------------------
  # merge 
  results_pd = pd.concat(
    [data_pd, preds_pd],
    axis=1
    )
 
  # assign station to results
  results_pd['station_id']=station_id
  results_pd['model_version']=model_version
  # ---------------------------------
  
  return results_pd[
    ['model_version','station_id', 'ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']
    ]

# COMMAND ----------

# MAGIC %md we implement the function for generating prediction intervals:
# MAGIC [link to blog post](https://blog.datadive.net/prediction-intervals-for-random-forests/)

# COMMAND ----------

def pred_ints(model, X, interval_width):
    percentile = interval_width * 100
    err_down = []
    err_up = []
    for x in range(len(X)):
        preds = []
        for pred in model.estimators_:
            preds.append(pred.predict(X[x].reshape(1,-1))[0])
        err_down.append(np.percentile(preds, (100 - percentile) / 2. ))
        err_up.append(np.percentile(preds, 100 - (100 - percentile) / 2.))
    return np.asarray(err_down), np.asarray(err_up)

# COMMAND ----------

# MAGIC %md And now we are ready to assemble our dataset for training.  While we will not be employing any timeseries techniques, we will derive some features from the period timestamp.  Hour of day, day of week and the year itself will be employed along with a flag indicating whether or not a period is associated with a holiday. Month will not be used as month appears to correlate with temperature which is one of the two weather variables we will employ (along with precipitation).

# COMMAND ----------

from pyspark.sql.functions import lit

# assemble historical dataset for training
inputs = spark.sql('''
   SELECT
    a.station_id,
    a.hour as ds,
    EXTRACT(year from a.hour) as year,
    EXTRACT(dayofweek from a.hour) as dayofweek,
    EXTRACT(hour from a.hour) as hour,
    CASE WHEN d.date IS NULL THEN 0 ELSE 1 END as is_holiday,
    COALESCE(c.tot_precip_mm,0) as precip_mm,
    c.avg_temp_f as temp_f,
    COALESCE(b.rentals,0) as y
  FROM ( -- all rental hours by currently active stations
    SELECT 
      y.station_id,
      x.hour
    FROM citibike.periods x
    INNER JOIN citibike.stations_most_active y
     ON x.hour BETWEEN '{0}' AND '{1}'
    ) a
  LEFT OUTER JOIN citibike.rentals b
    ON a.station_id=b.station_id AND a.hour=b.hour
  LEFT OUTER JOIN citibike.weather c
    ON a.hour=c.time
  LEFT OUTER JOIN citibike.holidays d
    ON TO_DATE(a.hour)=d.date
  '''.format(start_date, end_date)
  )

# generate models and save the training results
forecast = (
  inputs
    .groupBy(inputs.station_id)
    .applyInPandas(train_model, schema=result_schema)
 ).write.format('delta').mode('append').save('dbfs:/FileStore/lpalum/citibike/forecast_regression_timeweather')

#forecast.createOrReplaceTempView('forecast_regression_timeweather')

# COMMAND ----------

# MAGIC %md Create a database table with the training result data

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS citibike.forecast_regression_timeweather;
# MAGIC 
# MAGIC CREATE TABLE citibike.forecast_regression_timeweather 
# MAGIC USING DELTA
# MAGIC LOCATION 'dbfs:/FileStore/lpalum/citibike/forecast_regression_timeweather'

# COMMAND ----------

# MAGIC %md Again, we create the function for visualizing our data:

# COMMAND ----------

# modified from https://github.com/facebook/prophet/blob/master/python/fbprophet/plot.py

from matplotlib import pyplot as plt
from matplotlib.dates import (
        MonthLocator,
        num2date,
        AutoDateLocator,
        AutoDateFormatter,
    )
from matplotlib.ticker import FuncFormatter

def generate_plot( model, forecast_pd, xlabel='ds', ylabel='y'):
  ax=None
  figsize=(10, 6)

  if ax is None:
      fig = plt.figure(facecolor='w', figsize=figsize)
      ax = fig.add_subplot(111)
  else:
      fig = ax.get_figure()
  
  history_pd = forecast_pd[forecast_pd['y'] != np.NaN]
  fcst_t = forecast_pd['ds'].dt.to_pydatetime()
  
  ax.plot(history_pd['ds'].dt.to_pydatetime(), history_pd['y'], 'k.')
  ax.plot(fcst_t, forecast_pd['yhat'], ls='-', c='#0072B2')
  ax.fill_between(fcst_t, forecast_pd['yhat_lower'], forecast_pd['yhat_upper'],
                  color='#0072B2', alpha=0.2)

  # Specify formatting to workaround matplotlib issue #12925
  locator = AutoDateLocator(interval_multiples=False)
  formatter = AutoDateFormatter(locator)
  ax.xaxis.set_major_locator(locator)
  ax.xaxis.set_major_formatter(formatter)
  ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  fig.tight_layout()

  return fig

# COMMAND ----------

# MAGIC %md And now we can explore Station 497 graphically:

# COMMAND ----------

# extract the forecast from our persisted dataset
forecast_pd = spark.sql('''
      SELECT
        a.ds,
        CASE WHEN a.ds > b.end_date THEN NULL ELSE a.y END as y,
        a.yhat,
        a.yhat_lower,
        a.yhat_upper
      FROM citibike.forecast_regression_timeweather a
      INNER JOIN citibike.stations_active b
        ON a.station_id=b.station_id
      WHERE 
        b.station_id=497
      ORDER BY a.ds
      ''').toPandas()

# COMMAND ----------

# retrieve the model for this station 
logged_model = f"models:/497-reg-rf-model/Production"
model = mlflow.sklearn.load_model(logged_model)

# COMMAND ----------

from datetime import datetime

# construct a visualization of the forecast
predict_fig = generate_plot(model, forecast_pd, xlabel='hour', ylabel='rentals')

# adjust the x-axis to focus on a limited date range
xlim = predict_fig.axes[0].get_xlim()
new_xlim = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(hours=168), datetime.strptime(end_date, '%Y-%m-%d'))
predict_fig.axes[0].set_xlim(new_xlim)

# display the chart
display(predict_fig)

# COMMAND ----------

# MAGIC %md Again, we generate our per-station evaluation metrics along with a summary metric for comparison with our other modeling techniques:

# COMMAND ----------

# MAGIC %sql -- per station
# MAGIC SELECT
# MAGIC   e.station_id,
# MAGIC   e.error_sum/n as MAE,
# MAGIC   e.error_sum_abs/n as MAD,
# MAGIC   e.error_sum_sqr/n as MSE,
# MAGIC   POWER(e.error_sum_sqr/n, 0.5) as RMSE,
# MAGIC   e.error_sum_abs_prop_y/n as MAPE
# MAGIC FROM (
# MAGIC   SELECT -- error base values 
# MAGIC     x.station_id,
# MAGIC     COUNT(*) as n,
# MAGIC     SUM(x.yhat-x.y) as error_sum,
# MAGIC     SUM(ABS(x.yhat-x.y)) as error_sum_abs,
# MAGIC     SUM(POWER((x.yhat-x.y),2)) as error_sum_sqr,
# MAGIC     SUM(ABS((x.yhat-x.y)/x.y_corrected)) as error_sum_abs_prop_y,
# MAGIC     SUM(ABS((x.yhat-x.y)/x.yhat)) as error_sum_abs_prop_yhat,
# MAGIC     SUM(x.y) as sum_y,
# MAGIC     SUM(x.yhat) as sum_yhat
# MAGIC   FROM ( -- actuals vs. forecast
# MAGIC     SELECT
# MAGIC       a.station_id,
# MAGIC       a.ds as ds,
# MAGIC       CAST(COALESCE(a.y,0) as float) as y,
# MAGIC       CAST(COALESCE(a.y,1) as float) as y_corrected,
# MAGIC       a.yhat
# MAGIC     FROM citibike.forecast_regression_timeweather a
# MAGIC     INNER JOIN citibike.stations b
# MAGIC       ON a.station_id = b.station_id AND
# MAGIC          a.ds <= b.end_date
# MAGIC      ) x
# MAGIC    GROUP BY x.station_id
# MAGIC   ) e
# MAGIC ORDER BY e.station_id

# COMMAND ----------

# MAGIC %sql -- all stations
# MAGIC 
# MAGIC SELECT
# MAGIC   e.error_sum/n as MAE,
# MAGIC   e.error_sum_abs/n as MAD,
# MAGIC   e.error_sum_sqr/n as MSE,
# MAGIC   POWER(e.error_sum_sqr/n, 0.5) as RMSE,
# MAGIC   e.error_sum_abs_prop_y/n as MAPE
# MAGIC FROM (
# MAGIC   SELECT -- error base values 
# MAGIC     COUNT(*) as n,
# MAGIC     SUM(x.yhat-x.y) as error_sum,
# MAGIC     SUM(ABS(x.yhat-x.y)) as error_sum_abs,
# MAGIC     SUM(POWER((x.yhat-x.y),2)) as error_sum_sqr,
# MAGIC     SUM(ABS((x.yhat-x.y)/x.y_corrected)) as error_sum_abs_prop_y,
# MAGIC     SUM(ABS((x.yhat-x.y)/x.yhat)) as error_sum_abs_prop_yhat,
# MAGIC     SUM(x.y) as sum_y,
# MAGIC     SUM(x.yhat) as sum_yhat
# MAGIC   FROM ( -- actuals vs. forecast
# MAGIC     SELECT
# MAGIC       a.ds as ds,
# MAGIC       CAST(COALESCE(a.y,0) as float) as y,
# MAGIC       CAST(COALESCE(a.y,1) as float) as y_corrected,
# MAGIC       a.yhat
# MAGIC     FROM citibike.forecast_regression_timeweather a
# MAGIC     INNER JOIN citibike.stations b
# MAGIC       ON a.station_id = b.station_id AND
# MAGIC          a.ds <= b.end_date
# MAGIC      ) x
# MAGIC   ) e

# COMMAND ----------

import json

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
