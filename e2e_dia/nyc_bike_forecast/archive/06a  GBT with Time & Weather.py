# Databricks notebook source
# MAGIC %md ##Forecasting Using GBT with Temporal & Weather Features
# MAGIC In this notebook, we will build regression models to forecast rentals using some basic temporal information and some weather data. As before, we will start by installing the libraries needed for this work.  Notice we're installing a relatively new version of SciKit-Learn to gain access to some expanded functionality for the RandomForestRegressor:

# COMMAND ----------

spark.conf.set("spark.sql.shuffle.partitions", "32")  # Configure the size of shuffles the same as core count
spark.conf.set("spark.sql.adaptive.enabled", "true")  # Spark 3.0 AQE - coalescing post-shuffle partitions, converting sort-merge join to broadcast join, and skew join optimization
dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')

# COMMAND ----------

from datetime import datetime as dt
from datetime import timedelta

tdf=spark.sql("select min(start_date) as a,max(start_date),max(end_date) as b from citibike.stations_most_active;")[["a","b"]].collect()
dbutils.widgets.removeAll()

dbutils.widgets.text('01.start_date', str(tdf[0]["a"].strftime("%Y-%m-%d")))
dbutils.widgets.text('02.end_date', str(tdf[0]["b"].strftime("%Y-%m-%d")))
dbutils.widgets.text('03.hours_to_forecast', '36')

start_date = str(dbutils.widgets.get('01.start_date'))
end_date = str(dbutils.widgets.get('02.end_date'))
hours_to_forecast = int(dbutils.widgets.get('03.hours_to_forecast'))
print(start_date,end_date,hours_to_forecast)

# COMMAND ----------

# MAGIC %md Now we define our function.  As with the last notebook, the incoming DataFrame will contain both historical values and future (predicted) values for weather. We will need to exclude the future values during training:

# COMMAND ----------

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.metrics import mean_absolute_error

import shutil

from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor

# structure of the dataset returned by the function
result_schema =StructType([
  StructField('station_id',IntegerType()),
  StructField('ds',TimestampType()),
  StructField('y', FloatType()),
  StructField('yhat', FloatType())
  ])

# forecast function
#@pandas_udf( result_schema, PandasUDFType.GROUPED_MAP )
def get_forecast(group_pd):
  
  # DATA PREP
  # ---------------------------------
  # identify station id and hours to forecast
  station_id = group_pd['station_id'].iloc[0]
  hours_to_forecast=group_pd['hours_to_forecast'].iloc[0]
  
  # fill records (rows) with nan values
  data_pd = group_pd.fillna(method='ffill').fillna(method='bfill')
  
  # extract valid historical data
  history_pd = data_pd[data_pd['is_historical']==1]
  
  # separate features and labels
  X_all = data_pd.drop(['station_id','ds','is_historical','hours_to_forecast'], axis=1).values
  X_hist = history_pd.drop(['station_id','ds','is_historical','hours_to_forecast'], axis=1).values

  # ---------------------------------  
  
  with mlflow.start_run(run_name="citibike-randomforest-with-regressors-{0}".format(station_id)) as run:

    # TRAIN MODEL
    # ---------------------------------  
    # train model
    from pyspark.ml.feature import VectorAssembler, VectorIndexer

    # Remove the target column from the input feature set.
    featuresCols = data_pd.drop(['station_id','ds','is_historical','hours_to_forecast','y'], axis=1).columns

    # vectorAssembler combines all feature columns into a single feature vector column, "rawFeatures".
    vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol="rawFeatures")

    # vectorIndexer identifies categorical features and indexes them, and creates a new column "features". 
    vectorIndexer = VectorIndexer(inputCol="rawFeatures", outputCol="features", maxCategories=4)
    
    from pyspark.ml.regression import GBTRegressor

    # The next step is to define the model training stage of the pipeline. 
    # The following command defines a GBTRegressor model that takes an input column "features" by default and learns to predict the labels in the "y" column. 
    gbt = GBTRegressor(labelCol="y")

    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    from pyspark.ml.evaluation import RegressionEvaluator

    # Define a grid of hyperparameters to test:
    #  - maxDepth: maximum depth of each decision tree 
    #  - maxIter: iterations, or the total number of trees 
    paramGrid = ParamGridBuilder()\
      .addGrid(gbt.maxDepth, [2, 5])\
      .addGrid(gbt.maxIter, [10, 100])\
      .build()

    # Define an evaluation metric.  The CrossValidator compares the true labels with predicted values for each combination of parameters, and calculates this value to determine the best model.
    evaluator = RegressionEvaluator(metricName="rmse", labelCol=gbt.getLabelCol(), predictionCol=gbt.getPredictionCol())

    # Declare the CrossValidator, which performs the model tuning.
    cv = CrossValidator(estimator=gbt, evaluator=evaluator, estimatorParamMaps=paramGrid)
    
 
    from pyspark.ml import Pipeline
    pipeline = Pipeline(stages=[vectorAssembler, vectorIndexer, cv])
    
    pipelineModel = pipeline.fit(X_hist)
    
    predictions = pipelineModel.transform(X_all)
    
    rmse = evaluator.evaluate(predictions)
    
    # save models for potential later use
    mlflow.spark.log_model(pipelineModel, 
                             artifact_path="{0}_gbt_model".format(station_id),
                             registered_model_name="{}-reg-gbt-model".format(station_id)
                            )
    # Log params
    mlflow.log_params({'maxDepth': [2,5], 'maxIter': [10,100]})
    
    # log performance
    mlflow.log_metric("rmse", rmse)
    
    # move this latest version of the model to the production stage
    client = MlflowClient()
    model_name = "{}-reg-gbt-model".format(station_id)
    model_version = dict(client.search_model_versions(f"name='{model_name}'")[0])['version']
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage="Production"
    )
    
    # ---------------------------------
  
  # FORECAST
  # ---------------------------------  
  # generate forecast
  data_pd['yhat'] = predictions
  
  # ---------------------------------
  
  return data_pd[
    ['station_id', 'ds', 'y', 'yhat']
    ]

# COMMAND ----------

# MAGIC %md And now we are ready to assemble our dataset for training and forecasting.  While we will not be employing any timeseries techniques, we will derive some features from the period timestamp.  Hour of day, day of week and the year itself will be employed along with a flag indicating whether or not a period is associated with a holiday. Month will not be used as month appears to correlate with temperature which is one of the two weather variables we will employ (along with precipitation).
# MAGIC 
# MAGIC As before, we will need to include forecasted temperature and precipitation data so that the data set assembled here will employ the *is_historical* flag to seperate historical from future values:

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
    COALESCE(b.rentals,0) as y,
    a.is_historical,
    {0} as hours_to_forecast
  FROM ( -- all rental hours by currently active stations
    SELECT 
      y.station_id,
      x.hour,
      CASE WHEN x.hour <= '{2}' THEN 1 ELSE 0 END as is_historical
    FROM citibike.periods x
    INNER JOIN citibike.stations_most_active y
     ON x.hour BETWEEN '{1}' AND ('{2}' + INTERVAL {0} HOURS)
    ) a
  LEFT OUTER JOIN citibike.rentals b
    ON a.station_id=b.station_id AND a.hour=b.hour
  LEFT OUTER JOIN citibike.weather c
    ON a.hour=c.time
  LEFT OUTER JOIN citibike.holidays d
    ON TO_DATE(a.hour)=d.date
  '''.format(hours_to_forecast, start_date, end_date)
  )

# generate forecast
forecast = (
  inputs
    .groupBy(inputs.station_id, lit(hours_to_forecast))
    .applyInPandas(get_forecast, schema=result_schema)
  )
forecast.createOrReplaceTempView('forecast_gbt_timeweather')

# COMMAND ----------

# MAGIC %md We can now trigger the execution of our logic and load the resulting forecasts to a table for long-term persistence:

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS citibike.forecast_gbt_timeweather;
# MAGIC 
# MAGIC CREATE TABLE citibike.forecast_gbt_timeweather 
# MAGIC USING DELTA
# MAGIC AS
# MAGIC   SELECT * 
# MAGIC   FROM forecast_gbt_timeweather

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
      FROM citibike.forecast_gbt_timeweather a
      INNER JOIN citibike.stations_active b
        ON a.station_id=b.station_id
      WHERE 
        b.station_id=497
      ORDER BY a.ds
      ''').toPandas()

# COMMAND ----------

# retrieve the model for this station 
logged_model = 'runs:/26b511f7a4944e83ac77f56c5951ba63/497_model'
model = mlflow.sklearn.load_model(logged_model)

# COMMAND ----------

from datetime import datetime

# construct a visualization of the forecast
predict_fig = generate_plot(model, forecast_pd, xlabel='hour', ylabel='rentals')

# adjust the x-axis to focus on a limited date range
xlim = predict_fig.axes[0].get_xlim()
new_xlim = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(hours=hours_to_forecast), datetime.strptime(end_date, '%Y-%m-%d') + timedelta(hours=hours_to_forecast))
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
# MAGIC     FROM citibike.forecast_gbt_timeweather a
# MAGIC     INNER JOIN citibike.stations b
# MAGIC       ON a.station_id = b.station_id AND
# MAGIC          a.ds <= b.end_date
# MAGIC      ) x
# MAGIC   ) e

# COMMAND ----------


