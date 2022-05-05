# Databricks notebook source
# MAGIC %md
# MAGIC ## End 2 End Example of a Data Intensive Application
# MAGIC 
# MAGIC In this application we are forecasting the demand for bike rentals at a number of the most popular stations in the NYC Citibike system.
# MAGIC The application is organized as follows:
# MAGIC 1. An ETL process creates tables in the **citibike** database
# MAGIC  - holidays
# MAGIC  - periods
# MAGIC  - rentals
# MAGIC  - stations
# MAGIC  - stations_active
# MAGIC  - stations_most_active
# MAGIC  - weather
# MAGIC  - forecast_regression_timeweather
# MAGIC 2. An EDA process explores the records in those databases and highlights trend and seasonality
# MAGIC 3. A modeling process trains and tunes a forecasting model for each of the most active stations using the specified time range. The model is registered as **Staging** if there is a production model for the station or if not it is registered as **Production**.
# MAGIC 4. An Application process allows a given station to be selected and a forecast of a specific number of hours to be done.  The **Production** and **Staging** model versions are also compared using recent historic data.  The process also allows a model to be **promoted** from Staging to Production.

# COMMAND ----------

from datetime import datetime as dt
from datetime import timedelta
import json

dbutils.widgets.removeAll()

dbutils.widgets.text('01.start_date', "2018-01-01")
dbutils.widgets.text('02.end_date', "2019-03-15")
dbutils.widgets.text('03.hours_to_forecast', '72')
dbutils.widgets.text('04.stations_to_model', '3')


start_date = str(dbutils.widgets.get('01.start_date'))
end_date = str(dbutils.widgets.get('02.end_date'))
hours_to_forecast = int(dbutils.widgets.get('03.hours_to_forecast'))
stations_to_model = int(dbutils.widgets.get('04.stations_to_model'))
print(start_date,end_date,hours_to_forecast,stations_to_model)

# COMMAND ----------

# DBTITLE 1,ETL
# Run the Data Prepartion
result = dbutils.notebook.run("/Shared/e2e_dia/nyc_bike_forecast/01 Data Preparation", 3600, {"04.stations_to_model":stations_to_model})

# Check the results
assert json.loads(result)["exit_code"] == "OK", "Data Preparation Failed!" # Check to see that it worked

# COMMAND ----------

# DBTITLE 1,EDA
# Run the Data Exploration
result = dbutils.notebook.run("/Shared/e2e_dia/nyc_bike_forecast/02 Exploratory Analysis", 3600)

# Check the results
assert json.loads(result)["exit_code"] == "OK", "EDA Failed!" # Check to see that it worked

# COMMAND ----------

# DBTITLE 1,Modeling with SKlearn (Random Forest)
# Run the Random Forest time series model on the top stations using the weather regression factors.
result = dbutils.notebook.run("/Shared/e2e_dia/nyc_bike_forecast/06 RandomForest with Time & Weather", 3600, {"01.start_date":start_date, "02.end_date":end_date})

# Check the results
assert json.loads(result)["exit_code"] == "OK", "Modeling Failed!" # Check to see that it worked

# COMMAND ----------

# DBTITLE 1,Application and Monitoring
# Run the Random Forest time series model on the top stations using the weather regression factors.
result = dbutils.notebook.run("/Shared/e2e_dia/nyc_bike_forecast/07 Bike Demand Forecast", 3600, {"01.start_date":start_date, "02.end_date":end_date,"03.hours_to_forecast":hours_to_forecast,"04.station":"Broadway & E 22 St","05.promote_model":"No"})

# Check the results
assert json.loads(result)["exit_code"] == "OK", "Application and Monitoring Failed!" # Check to see that it worked
