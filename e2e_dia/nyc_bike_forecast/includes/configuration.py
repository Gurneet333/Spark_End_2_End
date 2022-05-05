# Databricks notebook source
spark.conf.set("spark.sql.shuffle.partitions", "32")  # Configure the size of shuffles the same as core count
spark.conf.set("spark.sql.adaptive.enabled", "true")  # Spark 3.0 AQE - coalescing post-shuffle partitions, converting sort-merge join to broadcast join, and skew join optimization

USER='lpalum'   # CHANGE TO YOUR USER NAME

# Setup the hive meta store if it does not exist and select database as the focus of future sql commands in this notebook
#spark.sql(f"CREATE DATABASE IF NOT EXISTS dscc202_{USER}")
#spark.sql(f"USE dscc202_{USER}")

# Delta Tables stored here...
BASE_DELTA_TABLE = f"dbfs:/FileStore/tables/{USER}/citibike/"
dbutils.fs.mkdirs(BASE_DELTA_TABLE)

dbutils.fs.ls(f"dbfs:/FileStore/tables/{USER}/")

# COMMAND ----------

# Raw realtime data sources
BIKE_STATION_JSON = "https://gbfs.citibikenyc.com/gbfs/es/station_information.json"
BIKE_STATION_STATUS_JSON = "https://gbfs.citibikenyc.com/gbfs/es/station_status.json"
WEATHER_FORECAST_JSON = "https://api.openweathermap.org/data/2.5/onecall?lat=40.7128792&lon=-74.0060&exclude=current,minutely,daily,alerts&appid=13096c31b7822092c189c5c4682e574c"

#mounted directories that hold the historic bike trip and weather data
BIKE_FILE_ROOT = "/mnt/dscc202-datasets/nyc_bikes/"
WEATHER_FILE_ROOT = "/mnt/dscc202-datasets/weather/"

# COMMAND ----------

# Raw weather data
DELTA_TABLE_WEATHER = "dbfs:/FileStore/tables/lloyd.palum@rochester.edu/" + "WEATHER_BRONZE"

DELTA_TABLE_BIKE_TRIPS = BASE_DELTA_TABLE + "BRONZE_BIKE_TRIP"
DELTA_TABLE_BIKE_TRIPS_CHKPT = BASE_DELTA_TABLE + "_checkpoints/BRONZE_BIKE_TRIP"

DELTA_TABLE_NYC_WEATHER = BASE_DELTA_TABLE + "BRONZE_NYC_WEATHER"
DELTA_TABLE_NYC_WEATHER_CHKPT = BASE_DELTA_TABLE + "_checkpoints/BRONZE_NYC_WEATHER"


# COMMAND ----------


