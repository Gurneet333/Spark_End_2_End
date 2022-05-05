# Databricks notebook source
# MAGIC %md ##Data Preparation
# MAGIC 
# MAGIC For our forecasting exercise, we will make use of Citi Bike NYC trip history data aggregated to an hourly level and hourly weather data from Visual Crossing.  This notebook will prepare our data files for this work, but because of the terms of use associated with the datasets we will use, we will not provide instructions on how to access and download these.  If you wish to make use of these data, please visit the data providers' websites (linked to below), review their Terms of Use (also linked to below), and employ the data appropriately:</p>
# MAGIC 
# MAGIC * [[Terms of Use]](https://www.citibikenyc.com/data-sharing-policy) [Citibike NYC Trip History Data](https://s3.amazonaws.com/tripdata/index.html)
# MAGIC * [[Terms of Use]](https://www.visualcrossing.com/weather-services-terms) [Visual Crossing Hourly Weather Data for ZipCode 10001](https://www.visualcrossing.com/weather/weather-data-services)
# MAGIC 
# MAGIC Please note that the Citi Bike NYC data files are provided in a ZIP format. The steps below assume you have unzipped these files and removed any files representing overlapping data sets so that each trip is represented once and only once in the folder below. With the historical trip data loaded to a folder we will identify as /mnt/citibike/trips and weather data for Zip Code 10001 (Central Manhattan) downloaded to a folder we will identify as /mnt/weather/manhattan, let's examine the contents of each:

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

# MAGIC %run ./includes/utilities

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark import SparkFiles

# mount the source s3 dataset bucket
mount_datasets()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 0: Create Bronze Trip and Weather Data

# COMMAND ----------

# Define the structure of the csv bike trip records read from S3
schema = StructType([ \
    StructField("tripduration",LongType(),True), \
    StructField("starttime",TimestampType(),True), \
    StructField("stoptime",TimestampType(),True), \
    StructField("start_station_id", LongType(), True), \
    StructField("start_station_name", StringType(), True), \
    StructField("start_station_latitude", DoubleType(), True), \
    StructField("start_station_longitude", DoubleType(), True), \
    StructField("end_station_id", LongType(), True), \
    StructField("end_station_name", StringType(), True), \
    StructField("end_station_latitude", DoubleType(), True), \
    StructField("end_station_longitude", DoubleType(), True), \
    StructField("bikeid", LongType(), True), \
    StructField("usertype", StringType(), True), \
    StructField("birth_year", LongType(), True), \
    StructField("gender", LongType(), True)
  ])

# establish the read stream from the source S3 bucket for the trip data
streamingTripInputDF = (
  spark
    .readStream                       
    .schema(schema)                   # Set the schema of the csv data
    .option("maxFilesPerTrigger", 1)  # Treat a sequence of files as a stream by picking one file at a time
    .option("header", True)           # header row is included in the csv files
    .csv(BIKE_FILE_ROOT)
)

# Setup the write stream sink into the Bronze delta table for trips
tripQuery = streamingTripInputDF.writeStream \
  .format("delta") \
  .outputMode("append") \
  .option("checkpointLocation", DELTA_TABLE_BIKE_TRIPS_CHKPT) \
  .trigger(once=True) \
  .queryName("bikeTripStream") \
  .start(DELTA_TABLE_BIKE_TRIPS)

# Grab the weather data that is already in a delta table

# streaming read of the delta table
nycStreamingInputDF=(
  spark
    .readStream
    .format('delta') 
    .load(DELTA_TABLE_WEATHER)
    .where("NAME='NY CITY CENTRAL PARK, NY US'")
)

# Setup the write stream sink into the Bronze delta table for weather
weatherQuery = nycStreamingInputDF.writeStream \
  .format("delta") \
  .outputMode("append") \
  .option("checkpointLocation", DELTA_TABLE_NYC_WEATHER_CHKPT) \
  .trigger(once=True) \
  .queryName("weatherStream") \
  .start(DELTA_TABLE_NYC_WEATHER)

# get the list of active streaming queries and wait for this query to end before proceeding
[(q.name,q.id) for q in spark.streams.active]
weatherQuery.awaitTermination()
tripQuery.awaitTermination()

# COMMAND ----------

# Create two temp views
spark.read.format('delta').load(DELTA_TABLE_BIKE_TRIPS).createOrReplaceTempView("bike_trips_temp_view")
spark.read.format('delta').load(DELTA_TABLE_NYC_WEATHER).createOrReplaceTempView("nyc_weather_temp_view")

# COMMAND ----------

display( dbutils.fs.ls(DELTA_TABLE_BIKE_TRIPS))

# COMMAND ----------

# MAGIC %sql
# MAGIC select min(starttime),max(starttime) from bike_trips_temp_view

# COMMAND ----------

# MAGIC %sql
# MAGIC select min(DATE),max(DATE) from nyc_weather_temp_view

# COMMAND ----------

# MAGIC %md We will now define data objects on these files.  These objects will be placed in a catalog named *citibike* so that all assets for this exercise are kept seperate from others in our environment:

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP DATABASE IF EXISTS citibike CASCADE;
# MAGIC CREATE DATABASE citibike;

# COMMAND ----------

# MAGIC %md ###Step 1: Prepare Bike Rentals

# COMMAND ----------

# MAGIC %md Using the Citi Bike NYC trip history data, we can calculate the number of bikes rented from each station on an hourly basis.  To do this, we must first access the trip history data which is organized as a series of CSV files. These files implement a consistent schema across the date range for which we have trip history.  However, there are some variations in the formatting of the start (and stop) times for trips which needs to be addressed in code.</p>  
# MAGIC In addition, there are many fields in the dataset that we do not need in order to calculate hourly rentals.  These will be discarded as we construct our rental history dataset:

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *

# schema for the raw trip history data
trip_schema = StructType([
  StructField('tripduration', StringType()),
  StructField('start_time', StringType()),
  StructField('stop_time',  StringType()),
  StructField('start_station_id', StringType()),
  StructField('start_station_name', StringType()),
  StructField('start_station_latitude', StringType()),
  StructField('start_station_longitude', StringType()),
  StructField('end_station_id', StringType()),
  StructField('end_station_name', StringType()),
  StructField('end_station_latitude', StringType()),
  StructField('end_station_longitude', StringType()),
  StructField('bike_id', StringType()),
  StructField('user_type', StringType()),
  StructField('birth_year', StringType()),
  StructField('user_gender', StringType()),
  ])

# read the raw trip history data to dataframe
raw = spark.read.format("delta").load(DELTA_TABLE_BIKE_TRIPS)

# cleanse the data in preparation for analysis
cleansed = (
  raw
    .select(
      raw.start_station_id.cast(IntegerType()).alias('station_id'),
      col("starttime").alias('start_time')
      )
    .filter('(station_id Is Not Null) AND (start_time Is Not Null)') # remove any bad records
  )
cleansed.createOrReplaceTempView('trips')

display(cleansed)

# COMMAND ----------

# MAGIC %md We can now calculate the total number of rentals by hour and station and persist this data for re-use:

# COMMAND ----------

# this ensures we don't duplicate records on repeated runs of this notebook
try:
  dbutils.fs.rm('dbfs:/FileStore/lpalum/citibike/rentals', recurse=True)
except:
  pass

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS citibike.rentals;
# MAGIC 
# MAGIC CREATE TABLE citibike.rentals(
# MAGIC   station_id INTEGER,
# MAGIC   hour TIMESTAMP,
# MAGIC   rentals INTEGER
# MAGIC   )
# MAGIC   USING delta
# MAGIC   PARTITIONED BY (station_id)
# MAGIC   LOCATION 'dbfs:/FileStore/lpalum/citibike/rentals';
# MAGIC   
# MAGIC INSERT INTO citibike.rentals
# MAGIC   SELECT
# MAGIC     x.station_id,
# MAGIC     x.hour,
# MAGIC     COUNT(*) as rentals
# MAGIC   FROM ( 
# MAGIC     SELECT
# MAGIC       station_id,
# MAGIC       CAST( -- truncate start time to the hour
# MAGIC         CONCAT(
# MAGIC           CAST(DATE(start_time) as string), ' ',
# MAGIC           EXTRACT(hour from start_time), ':00:00.000'
# MAGIC           ) as TIMESTAMP
# MAGIC         ) as hour
# MAGIC       FROM trips
# MAGIC     ) x
# MAGIC     GROUP BY x.station_id, x.hour;  

# COMMAND ----------

# view the data now in our rentals table
display(
  spark.table('citibike.rentals')
)

# COMMAND ----------

# drop view for clarity in last step of this notebook
spark.sql('DROP VIEW trips')

# COMMAND ----------

# MAGIC %md ###Step 2: Prepare Station Info
# MAGIC 
# MAGIC It is important to know the dates for which each station is in operation in the bike share network. We will assume that a station came online at midnight on the day of its first rental. We will also assume each station is potentially *retired* after 11 PM on the last date for which we have rental data:

# COMMAND ----------

from pyspark.sql.functions import to_date, expr, min, max

# determine starting and ending dates for each station
(
  spark
    .table('citibike.rentals')
    .withColumn('start_of_date', to_date('hour').cast(TimestampType()))          # truncate date to start of midnight
    .withColumn('end_of_date', expr('start_of_date + INTERVAL 23 HOURS'))        # derive 11 PM on truncated date
    .groupBy('station_id').agg(
      min('start_of_date').alias('start_date'),                                  # min truncated date is station start date 
      max('end_of_date').alias('end_date')                                       # max 11 PM truncated date is station end date
      )
    .write
    .format('delta')
    .mode('overwrite')
    .save('dbfs:/FileStore/lpalum/citibike/stations')
  )

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS citibike.stations;
# MAGIC 
# MAGIC CREATE TABLE citibike.stations
# MAGIC   USING DELTA
# MAGIC   LOCATION 'dbfs:/FileStore/lpalum/citibike/stations';

# COMMAND ----------

display(
  spark.table('citibike.stations')
  )

# COMMAND ----------

# MAGIC %md To make it easier to focus on our active stations, let's create a view that limits the stations to those flagged as *retired* at the end of the last period for which we have data.  We will also exclude from our active stations dataset those stations that only came online within the last 30 days prior to the end of that same month:

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP VIEW IF EXISTS citibike.stations_active;
# MAGIC CREATE VIEW citibike.stations_active 
# MAGIC AS
# MAGIC   SELECT *
# MAGIC   FROM citibike.stations
# MAGIC   WHERE 
# MAGIC     end_date = (SELECT MAX(end_date) FROM citibike.stations) AND  -- currently active
# MAGIC     DATEDIFF(end_date, start_date) > 30                           -- has 30 days worth of history
# MAGIC     ;

# COMMAND ----------

# MAGIC %md ###Step 3: Prepare Calendar Info
# MAGIC 
# MAGIC To assist with the assembly of a complete dataset, it would be helpful to construct a calendar.  We'll create a calendar with one entry for each date between the first start date and last end date associated with our stations. We will add 10 days to the last end date so that our calendar extends into the future over a period for which we may potentially create forecasts:

# COMMAND ----------

from pyspark.sql.functions import lit, min, max, to_date
from pyspark.sql.types import *

from datetime import datetime

# get first and last date in stations dataset
first_date, last_date = (
    spark
      .table('citibike.stations')
      .agg(
        min('start_date').alias('first_date'),
        max(to_date('end_date').cast(TimestampType())).alias('last_date')
        )
    ).collect()[0]

# calculate days between first and last dates + 10 days for forecasting
days_between = (last_date - first_date).days + 10

# derive complete list of dates between first and last dates
dates = (
  spark
    .range(0,days_between)
    .withColumn('id', col('id').cast(IntegerType()))
    .withColumnRenamed('id','days')
    .withColumn('init_date', lit(first_date))
    .selectExpr('cast(date_add(init_date, days) as timestamp) as date')
  )

# COMMAND ----------

# MAGIC %md Some of the dates in our calendar will be holidays.  We'll derive holidays for the United States using the holidays library and then persist this data in a queriable table for later use:

# COMMAND ----------

import holidays

# identify holidays within years associated with first through last dates
holidays = (
  spark.createDataFrame(
    holidays.UnitedStates(years=range(first_date.year, last_date.year+1)).items(),
    ['date','holiday']
    )
    .selectExpr('cast(date as timestamp) as date', 'holiday')
  )

(
  holidays
    .write
    .format('delta')
    .mode('overwrite')
    .save('dbfs:/FileStore/lpalum/citibike/holidays')
)

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS citibike.holidays;
# MAGIC 
# MAGIC CREATE TABLE citibike.holidays 
# MAGIC   USING delta 
# MAGIC   LOCATION 'dbfs:/FileStore/lpalum/citibike/holidays';

# COMMAND ----------

# MAGIC %md As our rental data is organized by hourly intervals, we should align our calendar data with this, deriving 24 entries for each date currently in the dates dataset:

# COMMAND ----------

from pyspark.sql.functions import expr

# generate values 0 through 23 to represent the hours in a day 
hours = spark.range(0,24).withColumnRenamed('id', 'hour_increment')

# cross-join hours with dates to create hourly timestamps
periods = (
  dates
    .crossJoin(hours)
    .withColumn('hour', expr('cast( cast(date as integer)+(3600*hour_increment) as timestamp)'))
    .select('hour')
    )

(
  periods
    .write
    .format('delta')
    .mode('overwrite')
    .save('dbfs:/FileStore/lpalum/citibike/periods')
  )

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS citibike.periods;
# MAGIC 
# MAGIC CREATE TABLE citibike.periods
# MAGIC   USING delta 
# MAGIC   LOCATION 'dbfs:/FileStore/lpalum/citibike/periods';

# COMMAND ----------

# MAGIC %md ###Step 4: Prepare Weather
# MAGIC 
# MAGIC The last dataset we need to prepare is our weather data. As this data is already organized at an hourly level, the work involved here is minimal:

# COMMAND ----------

from pyspark.sql.types import *
"""
weather_schema = StructType([
  StructField('address', StringType()),
  StructField('hour', TimestampType()),
  StructField('max_temp_f', FloatType()),
  StructField('min_temp_f', FloatType()),
  StructField('avg_temp_f', FloatType()),
  StructField('dew_point_f', FloatType()),
  StructField('relative_humidity_percent', FloatType()),
  StructField('heat_index_f', FloatType()),
  StructField('wind_speed_mph', FloatType()),
  StructField('wind_gust_mph', FloatType()),
  StructField('wind_direction', FloatType()),
  StructField('wind_chill_f', FloatType()),
  StructField('precip_in', FloatType()),
  StructField('precip_cover_percent', FloatType()),
  StructField('snow_depth_in', FloatType()),
  StructField('visibility', FloatType()),
  StructField('cloud_cover_percent', FloatType()),
  StructField('sea_level_pressure', FloatType()),
  StructField('weather_type', StringType()),
  StructField('latitude', FloatType()),
  StructField('longitude', FloatType()),
  StructField('resolved_address', StringType()),
  StructField('name', StringType()),
  StructField('info', StringType()),
  StructField('conditions',  StringType()),
  StructField('contributing_stations',  StringType())
  ])

weather = (
  spark.read.csv(
    '/mnt/weather/manhattan/', 
    header=True, 
    schema=weather_schema, 
    timestampFormat='MM/dd/yyyy HH:mm:ss'
    )
    .filter('hour Is Not Null')
  )
"""
weather=(spark.read.format("delta").load(DELTA_TABLE_NYC_WEATHER)
        .withColumn('temp_f', split(col('TMP'),",")[0]*9/50+32)
        .withColumn('temp_qual', split(col('TMP'),",")[1])
        .withColumn('precip_hr_dur', split(col('AA1'),",")[0])
        .withColumn('precip_mm_intvl', split(col('AA1'),",")[1]/10)
        .withColumn('precip_cond', split(col('AA1'),",")[2])
        .withColumn('precip_qual', split(col('AA1'),",")[3])
        .withColumn('precip_mm', col('precip_mm_intvl')/col('precip_hr_dur'))
        .withColumn("time", date_trunc('hour', "DATE"))
        .where("REPORT_TYPE='FM-15' and precip_qual='5' and temp_qual='5' and time Is Not Null")
        .groupby("time")
        .agg(mean('temp_f').alias('avg_temp_f'), \
             sum('precip_mm').alias('tot_precip_mm'))
   )

# save weather data to delta format
(
  weather
    .write
    .format('delta')
    .mode('overwrite')
    .save('dbfs:/FileStore/lpalum/citibike/weather/manhattan_delta')
  )

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS citibike.weather;
# MAGIC 
# MAGIC CREATE TABLE citibike.weather 
# MAGIC   USING delta 
# MAGIC   LOCATION 'dbfs:/FileStore/lpalum/citibike/weather/manhattan_delta'

# COMMAND ----------

# MAGIC %md ###Step 5: Verify Data Objects in Catalog
# MAGIC 
# MAGIC We should now have a complete catalog of data objects with which to perform our work:

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES IN citibike;

# COMMAND ----------

# MAGIC %md ###One More Thing...
# MAGIC 
# MAGIC In the next notebook, we will explore a problem with stations having large numbers of inactive periods.  To focus on our most active stations (all of which will have some but hopefully few inactive periods themselves), we will create a table identifying our most active stations:

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text('04.stations_to_model', '3')
stations_to_model = int(dbutils.widgets.get('04.stations_to_model')) #input

# COMMAND ----------

spark.sql("""DROP TABLE IF EXISTS citibike.stations_most_active;""")

query=f"""
CREATE TABLE citibike.stations_most_active 
  USING DELTA 
  AS
    SELECT
      a.station_id,
      a.start_date,
      a.end_date,
      100 * COUNT(*) /
      SUM(COALESCE(b.rentals, 1)) as Percent_Hours_With_No_Rentals
    FROM ( -- all rental hours by currently active stations
      SELECT
        y.station_id,
        y.start_date,
        y.end_date,
        x.hour
      FROM citibike.periods x
      INNER JOIN citibike.stations_active y
        ON x.hour BETWEEN y.start_date AND y.end_date
      ) a
    LEFT OUTER JOIN citibike.rentals b
      ON a.station_id=b.station_id AND a.hour=b.hour
    GROUP BY a.station_id, a.start_date, a.end_date
    ORDER BY Percent_Hours_with_No_Rentals
    LIMIT {stations_to_model};
"""
spark.sql(query)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from citibike.stations_most_active;

# COMMAND ----------

import json

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
