# Databricks notebook source
# DBTITLE 0,Data Engineering
# MAGIC %md
# MAGIC ## Data Engineering
# MAGIC - Enable access to the data
# MAGIC - Make it reliable for application
# MAGIC 
# MAGIC <img src="https://data-science-at-scale.s3.amazonaws.com/data-science-at-scale/images/delta_lake_pipeline_1.png" style="width:400;">
# MAGIC 
# MAGIC - [Citibike Data](https://www.citibikenyc.com/system-data)
# MAGIC - [NOAA Weather Data](# https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf)

# COMMAND ----------

# MAGIC %run ../includes/configuration

# COMMAND ----------

# MAGIC %run ./includes/utilities

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark import SparkFiles

# mount the s3 dataset bucket
mount_datasets()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Bronze Trip and Weather Data

# COMMAND ----------

# DBTITLE 1,Read the Bike Trips (stream from S3 bucket)
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

# COMMAND ----------

# Remove the trip delta table and its checkpoint
#dbutils.fs.rm(DELTA_TABLE_BIKE_TRIPS, recurse=True)
#dbutils.fs.rm(DELTA_TABLE_BIKE_TRIPS_CHKPT, recurse=True)
dbutils.fs.ls("dbfs:/FileStore/tables/lpalum")

# COMMAND ----------

# establish the read stream from the source S3 bucket for the trip data
streamingTripInputDF = (
  spark
    .readStream                       
    .schema(schema)                   # Set the schema of the csv data
    .option("maxFilesPerTrigger", 1)  # Treat a sequence of files as a stream by picking one file at a time
    .option("header", True)           # header row is included in the csv files
    .csv(BIKE_FILE_ROOT)
)

# COMMAND ----------

# Setup the write stream sink into the Bronze delta table for trips
query = streamingTripInputDF.writeStream \
  .format("delta") \
  .outputMode("append") \
  .option("checkpointLocation", DELTA_TABLE_BIKE_TRIPS_CHKPT) \
  .trigger(once=True) \
  .queryName("bikeTripStream") \
  .start(DELTA_TABLE_BIKE_TRIPS)

# COMMAND ----------

# get the list of active streaming queries and wait for this query to end before proceeding
[(q.name,q.id) for q in spark.streams.active]
query.awaitTermination()

# COMMAND ----------

spark.read.format('delta').load(DELTA_TABLE_BIKE_TRIPS).createOrReplaceTempView("bike_trips_temp_view")

# COMMAND ----------

# MAGIC %sql
# MAGIC select min(starttime),max(starttime) from bike_trips_temp_view

# COMMAND ----------

# DBTITLE 1,Read the NYC weather (stream from S3 bucket)
# Remove the trip delta table and its checkpoint
#dbutils.fs.rm(DELTA_TABLE_NYC_WEATHER, recurse=True)
#dbutils.fs.rm(DELTA_TABLE_NYC_WEATHER_CHKPT, recurse=True)
dbutils.fs.ls("dbfs:/FileStore/tables/lpalum")

# COMMAND ----------

BASE_DELTA_TABLE = f"dbfs:/FileStore/tables/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')}/"
DELTA_TABLE_WEATHER = BASE_DELTA_TABLE + "WEATHER_BRONZE"

# COMMAND ----------

# streaming read of the delta table
nycStreamingInputDF=(
  spark
    .readStream
    .format('delta') 
    .load(DELTA_TABLE_WEATHER)
    .where("NAME='NY CITY CENTRAL PARK, NY US'")
)

# COMMAND ----------

# Setup the write stream sink into the Bronze delta table for weather
query = nycStreamingInputDF.writeStream \
  .format("delta") \
  .outputMode("append") \
  .option("checkpointLocation", DELTA_TABLE_NYC_WEATHER_CHKPT) \
  .trigger(once=True) \
  .queryName("weatherStream") \
  .start(DELTA_TABLE_NYC_WEATHER)

# COMMAND ----------

# get the list of active streaming queries and wait for this query to end before proceeding
[(q.name,q.id) for q in spark.streams.active]
query.awaitTermination()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Silver Model Training Data
# MAGIC - Extract temperature in farenheit, wind speed in meters per second and precipiation in mm
# MAGIC - average the temp and precip per hour
# MAGIC - determine the number of bikes rented, returned and the change in inventory per hour per station
# MAGIC - join the weather and inventory data based on time and update the silver modeling data table

# COMMAND ----------

dbutils.fs.rm(DELTA_TABLE_MODELING_DATA, recurse=True)

# COMMAND ----------

nycWeatherDF=(spark.read.format("delta").load(DELTA_TABLE_NYC_WEATHER)
        .withColumn('temp_f', split(col('TMP'),",")[0]*9/50+32)
        .withColumn('temp_qual', split(col('TMP'),",")[1])
        .withColumn('precip_hr_dur', split(col('AA1'),",")[0])
        .withColumn('precip_mm_intvl', split(col('AA1'),",")[1]/10)
        .withColumn('precip_cond', split(col('AA1'),",")[2])
        .withColumn('precip_qual', split(col('AA1'),",")[3])
        .withColumn('precip_mm', col('precip_mm_intvl')/col('precip_hr_dur'))
        .withColumn("time", date_trunc('hour', "DATE"))
        .where("REPORT_TYPE='FM-15' and precip_qual='5' and temp_qual='5'")
        .groupby("time")
        .agg(mean('temp_f').alias('avg_temp_f'), \
             sum('precip_mm').alias('tot_precip_mm'))
   )

# COMMAND ----------

stationStartTripDF=(spark.read.format("delta").load(DELTA_TABLE_BIKE_TRIPS)
               .filter(col("starttime").isNotNull() & col("start_station_id").isNotNull())
               .withColumn("time", date_trunc('hour', "starttime"))
               .groupby("time","start_station_id")
               .agg(count('start_station_id').alias("total_rentals"))
               .sort("start_station_id","time")
              )
display(stationStartTripDF)

# COMMAND ----------

stationEndTripDF=(spark.read.format("delta").load(DELTA_TABLE_BIKE_TRIPS)
               .filter(col("stoptime").isNotNull() & col("end_station_id").isNotNull())
               .withColumn("time", date_trunc('hour', "stoptime"))
               .groupby("time","end_station_id")
               .agg(count('end_station_id').alias("total_returns"))
               .sort("end_station_id","time")
              )
display(stationEndTripDF)

# COMMAND ----------

stationCapacityDF = spark.createDataFrame(get_bike_stations()[['capacity','station_id','name']]).sort('station_id')
display(stationCapacityDF)

# COMMAND ----------

theDF= (stationStartTripDF
        .join(stationEndTripDF, "time")
        .where("end_station_id=start_station_id")
        .withColumnRenamed("start_station_id","station_id")
        .drop("end_station_id")
        .withColumn('inventory_change', col('total_returns')-col('total_rentals'))
        .join(nycWeatherDF,"time")
        .join(stationCapacityDF, 'station_id')
        .filter(col('capacity')!=0)
        .sort("station_id","time"))

display(theDF)

# COMMAND ----------

theDF.write \
  .format('delta') \
  .mode('overwrite') \
  .save(DELTA_TABLE_MODELING_DATA)

# COMMAND ----------

spark.read.load(DELTA_TABLE_MODELING_DATA).count()

# COMMAND ----------

# Setup the metadata for the bronze and silver tables
drop_make("bronze_bike_trips",DELTA_TABLE_BIKE_TRIPS)
drop_make("bronze_nyc_weather",DELTA_TABLE_NYC_WEATHER)
drop_make("silver_modeling_data",DELTA_TABLE_MODELING_DATA)

# COMMAND ----------

# MAGIC %md
# MAGIC <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img src="https://drive.google.com/uc?export=view&id=1ILbthTP3pUJMrbN0HawvChIXCnyM9-l1" alt="Data Science at Scale" style=" width: 250px; height: auto;"></a>
# MAGIC 
# MAGIC <a href="mailto:lpalum@gmail.com">lpalum at gmail</a>
