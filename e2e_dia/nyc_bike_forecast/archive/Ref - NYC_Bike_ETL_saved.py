# Databricks notebook source
# MAGIC %md
# MAGIC ## Application Overview
# MAGIC - What is the objective?
# MAGIC   - Augment/replace?
# MAGIC   - Optimize?
# MAGIC - Where is the data coming from?
# MAGIC - What is in the Data (metadata of the source)?
# MAGIC 
# MAGIC New York City has a vibrant and dynamic bike share system that was originally fielded by Citibank and has recently been taken over by Lyft.
# MAGIC The data is freely shared in the public domain here: [https://www.citibikenyc.com/system-data](https://www.citibikenyc.com/system-data)
# MAGIC 
# MAGIC One of the important aspects of running this system is to make sure that stations within the sharing network have the **right number** of bikes.  As you can
# MAGIC imagine there are certain **flows** within the network that cause an imbalance in the bike inventory at specific stations.  For example,
# MAGIC popular origin stations may have too fee bikes and popular destination stations may have too many bikes at certain times during the day.
# MAGIC 
# MAGIC The operators of this system need to routinely dispatch trailers that move bikes around the network from station to station to compensate for this
# MAGIC imbalance in flows.  We are going to develop a data intensive application that **augments** the operating staff with a dash board that projects
# MAGIC in real time what the inventory of bikes will be at the various stations in the network.  They can then use this information to drive
# MAGIC their redistribution of bikes.
# MAGIC 
# MAGIC - [Cornell research steers NYC bikes to needy stations](https://news.cornell.edu/stories/2015/01/cornell-research-steers-nyc-bikes-needy-stations)
# MAGIC - [A Dynamic Approach to Rebalancing Bike-Sharing Systems](https://www.mdpi.com/1424-8220/18/2/512/pdf)
# MAGIC - [Aukland Bike Data](https://nbviewer.jupyter.org/github/nicolasfauchereau/Auckland_Cycling/blob/master/notebooks/Auckland_cycling_and_weather.ipynb)
# MAGIC - [Prophet Forecasting Library](https://peerj.com/preprints/3190.pdf)

# COMMAND ----------

# DBTITLE 0,Data Engineering
# MAGIC %md
# MAGIC ## Data Engineering
# MAGIC - Enable access to the data
# MAGIC - Make it reliable for application
# MAGIC 
# MAGIC <img src="https://data-science-at-scale.s3.amazonaws.com/data-science-at-scale/images/delta_lake_pipeline_1.png" style="width:40%;height:auto;">

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

# MAGIC %run ./includes/utilities

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark import SparkFiles

# Setup the hive meta store if it does not exist and select database as the focus of future sql commands in this notebook
spark.sql(f"CREATE DATABASE IF NOT EXISTS dscc202_{USER}")
spark.sql(f"USE dscc202_{USER}")

# mount the s3 dataset bucket
mount_datasets()

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
streamingTripInputDF.writeStream \
  .format("delta") \
  .outputMode("append") \
  .option("checkpointLocation", DELTA_TABLE_BIKE_TRIPS_CHKPT) \
  .trigger(once=True) \
  .queryName("bikeTripStream") \
  .start(DELTA_TABLE_BIKE_TRIPS)

# COMMAND ----------

# get the list of active streaming queries
[(q.name,q.id) for q in spark.streams.active]
#streamingInputDF.stop()

# COMMAND ----------

#spark.streams.awaitAnyTermination()
#drop_make("bronze_bike_trips", DELTA_TABLE_BIKE_TRIPS)

# COMMAND ----------

#%sql
#select min(starttime),max(starttime) from bronze_bike_trips where start_station_id = 3263 or end_station_id = 3263;

# COMMAND ----------

# DBTITLE 1,Read the NYC weather (stream from S3 bucket)
# Remove the trip delta table and its checkpoint
#dbutils.fs.rm(DELTA_TABLE_NYC_WEATHER, recurse=True)
#dbutils.fs.rm(DELTA_TABLE_NYC_WEATHER_CHKPT, recurse=True)
dbutils.fs.ls("dbfs:/FileStore/tables/lpalum")

# COMMAND ----------

schema = StructType([           StructField('STATION',StringType(),True),StructField('DATE',TimestampType(),True),StructField('SOURCE',ShortType(),True),StructField('LATITUDE',DoubleType(),True),StructField('LONGITUDE',DoubleType(),True),StructField('ELEVATION',DoubleType(),True),StructField('NAME',StringType(),True),StructField('REPORT_TYPE',StringType(),True),StructField('CALL_SIGN',StringType(),True),StructField('QUALITY_CONTROL',StringType(),True),StructField('WND',StringType(),True),StructField('CIG',StringType(),True),StructField('VIS',StringType(),True),StructField('TMP',StringType(),True),StructField('DEW',StringType(),True),StructField('SLP',StringType(),True),StructField('AW1',StringType(),True),StructField('GA1',StringType(),True),StructField('GA2',StringType(),True),StructField('GA3',StringType(),True),StructField('GA4',StringType(),True),StructField('GE1',StringType(),True),StructField('GF1',StringType(),True),StructField('KA1',StringType(),True),StructField('KA2',StringType(),True),StructField('MA1',StringType(),True),StructField('MD1',StringType(),True),StructField('MW1',StringType(),True),StructField('MW2',StringType(),True),StructField('OC1',StringType(),True),StructField('OD1',StringType(),True),StructField('OD2',StringType(),True),StructField('REM',StringType(),True),StructField('EQD',StringType(),True),StructField('AW2',StringType(),True),StructField('AX4',StringType(),True),StructField('GD1',StringType(),True),StructField('AW5',StringType(),True),StructField('GN1',StringType(),True),StructField('AJ1',StringType(),True),StructField('AW3',StringType(),True),StructField('MK1',StringType(),True),StructField('KA4',StringType(),True),StructField('GG3',StringType(),True),StructField('AN1',StringType(),True),StructField('RH1',StringType(),True),StructField('AU5',StringType(),True),StructField('HL1',StringType(),True),StructField('OB1',StringType(),True),StructField('AT8',StringType(),True),StructField('AW7',StringType(),True),StructField('AZ1',StringType(),True),StructField('CH1',StringType(),True),StructField('RH3',StringType(),True),StructField('GK1',StringType(),True),StructField('IB1',StringType(),True),StructField('AX1',StringType(),True),StructField('CT1',StringType(),True),StructField('AK1',StringType(),True),StructField('CN2',StringType(),True),StructField('OE1',StringType(),True),StructField('MW5',StringType(),True),StructField('AO1',StringType(),True),StructField('KA3',StringType(),True),StructField('AA3',StringType(),True),StructField('CR1',StringType(),True),StructField('CF2',StringType(),True),StructField('KB2',StringType(),True),StructField('GM1',StringType(),True),StructField('AT5',StringType(),True),StructField('AY2',StringType(),True),StructField('MW6',StringType(),True),StructField('MG1',StringType(),True),StructField('AH6',StringType(),True),StructField('AU2',StringType(),True),StructField('GD2',StringType(),True),StructField('AW4',StringType(),True),StructField('MF1',StringType(),True),StructField('AA1',StringType(),True),StructField('AH2',StringType(),True),StructField('AH3',StringType(),True),StructField('OE3',StringType(),True),StructField('AT6',StringType(),True),StructField('AL2',StringType(),True),StructField('AL3',StringType(),True),StructField('AX5',StringType(),True),StructField('IB2',StringType(),True),StructField('AI3',StringType(),True),StructField('CV3',StringType(),True),StructField('WA1',StringType(),True),StructField('GH1',StringType(),True),StructField('KF1',StringType(),True),StructField('CU2',StringType(),True),StructField('CT3',StringType(),True),StructField('SA1',StringType(),True),StructField('AU1',StringType(),True),StructField('KD2',StringType(),True),StructField('AI5',StringType(),True),StructField('GO1',StringType(),True),StructField('GD3',StringType(),True),StructField('CG3',StringType(),True),StructField('AI1',StringType(),True),StructField('AL1',StringType(),True),StructField('AW6',StringType(),True),StructField('MW4',StringType(),True),StructField('AX6',StringType(),True),StructField('CV1',StringType(),True),StructField('ME1',StringType(),True),StructField('KC2',StringType(),True),StructField('CN1',StringType(),True),StructField('UA1',StringType(),True),StructField('GD5',StringType(),True),StructField('UG2',StringType(),True),StructField('AT3',StringType(),True),StructField('AT4',StringType(),True),StructField('GJ1',StringType(),True),StructField('MV1',StringType(),True),StructField('GA5',StringType(),True),StructField('CT2',StringType(),True),StructField('CG2',StringType(),True),StructField('ED1',StringType(),True),StructField('AE1',StringType(),True),StructField('CO1',StringType(),True),StructField('KE1',StringType(),True),StructField('KB1',StringType(),True),StructField('AI4',StringType(),True),StructField('MW3',StringType(),True),StructField('KG2',StringType(),True),StructField('AA2',StringType(),True),StructField('AX2',StringType(),True),StructField('AY1',StringType(),True),StructField('RH2',StringType(),True),StructField('OE2',StringType(),True),StructField('CU3',StringType(),True),StructField('MH1',StringType(),True),StructField('AM1',StringType(),True),StructField('AU4',StringType(),True),StructField('GA6',StringType(),True),StructField('KG1',StringType(),True),StructField('AU3',StringType(),True),StructField('AT7',StringType(),True),StructField('KD1',StringType(),True),StructField('GL1',StringType(),True),StructField('IA1',StringType(),True),StructField('GG2',StringType(),True),StructField('OD3',StringType(),True),StructField('UG1',StringType(),True),StructField('CB1',StringType(),True),StructField('AI6',StringType(),True),StructField('CI1',StringType(),True),StructField('CV2',StringType(),True),StructField('AZ2',StringType(),True),StructField('AD1',StringType(),True),StructField('AH1',StringType(),True),StructField('WD1',StringType(),True),StructField('AA4',StringType(),True),StructField('KC1',StringType(),True),StructField('IA2',StringType(),True),StructField('CF3',StringType(),True),StructField('AI2',StringType(),True),StructField('AT1',StringType(),True),StructField('GD4',StringType(),True),StructField('AX3',StringType(),True),StructField('AH4',StringType(),True),StructField('KB3',StringType(),True),StructField('CU1',StringType(),True),StructField('CN4',StringType(),True),StructField('AT2',StringType(),True),StructField('CG1',StringType(),True),StructField('CF1',StringType(),True),StructField('GG1',StringType(),True),StructField('MV2',StringType(),True),StructField('CW1',StringType(),True),StructField('GG4',StringType(),True),StructField('AB1',StringType(),True),StructField('AH5',StringType(),True),StructField('CN3',StringType(),True)])


# COMMAND ----------

# establish the read stream from the source S3 bucket for the weather data 2015 --> 2019
streamingInput2015DF = (
  spark
    .readStream                       
    .option("maxFilesPerTrigger", 1)  # Treat a sequence of files as a stream by picking one file at a time
    .schema(schema)
    .load(WEATHER_FILE_ROOT+f"2015.parquet")
)
nycStreamingInput2015DF = streamingInput2015DF.where("NAME='NY CITY CENTRAL PARK, NY US'")

# establish the read stream from the source S3 bucket for the weather data
streamingInput2016DF = (
  spark
    .readStream                       
    .option("maxFilesPerTrigger", 1)  # Treat a sequence of files as a stream by picking one file at a time
    .schema(schema)
    .load(WEATHER_FILE_ROOT+f"2016.parquet")
)
nycStreamingInput2016DF = streamingInput2016DF.where("NAME='NY CITY CENTRAL PARK, NY US'")

# establish the read stream from the source S3 bucket for the weather data
streamingInput2017DF = (
  spark
    .readStream                       
    .option("maxFilesPerTrigger", 1)  # Treat a sequence of files as a stream by picking one file at a time
    .schema(schema)
    .load(WEATHER_FILE_ROOT+f"2017.parquet")
)
nycStreamingInput2017DF = streamingInput2017DF.where("NAME='NY CITY CENTRAL PARK, NY US'")

# establish the read stream from the source S3 bucket for the weather data
streamingInput2018DF = (
  spark
    .readStream                       
    .option("maxFilesPerTrigger", 1)  # Treat a sequence of files as a stream by picking one file at a time
    .schema(schema)
    .load(WEATHER_FILE_ROOT+f"2018.parquet")
)
nycStreamingInput2018DF = streamingInput2018DF.where("NAME='NY CITY CENTRAL PARK, NY US'")

# establish the read stream from the source S3 bucket for the weather data
streamingInput2019DF = (
  spark
    .readStream                       
    .option("maxFilesPerTrigger", 1)  # Treat a sequence of files as a stream by picking one file at a time
    .schema(schema)
    .load(WEATHER_FILE_ROOT+f"2019.parquet")
)
nycStreamingInput2019DF = streamingInput2019DF.where("NAME='NY CITY CENTRAL PARK, NY US'")

# COMMAND ----------

# Take the union of the 5 yearly weather data streams 
nycStreamingInputDF = nycStreamingInput2019DF.union(nycStreamingInput2018DF).union(nycStreamingInput2017DF).union(nycStreamingInput2016DF).union(nycStreamingInput2015DF)

# COMMAND ----------

# Setup the write stream sink into the Bronze delta table for weather
nycStreamingInputDF.writeStream \
  .format("delta") \
  .outputMode("append") \
  .option("checkpointLocation", DELTA_TABLE_NYC_WEATHER_CHKPT) \
  .trigger(once=True) \
  .queryName("weatherStream") \
  .start(DELTA_TABLE_NYC_WEATHER)

# COMMAND ----------

# get the list of active streaming queries
[(q.name,q.id) for q in spark.streams.active]
#streamingInputDF.stop()

# COMMAND ----------

#spark.streams.awaitAnyTermination()
#drop_make("bronze_nyc_weather", DELTA_TABLE_NYC_WEATHER)

# COMMAND ----------

#%sql
#select min(DATE),max(DATE) from bronze_nyc_weather

# COMMAND ----------

# Grab pandas dataframes for weather
df_weather = get_current_weather()

DELTA_TABLE_WEATHER_FORECAST = BASE_DELTA_TABLE + "WEATHER_FORECAST"

# delete the old delta table
dbutils.fs.rm(DELTA_TABLE_WEATHER_FORECAST, recurse=True)

df_weather['wind_speed_mps']=df_weather['wind_speed'].apply(lambda x: x*0.44704)
df_weather['starttime']=df_weather['dt'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S',  time.gmtime(x-18000)))
df_weather['endtime']=df_weather['dt'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S',  time.gmtime(x-18000+3600)))
df_weather['endtime']=df_weather['dt'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S',  time.gmtime(x-18000+3600)))
df_weather['shortForecast']=df_weather['weather'].apply(lambda x: x[0]['description'])
df_weather.drop("weather", axis=1, inplace=True)
df_weather.drop("pop", axis=1, inplace=True)
df_weather.drop("snow.1h", axis=1, inplace=True)

# Write a new delta table
spark.createDataFrame(df_weather).write.format('delta').mode('overwrite').save(DELTA_TABLE_WEATHER_FORECAST)

drop_make("e2e_weather_forecast",DELTA_TABLE_WEATHER_FORECAST)


# COMMAND ----------

#mount_name = "ds402_flight_data"
#dbutils.fs.mount(create_s3_path("flight-data-feed").replace("s3","s3a"), "/mnt/%s" % mount_name)
#dbutils.fs.unmount("/mnt/%s" % mount_name)
#dbutils.fs.ls("/mnt/%s" % mount_name)                 

# COMMAND ----------

# To read a specific file listed in this dataframe: 
# pd.read_csv(BIKE_FILE_ROOT+df_bike_files.toPandas().iloc[-1].Name.replace(u'\xa0', u''))
# Grab pandas dataframes for bike trip history files
df_bike_files = get_bike_data_files()
df_bike_files.head()

# COMMAND ----------

# Grab pandas dataframes for station information
df_stations = get_bike_stations()

DELTA_TABLE_STATION_INFO = BASE_DELTA_TABLE + "STATION_INFO"

# delete the old delta table
dbutils.fs.rm(DELTA_TABLE_STATION_INFO, recurse=True)

# Write a new delta table
spark.createDataFrame(df_stations[['lon', 'legacy_id', 'station_id', 'station_type','lat', 'region_id', 'capacity', 'name']]).write.format('delta').mode('overwrite').save(DELTA_TABLE_STATION_INFO)

drop_make("e2e_station_info",DELTA_TABLE_STATION_INFO)

df_stations[['lon', 'legacy_id', 'station_id', 'station_type','lat', 'region_id', 'capacity', 'name']].head()


# COMMAND ----------

# Grab pandas dataframes for stations current status
df_station_status = get_bike_station_status()


DELTA_TABLE_STATION_STATUS = BASE_DELTA_TABLE + "STATION_STATUS"

# delete the old delta table
dbutils.fs.rm(DELTA_TABLE_STATION_STATUS, recurse=True)

# Write a new delta table
spark.createDataFrame(df_station_status[['num_ebikes_available', 'is_installed', 'legacy_id',
       'num_docks_disabled', 'num_docks_available',
       'is_renting', 'is_returning',
       'num_bikes_disabled', 'last_reported', 'station_status',
       'num_bikes_available', 'station_id']]).write.format('delta').mode('overwrite').save(DELTA_TABLE_STATION_STATUS)

drop_make("e2e_station_status",DELTA_TABLE_STATION_STATUS)

# COMMAND ----------

# DBTITLE 1,Read the weather data for New York from S3 (temp and description)



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Create silver training data 
# Where are we storing the training data
DELTA_TABLE_TRAIN_DATA = BASE_DELTA_TABLE + "SILVER_BIKE_TRAIN"

# delete the old delta table
dbutils.fs.rm(DELTA_TABLE_TRAIN_DATA, recurse=True)

# create the range of training data accumulation
train_range=[('2017-1-01', '2017-2-01'),
 ('2017-2-01', '2017-3-01'),
 ('2017-3-01', '2017-4-01'),
 ('2017-4-01', '2017-5-01'),
 ('2017-5-01', '2017-6-01'),
 ('2017-6-01', '2017-7-01'),
 ('2017-7-01', '2017-8-01'),
 ('2017-8-01', '2017-9-01'),
 ('2017-9-01', '2017-10-01'),
 ('2017-10-01', '2017-11-01'),
 ('2017-11-01', '2017-12-01'),
 ('2017-12-01', '2018-01-01'),
 ('2018-01-01', '2018-02-01'),
 ('2018-02-01', '2018-03-01'),
 ('2018-03-01', '2018-04-01'),
 ('2018-04-01', '2018-05-01'),
 ('2018-05-01', '2018-06-01'),
 ('2018-06-01', '2018-07-01')
]

for i in train_range:
  # Get bike used counts
  sdf=spark.read.format('delta').table("e2e_bike_trips").filter(f"starttime>='{i[0]}' and starttime<'{i[1]}'").toPandas().rename(columns={"bikeid":"bikes_rented"}).groupby('start_station_id').resample("H", on='starttime')['bikes_rented'].count().reset_index()
  # Get bike returned counts
  edf=spark.read.format('delta').table("e2e_bike_trips").filter(f"starttime>='{i[0]}' and starttime<'{i[1]}'").toPandas().rename(columns={"bikeid":"bikes_returned"}).groupby('end_station_id').resample("H", on='stoptime')['bikes_returned'].count().reset_index()
  
  # write the month out to the delta table
  tdf = (spark.createDataFrame(sdf) 
     .join(spark.createDataFrame(edf), (col("starttime")==col("stoptime")) & (col("start_station_id")==col("end_station_id") )) 
     .withColumnRenamed("start_station_id","station_id") 
     .withColumnRenamed("starttime","time") 
     .withColumn("inventory_change", col("bikes_returned")-col("bikes_rented")) 
     .select("time","station_id","inventory_change") 
     .join(spark.read.format('delta').load(DELTA_TABLE_NY_WEATHER), col("time")==col("datetime")).drop("datetime")
     .write.format('delta').mode('append').save(DELTA_TABLE_TRAIN_DATA))

# COMMAND ----------

# establish the metadata in the hive store.
drop_make("e2e_bike_train",DELTA_TABLE_TRAIN_DATA)

# COMMAND ----------

# take a look at the table
df = spark.read.format("delta").table("e2e_bike_train")
df.agg(min('time'),max('time')).show()

# COMMAND ----------

# MAGIC %sql
# MAGIC OPTIMIZE e2e_bike_trips
# MAGIC ZORDER by start_station_id, end_station_id

# COMMAND ----------

# MAGIC %sql
# MAGIC OPTIMIZE e2e_bike_train
# MAGIC ZORDER by station_id

# COMMAND ----------

#from pyspark.streaming import StreamingContext
#StreamingContext(spark,60).stop(stopSparkContext=False,stopGraceFully=True)

# COMMAND ----------

# MAGIC %md
# MAGIC <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img src="https://drive.google.com/uc?export=view&id=1ILbthTP3pUJMrbN0HawvChIXCnyM9-l1" alt="Data Science at Scale" style=" width: 250px; height: auto;"></a>
# MAGIC 
# MAGIC <a href="mailto:lpalum@gmail.com">lpalum at gmail</a>
