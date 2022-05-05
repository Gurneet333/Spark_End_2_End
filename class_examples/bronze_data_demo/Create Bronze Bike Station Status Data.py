# Databricks notebook source
# MAGIC %md
# MAGIC ## Bronze Data Ingest System Overview
# MAGIC <img src="https://data-science-at-scale.s3.amazonaws.com/images/e2e_applications/DIA+Framework-NYC+Bike+Status+Data.png">

# COMMAND ----------

!/databricks/python3/bin/python -m pip install --upgrade pip
!pip install folium

# COMMAND ----------

import pandas as pd
import json
from pyspark.sql.session import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
import time

# configureation of the spark environment appropriate for my cluster
spark.conf.set("spark.sql.shuffle.partitions", "32")  # Configure the size of shuffles the same as core count
spark.conf.set("spark.sql.adaptive.enabled", "true")  # Spark 3.0 AQE - coalescing post-shuffle partitions, converting sort-merge join to broadcast join, and skew join optimization
BASE_PATH = "/mnt/dscc202-datasets/misc/bike-feed/"

# COMMAND ----------

"""
Stream utility functions
"""
def stop_all_streams() -> bool:
    stopped = False
    for stream in spark.streams.active:
        stopped = True
        stream.stop()
    return stopped


def stop_named_stream(spark: SparkSession, namedStream: str) -> bool:
    stopped = False
    for stream in spark.streams.active:
        if stream.name == namedStream:
            stopped = True
            stream.stop()
    return stopped


def untilStreamIsReady(namedStream: str, progressions: int = 3) -> bool:
    queries = [query for query in spark.streams.active if query.name == namedStream]
    #queries = list(filter(lambda query: query.name == namedStream, spark.streams.active))
    while len(queries) == 0 or len(queries[0].recentProgress) < progressions:
        time.sleep(5)
        #queries = list(filter(lambda query: query.name == namedStream, spark.streams.active))
        queries = [query for query in spark.streams.active if query.name == namedStream]
    print("The stream {} is active and ready.".format(namedStream))
    return True

# This routine will mount the classroom datasets in S3
def mount_datasets():
  mount_name = "dscc202-datasets"
  s3_bucket = "s3a://dscc202-datasets/"
  try:
    dbutils.fs.mount(s3_bucket, "/mnt/%s" % mount_name)
  except:
    print("already mounted")

  display(dbutils.fs.ls("/mnt/%s" % mount_name)  )


# COMMAND ----------

# mount the class datasets
mount_datasets()

# delete the previous runs to make this idempotent
spark.sql(f"""DROP DATABASE IF EXISTS bikeDB CASCADE""")
spark.sql(f"CREATE DATABASE IF NOT EXISTS bikeDB")
spark.sql(f"USE bikeDB")

# COMMAND ----------

# Define the schema of the data read from the S3 bucket
jsonSchema = StructType([ \
  StructField("capacity",LongType(),True), \
  StructField("eightd_has_available_keys",BooleanType(),True), \
  StructField("eightd_has_key_dispenser",BooleanType(),True), \
  StructField("eightd_station_services",StringType(),True), \
  StructField("electric_bike_surcharge_waiver",BooleanType(),True), \
  StructField("external_id",StringType(),True), \
  StructField("has_kiosk",BooleanType(),True), \
  StructField("is_installed",LongType(),True), \
  StructField("is_renting",LongType(),True), \
  StructField("is_returning",LongType(),True), \
  StructField("last_reported",LongType(),True), \
  StructField("lat",DoubleType(),True), \
  StructField("legacy_id",StringType(),True), \
  StructField("lon",DoubleType(),True), \
  StructField("name",StringType(),True), \
  StructField("num_bikes_available",LongType(),True), \
  StructField("num_bikes_disabled",LongType(),True), \
  StructField("num_docks_available",LongType(),True), \
  StructField("num_docks_disabled",LongType(),True), \
  StructField("num_ebikes_available",LongType(),True), \
  StructField("region_id",StringType(),True), \
  StructField("rental_methods",StringType(),True), \
  StructField("rental_uris_android",StringType(),True), \
  StructField("rental_uris_ios",StringType(),True), \
  StructField("short_name",StringType(),True), \
  StructField("station_id",StringType(),True), \
  StructField("station_status",StringType(),True), \
  StructField("station_type",StringType(),True), \
  StructField("update_time",TimestampType(),True)
  ])

# COMMAND ----------

# DBTITLE 1,Read the incoming bike status from S3 and stream them to a delta table
bikeReadDF = (
  spark.readStream
  .format("json")
  .schema(jsonSchema)
  .load(BASE_PATH)
)

bikeWriteDF =(bikeReadDF.writeStream
 .format("delta")
 .queryName("bike_status_stream")
 .option("mode","append")
 .option("checkpointLocation", BASE_PATH + "raw_checkpoint")
 .option("path", BASE_PATH + "raw")
 .start())

untilStreamIsReady("bike_status_stream")


# COMMAND ----------

# DBTITLE 1,Create the meta data for the bronze bike status delta table
spark.sql(f"""
DROP TABLE IF EXISTS bronze_raw_bike_status;
""")

spark.sql(f"""
CREATE TABLE bronze_raw_bike_status
USING DELTA
LOCATION "{BASE_PATH}raw"
""")

# COMMAND ----------

# DBTITLE 1,Show where several of the stations are using the folium mapping package
# grab the station information 
stationDF=(spark.read
           .format("delta")
           .load(BASE_PATH+"raw")
           .where("is_renting = 1 and name in ('E 47 St & 2 Ave','Atlantic Ave & Fort Greene Pl','W 17 St & 8 Ave','MacDougal St & Prince St','E 2 St & Avenue C')")
           .select("name","station_id","lat","lon")
           .distinct()
          ).toPandas()

import folium
from folium.plugins import MarkerCluster

m = folium.Map(
    location=[40.7128, -74.0060],
    zoom_start=13,
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

m

# COMMAND ----------

# DBTITLE 1,Display the average bikes and docks available per station over the last hour (window)
bikeStreamInputDF = spark.readStream.format("delta").load(BASE_PATH+"raw")

streamingCountsDF = (
  bikeStreamInputDF
    .where("is_renting = 1 and name in ('E 47 St & 2 Ave','Atlantic Ave & Fort Greene Pl','W 17 St & 8 Ave','MacDougal St & Prince St','E 2 St & Avenue C')")
    .groupBy(
      bikeStreamInputDF.name,
      window(bikeStreamInputDF.update_time, "1 hour"))
  	.agg(avg("num_bikes_available").alias("num_bikes"),avg("num_docks_available").alias("num_docks"))
)
display(streamingCountsDF, streamName="hour_station_counts")

# COMMAND ----------

for s in spark.streams.active:
  s.stop()
