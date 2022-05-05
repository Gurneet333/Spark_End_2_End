# Databricks notebook source
# MAGIC %md
# MAGIC ## Bronze Data Ingest System Overview
# MAGIC <img src="https://data-science-at-scale.s3.amazonaws.com/images/e2e_applications/DIA+Framework-NYC+Bike+Status+Data.png">

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
    queries = list(filter(lambda query: query.name == namedStream, spark.streams.active))
    while len(queries) == 0 or len(queries[0].recentProgress) < progressions:
        time.sleep(5)
        queries = list(filter(lambda query: query.name == namedStream, spark.streams.active))
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

  #display(dbutils.fs.ls("/mnt/%s" % mount_name)  )


# COMMAND ----------

mount_datasets()

# COMMAND ----------

# DBTITLE 1,List the bronze bike-feed delta table (no partitioning)
# MAGIC %fs ls /mnt/dscc202-datasets/misc/bike-feed/raw

# COMMAND ----------

# MAGIC %md
# MAGIC ## How many directories comprise the delta table

# COMMAND ----------

len(dbutils.fs.ls("dbfs:/mnt/dscc202-datasets/misc/bike-feed/raw"))

# COMMAND ----------

# MAGIC %fs ls /mnt/dscc202-datasets/misc/bike-feed/raw/_delta_log/

# COMMAND ----------

display(spark.read.json("/mnt/dscc202-datasets/misc/bike-feed/raw/_delta_log/00000000000000000008.json"))

# COMMAND ----------

display(spark.read.parquet("dbfs:/mnt/dscc202-datasets/misc/bike-feed/raw/_delta_log/00000000000000000010.checkpoint.parquet"))

# COMMAND ----------

# DBTITLE 1,Re-partition the bronze bike feed delta table using the station_id 
bikeReadDF = (
  spark.read
  .format("delta")
  .load(BASE_PATH + "raw")
)

# COMMAND ----------

"""
READ THE UN-PARTITIONED TABLE AND FILTER ON STATION_ID = 72 (note time)
"""
bikeReadDF.select("station_id").filter("station_id=72").count()

# COMMAND ----------

(bikeReadDF.write
 .format("delta")
 .partitionBy("station_id")
 .mode("overwrite")
 .save(BASE_PATH + "raw-partitioned"))

# COMMAND ----------

# MAGIC %fs ls /mnt/dscc202-datasets/misc/bike-feed/raw-partitioned/

# COMMAND ----------

len(dbutils.fs.ls("dbfs:/mnt/dscc202-datasets/misc/bike-feed/raw-partitioned"))

# COMMAND ----------

bikeReadPartitionedDF = (
  spark.read
  .format("delta")
  .load(BASE_PATH + "raw-partitioned")
)

# COMMAND ----------

"""
READ THE PARTITIONED TABLE AND FILTER ON STATION_ID = 72 (note time)
"""
bikeReadPartitionedDF.select("station_id").filter("station_id=72").count()

# COMMAND ----------

# DBTITLE 1,Optimize the partitioned table
# MAGIC %sql
# MAGIC OPTIMIZE delta.`/mnt/dscc202-datasets/misc/bike-feed/raw-partitioned`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read different versions of the table

# COMMAND ----------

"""
READ THE OPTIMIZED PARTITIONED TABLE AND FILTER ON STATION_ID = 72 (note time)
"""
(
  spark.read
  .format("delta")
  .option("versionAsOf", 100)
  .load(BASE_PATH + "raw")
).select("station_id").filter("station_id=72").count()

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY bikedb.bronze_raw_bike_status

# COMMAND ----------

# MAGIC %sql
# MAGIC select name, station_id, count(1) as record_count from bikedb.bronze_raw_bike_status  VERSION AS OF 100 group by name, station_id order by record_count desc;

# COMMAND ----------

# MAGIC %sql
# MAGIC select name, station_id, count(1) as record_count from bikedb.bronze_raw_bike_status  TIMESTAMP AS OF '2022-03-16T18:12:02.000Z' group by name, station_id order by record_count desc;
