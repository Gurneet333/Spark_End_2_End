# Databricks notebook source
# DBTITLE 0,High Performance Spark Jobs and Queries with Databricks Delta
# MAGIC %md
# MAGIC # High Performance Spark Queries with Databricks Delta
# MAGIC Databricks Delta extends Apache Spark to simplify data reliability and boost Spark's performance.
# MAGIC 
# MAGIC Building robust, high performance data pipelines can be difficult due to: _lack of indexing and statistics_, _data inconsistencies introduced by schema changes_ and _pipeline failures_, _and having to trade off between batch and stream processing_.
# MAGIC 
# MAGIC With Databricks Delta, data engineers can build reliable and fast data pipelines. Databricks Delta provides many benefits including:
# MAGIC * Faster query execution with indexing, statistics, and auto-caching support
# MAGIC * Data reliability with rich schema validation and rransactional guarantees
# MAGIC * Simplified data pipeline with flexible UPSERT support and unified Structured Streaming + batch processing on a single data source.
# MAGIC 
# MAGIC ### Let's See How Databricks Delta Makes Spark Queries Faster!
# MAGIC 
# MAGIC In this example, we will see how Databricks Delta can optimize query performance. We create a standard table using Parquet format and run a quick query to observe its latency. We then run a second query over the Databricks Delta version of the same table to see the performance difference between standard tables versus Databricks Delta tables. 
# MAGIC 
# MAGIC Simply follow these 4 steps below:
# MAGIC * __Step 1__ : Create a standard Parquet based table using data from US based flights schedule data
# MAGIC * __Step 2__ : Run a query to to calculate number of flights per month, per originating airport over a year
# MAGIC * __Step 3__ : Create the flights table using Databricks Delta and optimize the table.
# MAGIC * __Step 4__ : Rerun the query in Step 2 and observe the latency. 
# MAGIC 
# MAGIC __Note:__ _Throughout the example we will be building few tables with a 10s of million rows. Some of the operations may take a few minutes depending on your cluster configuration._

# COMMAND ----------

# DBTITLE 1,Clean up Parquet tables
# MAGIC %fs rm -r /tmp/flights_parquet 

# COMMAND ----------

# DBTITLE 1,Clean up Databricks Delta tables
# MAGIC %fs rm -r /tmp/flights_delta

# COMMAND ----------

# DBTITLE 1,Step 0: Read flights data
flights = spark.read.format("csv") \
  .option("header", "true") \
  .option("inferSchema", "true") \
  .load("/databricks-datasets/asa/airlines/2008.csv")

# COMMAND ----------

# DBTITLE 1,Step 1: Write a Parquet based table using flights data
flights.write.format("parquet").mode("overwrite").partitionBy("Origin").save("/tmp/flights_parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC Once step 1 completes, the "flights" table contains details of US flights for a year. 
# MAGIC 
# MAGIC Next in Step 2, we run a query that get top 20 cities with highest monthly total flights on first day of week.

# COMMAND ----------

# DBTITLE 1,Step 2: Run a query
from pyspark.sql.functions import count

flights_parquet = spark.read.format("parquet").load("/tmp/flights_parquet")

display(flights_parquet.filter("DayOfWeek = 1").groupBy("Month","Origin").agg(count("*").alias("TotalFlights")).orderBy("TotalFlights", ascending=False).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC Once step 2 completes, you can observe the latency with the standard "flights_parquet" table. 
# MAGIC 
# MAGIC In step 3 and step 4, we do the same with a Databricks Delta table. This time, before running the query, we run the `OPTIMIZE` command with `ZORDER` to ensure data is optimized for faster retrieval. 

# COMMAND ----------

# DBTITLE 1,Step 3: Write a Databricks Delta based table using flights data
flights.write.format("delta").mode("overwrite").partitionBy("Origin").save("/tmp/flights_delta")

# COMMAND ----------

# DBTITLE 1,Step 3 Continued: OPTIMIZE the Databricks Delta table
display(spark.sql("DROP TABLE  IF EXISTS flights"))

display(spark.sql("CREATE TABLE flights USING DELTA LOCATION '/tmp/flights_delta'"))
                  
display(spark.sql("OPTIMIZE flights ZORDER BY (DayofWeek)"))

# COMMAND ----------

# DBTITLE 1,Step 4 : Rerun the query from Step 2 and observe the latency
flights_delta = spark.read.format("delta").load("/tmp/flights_delta")

display(flights_delta.filter("DayOfWeek = 1").groupBy("Month","Origin").agg(count("*").alias("TotalFlights")).orderBy("TotalFlights", ascending=False).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC The query over the Databricks Delta table runs much faster after `OPTIMIZE` is run. How much faster the query runs can depend on the configuration of the cluster you are running on, however should be **5-10X** faster compared to the standard table. 
