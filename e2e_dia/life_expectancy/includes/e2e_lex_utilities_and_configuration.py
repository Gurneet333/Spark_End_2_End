# Databricks notebook source
#imports
import pandas as pd
import numpy as np
import requests
import json
from pyspark import SparkFiles
from pyspark.sql import functions as F

# enter your user name here
USER = 'lpalum'

# COMMAND ----------

# DBTITLE 1,Configuration
# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

# World Health and World in Data Source URL
WHO_INDICATOR_DESC_URL='https://ghoapi.azureedge.net/api/Indicator'
WHO_COUNTRIES_URL='https://ghoapi.azureedge.net/api/DIMENSION/COUNTRY/DimensionValues'
WHO_EXAMPLE_URL='https://ghoapi.azureedge.net/api/WHOSIS_000001?$filter=Dim1%20eq%20%27MLE%27%20and%20date(TimeDimensionBegin)%20ge%202011-01-01%20and%20date(TimeDimensionBegin)%20lt%202012-01-01'
WHO_BASE_INDICATOR_URL='https://ghoapi.azureedge.net/api/'

WID_DRUG_DATA = "https://data-science-at-scale.s3.amazonaws.com/data/life_expectancy/deaths_drug_overdoses.csv"
WHO_LEX_DATA = "https://data-science-at-scale.s3.amazonaws.com/data/life_expectancy/life_expectancy_data.csv"
WORLD_POP_DATA = "https://data-science-at-scale.s3.amazonaws.com/data/life_expectancy/WPP2019_POP_F01_1_TOTAL_POPULATION_BOTH_SEXES.csv"

# Delta table storage paths
BASE_DELTA_TABLE = f"dbfs:/FileStore/tables/{USER}/"

DELTA_TABLE_WHO_INDICATOR_DESCRIPTIONS = BASE_DELTA_TABLE + "BRONZE_WHO_IND_DESC"
DELTA_TABLE_WHO_COUNTRIES = BASE_DELTA_TABLE + "BRONZE_WHO_COUNTRIES"
DELTA_TABLE_WHO_INDICATORS = BASE_DELTA_TABLE + "BRONZE_WHO_INDICATORS"
DELTA_TABLE_WID_DRUG_DEATHS = BASE_DELTA_TABLE + "BRONZE_WID_DRUG_DEATHS"
DELTA_TABLE_WHO_LEX = BASE_DELTA_TABLE + "BRONZE_WHO_LEX_FEATURES"
DELTA_TABLE_WORLD_POP = BASE_DELTA_TABLE + "BRONZE_WORLD_POPULATION"

# Setup the hive meta store if it does not exist
spark.sql(f"CREATE DATABASE IF NOT EXISTS dsc402_{USER}")
spark.sql(f"USE dsc402_{USER}");

# WHO Indicators of Interest
WHO_INDICATORS = ['AIR_3','DEVICES00','dptv','NUTRITION_2005','WHS2_160','WHS2_161','SA_0000001452','SA_0000001453','SA_0000001455','WHOSIS_000001']

# COMMAND ----------

# DBTITLE 1,Get the source data and Store in Bronze Delta Tables
def get_json(path):
  print(f"INFO: retreiving {path}")
  data= requests.get(path)
  if data.status_code == 200:
    json_recs=data.json()
  else:
    print(f"ERROR: unable to retreive {path} with status_code={data.status_code}")
    json_recs={"value":[]}
  return spark.read.json(sc.parallelize([json.dumps(i) for i in json_recs['value']]))

def write_delta_table(df, path, mode):
  print(f"INFO: {mode} {path}")
  return df.write.format("delta").mode(mode).save(path)

def get_who_indicator(indicator):
  df=get_json(WHO_BASE_INDICATOR_URL+indicator)
  df = df.select("SpatialDim","TimeDimensionValue","Value","IndicatorCode")
  df=df.withColumnRenamed("SpatialDim","Country").withColumnRenamed("TimeDimensionValue","Year").withColumn("Year",col("Year").cast(IntegerType())).withColumn("Value",col("Value").cast(DoubleType()))
  return write_delta_table(df,DELTA_TABLE_WHO_INDICATORS,'append')

def get_who_indicator_descriptions():
  return write_delta_table(get_json(WHO_INDICATOR_DESC_URL),DELTA_TABLE_WHO_INDICATOR_DESCRIPTIONS,'overwrite')

def get_who_countries():
  df=get_json(WHO_COUNTRIES_URL)
  df=(df.withColumnRenamed("Code","Country").withColumnRenamed("Title","Country_Name").withColumnRenamed("ParentCode","Region").withColumnRenamed("ParentTitle","Region_Name").select("Country","Country_Name","Region","Region_Name"))
  df = df.select([F.col(col).alias(col.lower()) for col in df.columns])
  return write_delta_table(df,DELTA_TABLE_WHO_COUNTRIES,'overwrite')

def get_who_lex_data():
  spark.sparkContext.addFile(WHO_LEX_DATA)
  df=spark.read.csv("file://"+SparkFiles.get(WHO_LEX_DATA.split("/")[-1]), header=True, inferSchema= True)
  df = df.select([F.col(col).alias(col.strip().replace("  "," ").replace("/","_").replace("-","_").replace(' ', '_').lower()) for col in df.columns])
  return write_delta_table(df,DELTA_TABLE_WHO_LEX,'overwrite')

def get_world_population_data():
  spark.sparkContext.addFile(WORLD_POP_DATA)
  df=spark.read.csv("file://"+SparkFiles.get(WORLD_POP_DATA.split("/")[-1]), header=True, inferSchema= True)
  df = df.select([F.col(col).alias(col.strip().replace(',', '').replace('*', '').replace("  "," ").replace("/","_").replace("-","_").replace(' ', '_').lower()) for col in df.columns])
  return write_delta_table(df,DELTA_TABLE_WORLD_POP,'overwrite')


def get_wid_drug_deaths():
  spark.sparkContext.addFile(WID_DRUG_DATA)
  df=spark.read.csv("file://"+SparkFiles.get(WID_DRUG_DATA.split("/")[-1]), header=True, inferSchema= True)
  df=(df.withColumnRenamed("Deaths - Opioid use disorders - Sex: Both - Age: All Ages (Number)","deaths_opioid")
  .withColumnRenamed("Deaths - Cocaine use disorders - Sex: Both - Age: All Ages (Number)","deaths_cocaine")
  .withColumnRenamed("Deaths - Amphetamine use disorders - Sex: Both - Age: All Ages (Number)","deaths_amphetamine")
  .withColumnRenamed("Deaths - Other drug use disorders - Sex: Both - Age: All Ages (Number)","deaths_other")
  .withColumnRenamed("Code","Country").select("Country","Year","deaths_opioid","deaths_cocaine","deaths_amphetamine","deaths_other"))
  df = df.select([F.col(col).alias(col.lower()) for col in df.columns])
  return write_delta_table(df,DELTA_TABLE_WID_DRUG_DEATHS,'overwrite')


def load_sql_tables():
  def drop_make(table_name,path):
    spark.sql(f"""DROP TABLE IF EXISTS {table_name}""")
    spark.sql(f"""
    CREATE TABLE {table_name}
    USING DELTA
    LOCATION "{path}"
    """)
    print(f"Register {table_name} using path: {path}")
    return
    
  drop_make('e2e_lex_who_countries', DELTA_TABLE_WHO_COUNTRIES)
  #drop_make('e2e_lex_who_indicator_desc', DELTA_TABLE_WHO_INDICATOR_DESCRIPTIONS)
  drop_make('e2e_lex_wid_drug_deaths', DELTA_TABLE_WID_DRUG_DEATHS)
  drop_make('e2e_lex_who_features', DELTA_TABLE_WHO_LEX)
  drop_make('e2e_lex_world_population', DELTA_TABLE_WORLD_POP)
  return
  

def get_bronze_raw_data():
  get_who_countries()
  #get_who_indicator_descriptions()
  get_wid_drug_deaths()
  get_who_lex_data()
  get_world_population_data()
  #for indicator in WHO_INDICATORS:
  #  get_who_indicator(indicator)

  load_sql_tables()
  return

def delete_delta_tables():
  dbutils.fs.rm(BASE_DELTA_TABLE , recurse=True)

# COMMAND ----------


