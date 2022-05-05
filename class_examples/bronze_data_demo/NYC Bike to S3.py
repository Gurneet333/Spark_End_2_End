# Databricks notebook source
# MAGIC %md
# MAGIC ## Reterieve NYC Bike share station data and write to an s3 bucket
# MAGIC https://www.citibikenyc.com/system-data
# MAGIC - Station Information (dimensional data): https://gbfs.citibikenyc.com/gbfs/en/station_information.json
# MAGIC - Station Status Information: https://gbfs.citibikenyc.com/gbfs/en/station_status.json
# MAGIC 
# MAGIC <img src="https://data-science-at-scale.s3.amazonaws.com/images/e2e_applications/DIA+Framework-NYC+Bike+Status+Data.png">

# COMMAND ----------

!pip install tqdm

# COMMAND ----------

import pandas as pd
import time
from tqdm import tqdm

# Data URLS (configuration)
BIKE_STATION_STATUS_JSON="https://gbfs.citibikenyc.com/gbfs/en/station_status.json"
BIKE_STATION_JSON="https://gbfs.citibikenyc.com/gbfs/en/station_information.json"
BASE_PATH = "/mnt/dscc202-datasets/misc/bike-feed/"

# COMMAND ----------

# utility functions
# This routine will mount the classroom datasets in S3
def mount_datasets():
  mount_name = "dscc202-datasets"
  s3_bucket = "s3a://dscc202-datasets/"
  try:
    dbutils.fs.mount(s3_bucket, "/mnt/%s" % mount_name)
  except:
    print("already mounted")

  display(dbutils.fs.ls("/mnt/%s" % mount_name)  )
  
mount_datasets()

# remove the previous runs
dbutils.fs.rm(BASE_PATH, recurse=True)

# COMMAND ----------

# Grab the station dimension data
try:
    stationsDF = pd.read_json(BIKE_STATION_JSON)
    update_time = stationsDF['last_updated'].values[0]
    stationsDF = pd.json_normalize(stationsDF['data']['stations']).set_index("station_id").drop("legacy_id", axis=1)
    stationsDF.columns = [x.replace(".","_") for x in stationsDF.columns]
except:
    print("Unable to retreive station status")

# COMMAND ----------

last_update_time = update_time
for _ in tqdm(range(300)):
  # Grab the station status data
  try:
      statusDF = pd.read_json(BIKE_STATION_STATUS_JSON)
      update_time = statusDF['last_updated'].values[0]
      statusDF = pd.json_normalize(statusDF['data']['stations']).set_index("station_id")
  except:
      print("Unable to retreive station status")
  if update_time > last_update_time:
    df = statusDF.join(stationsDF)
    df['update_time'] = update_time
    df.reset_index(inplace=True)
    # write the data to S3 bucket with update time as the filename
    dbutils.fs.put(f"{BASE_PATH}{update_time}.json", df.to_json(orient='records'))

  last_update_time=update_time
  time.sleep(15)
