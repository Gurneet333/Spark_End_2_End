# Databricks notebook source
dbutils.fs.ls("/databricks-datasets/COVID/covid-19-data/us.csv")

# COMMAND ----------

df=spark.read.format("csv").option("header", True).load("/databricks-datasets/COVID/covid-19-data/us.csv")

# COMMAND ----------

display(df)

# COMMAND ----------

from pathlib import Path

import numpy as np
import pandas as pd
import requests

import pandas_profiling
from pandas_profiling.utils.cache import cache_file

# COMMAND ----------

displayHTML(pandas_profiling.ProfileReport(df.toPandas()).html)

# COMMAND ----------

file_name = cache_file(
    "meteorites.csv",
    "https://data.nasa.gov/api/views/gh4g-9sfh/rows.csv?accessType=DOWNLOAD",
)

df = pd.read_csv(file_name)

df.head()

# COMMAND ----------

# Note: Pandas does not support dates before 1880, so we ignore these for this analysis
df["year"] = pd.to_datetime(df["year"], errors="coerce")

# Example: Constant variable
df["source"] = "NASA"

# Example: Boolean variable
df["boolean"] = np.random.choice([True, False], df.shape[0])

# Example: Mixed with base types
df["mixed"] = np.random.choice([1, "A"], df.shape[0])

# Example: Highly correlated variables
df["reclat_city"] = df["reclat"] + np.random.normal(scale=5, size=(len(df)))

# Example: Duplicate observations
duplicates_to_add = pd.DataFrame(df.iloc[0:10])
duplicates_to_add["name"] = duplicates_to_add["name"] + " copy"

df = df.append(duplicates_to_add, ignore_index=True)

df.head()

# COMMAND ----------

report = df.profile_report(
    sort="None", html={"style": {"full_width": True}}, progress_bar=False
)
displayHTML(report.html)

# COMMAND ----------


