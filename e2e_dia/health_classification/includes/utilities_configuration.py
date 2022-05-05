# Databricks notebook source
from urllib.request import urlretrieve
import os

import pandas as pd
import numpy as np
import os
import mlflow

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pyspark.sql.functions import mean, col

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Data Paths

# COMMAND ----------

# Fill in your user name here....
username = "lpalum"
experiment_id = None

# COMMAND ----------

projectPath     = f"dbfs:/FileStore/tables/{username}/healthapp/"
landingPath     = projectPath + "landing/"
silverDailyPath = projectPath + "daily/"
dimUserPath     = projectPath + "users/"
goldPath        = projectPath + "gold/"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure Database

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP DATABASE IF EXISTS healthapp CASCADE;
# MAGIC CREATE DATABASE healthapp;
# MAGIC USE healthapp;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Utility functions

# COMMAND ----------

def retrieve_data(file: str) -> bool:
  """Download file from remote location to landing path."""
  base_url = "https://files.training.databricks.com/static/data/health-tracker/"
  urlretrieve(base_url + file, "/tmp/" + file)
  dbutils.fs.mv("file:/tmp/"+file,landingPath+file)
  return 

def load_delta_table(file: str, delta_table_path: str) -> bool:
  "Load a parquet file as a Delta table."
  parquet_df = spark.read.format("parquet").load(landingPath + file)
  parquet_df.write.format("delta").mode("overwrite").save(delta_table_path)
  return 

def process_file(file_name: str, path: str,  table_name: str) -> bool:
  """
  1. retrieve file
  2. load as delta table
  3. register table in the metastore
  """

  retrieve_data(file_name)
  print(f"Retrieve {file_name}.")

  load_delta_table(file_name, path)
  print(f"Load {file_name} to {path}")

  spark.sql(f"""
  DROP TABLE IF EXISTS {table_name}
  """)

  spark.sql(f"""
  CREATE TABLE {table_name}
  USING DELTA
  LOCATION "{path}"
  """)

  print(f"Register {table_name} using path: {path}")


# COMMAND ----------

def mlflow_run(experiment_id, estimator, param_grid, data):
    (X_train, X_test, y_train, y_test) = data

    with mlflow.start_run(experiment_id=experiment_id) as run:
        gs = GridSearchCV(estimator, param_grid)
        gs.fit(X_train, y_train)

        train_acc = gs.score(X_train, y_train)
        test_acc = gs.score(X_test, y_test)
        mlflow.log_param("model",
                         (str(estimator.__class__)
                          .split('.')[-1].replace("'>","")))

        mlflow.sklearn.log_model(gs.best_estimator_, "model")

        for param, value in gs.best_params_.items():
            mlflow.log_param(param, value)
        mlflow.log_metric("train acc", train_acc)
        mlflow.log_metric("test acc", test_acc)
        
def prepare_results(experiment_id):
    results = mlflow.search_runs(experiment_id)
    columns = [
      col for col in results.columns
      if any([
        'metric' in col,
        'param' in col,
        'artifact' in col
      ])
    ]
    return results[columns]

def prepare_coefs(experiments, lifestyles, feature_columns):
    # s3://mlflow/1/84e2e9717ce742bc8b99dc71af4388b1/artifacts/model
   
    models = [
      mlflow.sklearn.load_model(artifact_uri + "/model")
      for artifact_uri in experiments['artifact_uri'].values
    ]

    models = [
      {**model.get_params(),
        "coefs" : model.coef_
      } for model in models
    ]
    coefs = pd.DataFrame(models)
    coefs = coefs[["C", "l1_ratio", "penalty", "coefs"]]
    coefs["coefs"] = (
      coefs["coefs"]
      .apply(
        lambda artifact: [
          (lifestyle, coefs)
          for lifestyle, coefs
          in zip(lifestyles, artifact)
        ]
      )
    )
    coefs = coefs.explode("coefs")
    coefs["lifestyle"] = coefs["coefs"].apply(lambda artifact: artifact[0])
    coefs["coefs"] = coefs["coefs"].apply(lambda artifact: artifact[1])
    coefs.set_index(["C", "l1_ratio", "penalty", "lifestyle"], inplace=True)
    coefs = coefs["coefs"].apply(pd.Series)
    print(coefs.columns)

    coefs.columns = feature_columns
    ax = coefs.T.plot(figsize=(20,7))
    ax.set_xticks(range(len(coefs.columns)));
    ax.set_xticklabels(coefs.columns.tolist(), rotation=45)
    return coefs, ax
