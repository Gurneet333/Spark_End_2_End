# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow Project: Reproduce or Retrain GitHub Published Models

# COMMAND ----------

import mlflow

# COMMAND ----------

params_1 = {"alpha": 0.5}
git_uri = "https://github.com/mlflow/mlflow-example"

# COMMAND ----------

params_2 = {"batch_size": 20,  "epochs": 200}
git_uri_2 = "https://github.com/dmatrix/mlflow-workshop-project-expamle-1"

# COMMAND ----------

def run_project(uri, parameters=None):
    mlflow.projects.run(uri, parameters=parameters)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reproduce the first GitHub Project
# MAGIC 
# MAGIC The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# MAGIC 
# MAGIC P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# MAGIC 
# MAGIC **Problem**: Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
# MAGIC 
# MAGIC GitHub Project: https://github.com/mlflow/mlflow-example

# COMMAND ----------

run_project(git_uri, parameters=params_1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reproduce the second GitHub Project
# MAGIC This example demonstrates how you can package an MLflow Project into GitHub and share it with others to reproduce runs.
# MAGIC 
# MAGIC **Problem**: Build a simple Linear NN Model that predicts Celsius temperaturers from training data with Fahrenheit degree
# MAGIC 
# MAGIC GitHub Project: https://github.com/dmatrix/mlflow-workshop-project-expamle-1

# COMMAND ----------

mlflow.projects.run(git_uri_2, parameters=params_2)
