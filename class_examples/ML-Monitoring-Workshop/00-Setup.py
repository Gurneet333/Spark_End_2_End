# Databricks notebook source
DB_NAME = "g00"
print(DB_NAME)

# COMMAND ----------

# MAGIC %scala
# MAGIC val DB_NAME = "g00"
# MAGIC println(DB_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC **Create the folders for the Experiment**  
# MAGIC   
# MAGIC You can use your `\Shared` folder in the workspace to create a folder for to house the experiment. Do this and then specify the folder path to the notebook.

# COMMAND ----------

PATH_TO_MLFLOW_EXPERIMENT = "/Shared/experiment_created"
print(PATH_TO_MLFLOW_EXPERIMENT)

# COMMAND ----------


