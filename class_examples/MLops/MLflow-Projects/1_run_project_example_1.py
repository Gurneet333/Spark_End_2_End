# Databricks notebook source
# MAGIC %md ### MLflow Project Example
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><td>
# MAGIC     <img src="https://raw.githubusercontent.com/dmatrix/mlflow-workshop-part-2/master/images/project.png"
# MAGIC          alt="Bank Note " width="400">
# MAGIC   </td></tr>
# MAGIC </table>
# MAGIC 
# MAGIC An MLflow Project is a format for packaging data science code in a reusable and reproducible way, based primarily on conventions. 
# MAGIC 
# MAGIC In addition, the Projects component includes an API and command-line tools for running projects, making it possible to chain together projects into workflows.
# MAGIC You can run projects as:
# MAGIC 
# MAGIC * From command line: ```mlflow run git://<my_project> -P <arg>=<value> ... -P <arg>=<value>```
# MAGIC * In GitHub Repo: ``` cd <gitbub_project_directory>; mlflow run . -e <entry_point> -P <arg>=<value> ... -P <arg>=<value>```
# MAGIC * Programmatically: ``` mlflow.run("git://<my_project>", parameters={'arg':value, 'arg':value})```
# MAGIC * Programmatically: ``` mlflow.projects.run("git://<my_project>", parameters={'arg':value, 'arg':value})```

# COMMAND ----------

# MAGIC %md ### What's does a MLflow Project Look Like?
# MAGIC 
# MAGIC 
# MAGIC [MLflow Project Example](https://github.com/mlflow/mlflow-example)
# MAGIC  * MLProject
# MAGIC  * conda.yaml
# MAGIC  * code ...
# MAGIC  * data

# COMMAND ----------

import mlflow
import warnings
from mlflow import projects
print(mlflow.__version__)

# COMMAND ----------

# MAGIC %md Define arguments for alpha

# COMMAND ----------

parameters = [{'alpha': 0.3}]
ml_project_uri = "https://github.com/mlflow/mlflow-example"

# COMMAND ----------

# MAGIC %md Use MLflow Project API
# MAGIC  * [mlflow.projects.run(...)](https://mlflow.org/docs/latest/python_api/mlflow.projects.html#mlflow.projects.run)
# MAGIC  * Returns [SubmittedRun](https://mlflow.org/docs/latest/python_api/mlflow.projects.html#mlflow.projects.SubmittedRun) object

# COMMAND ----------

warnings.filterwarnings("ignore", category=DeprecationWarning)
# Iterate over three different runs with different parameters
for param in parameters:
  print(f"Running with param = {param}"),
  res_sub = projects.run(ml_project_uri, parameters=param)
  print(f"status={res_sub.get_status()}")
  print(f"run_id={res_sub.run_id}")

# COMMAND ----------

# MAGIC %md ### Check the MLflow UI
# MAGIC  * Check for conda.yaml
# MAGIC  * Check Metrics, parameters, artifacts, etc

# COMMAND ----------

# MAGIC %md ### Homework Assignment. Try different runs with:
# MAGIC * Change or add parameters `alpha`values
# MAGIC * Check in MLfow UI if the metrics are affected
# MAGIC * Add Notes & Tags
# MAGIC * Compare Runs pick two best runs
# MAGIC * Annotate with descriptions 
# MAGIC * Try running [Keras Project Example](https://github.com/dmatrix/mlflow-workshop-project-expamle-1): A simple Linear NN Model that predicts Celsius temperaturers from training data with Fahrenheit degree
# MAGIC   * Try running examples in the PyTorch equivalent in`extras` directory folder

# COMMAND ----------

project_uri = 'https://github.com/dmatrix/mlflow-workshop-project-expamle-1'

# COMMAND ----------

res_sub = projects.run(project_uri)
print(f"status={res_sub.get_status()}")
print(f"run_id={res_sub.run_id}")
