# Databricks notebook source
# MAGIC %md #### Regression Model
# MAGIC 
# MAGIC We want to predict the gas consumption (in millions of gallons/year) in 48 of the US states
# MAGIC based on some key features. 
# MAGIC 
# MAGIC These features are 
# MAGIC  * petrol tax (in cents); 
# MAGIC  * per capital income (in US dollars);
# MAGIC  * paved highway (in miles); and
# MAGIC  * population of people with driving licences
# MAGIC 
# MAGIC  
# MAGIC #### Solution:
# MAGIC 
# MAGIC Since this is a regression problem where the value is a range of numbers, we can use the
# MAGIC common Random Forest Algorithm in Scikit-Learn. Most regression models are evaluated with
# MAGIC four [standard evalution metrics](https://medium.com/usf-msds/choosing-the-right-metric-for-machine-learning-models-part-1-a99d7d7414e4): 
# MAGIC 
# MAGIC * Mean Absolute Error (MAE)
# MAGIC * Mean Squared Error (MSE)
# MAGIC * Root Mean Squared Error (RSME)
# MAGIC * R-squared (r2)
# MAGIC 
# MAGIC This example is borrowed from this [source](https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/) and modified and modularized for this tutorial
# MAGIC 
# MAGIC Aim of this this:
# MAGIC 
# MAGIC 1. Understand MLflow Tracking API
# MAGIC 2. How to use the MLflow Tracking API
# MAGIC 3. Use the MLflow API to experiment several Runs
# MAGIC 4. Interpret and observe runs via the MLflow UI
# MAGIC 
# MAGIC Some Resources:
# MAGIC * https://mlflow.org/docs/latest/python_api/mlflow.html
# MAGIC * https://www.saedsayad.com/decision_tree_reg.htm
# MAGIC * https://towardsdatascience.com/understanding-random-forest-58381e0602d2
# MAGIC * https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
# MAGIC * https://towardsdatascience.com/regression-an-explanation-of-regression-metrics-and-what-can-go-wrong-a39a9793d914
# MAGIC * https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/

# COMMAND ----------

# MAGIC %md Define all the classes and bring them into scope

# COMMAND ----------

# MAGIC %run ./setup/class_setup

# COMMAND ----------

# MAGIC %md ### Load the Dataset

# COMMAND ----------

# Load and print dataset
dataset = Utils.load_data("https://raw.githubusercontent.com/dmatrix/tmls-workshop/master/tracking/data/petrol_consumption.csv")
dataset.head(5)

# COMMAND ----------

# MAGIC %md Get descriptive statistics for the features

# COMMAND ----------

dataset.describe().transpose()

# COMMAND ----------

# MAGIC %md ## Let's create an explicit experiment for our work

# COMMAND ----------


from mlflow.tracking import MlflowClient

# Create an experiment with a name that is unique and case sensitive.
client = MlflowClient()
experiment_id = client.create_experiment("/Shared/class_examples/MLops/MLflow-Tracking/Tracking-Example-Experiment")
client.set_experiment_tag(experiment_id, "sklearn", "RFregression")

# Fetch experiment metadata information
experiment = client.get_experiment(experiment_id)
print("Name: {}".format(experiment.name))
print("Experiment_id: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Tags: {}".format(experiment.tags))
print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

# COMMAND ----------

# Iterate over several runs with different parameters, such as number of trees. 
# For excercises, try changing max_depth, number of estimators, and consult the documentation what other tunning parameters
# may affect a better outcome and supply them to the class constructor
max_depth = 0
for n in range (75, 200, 25):
  max_depth = max_depth + 2
  params = {"n_estimators": n, "max_depth": max_depth}
  rfr = RFRModel.new_instance(params)
  (experimentID, runID) = rfr.mlflow_run(dataset, run_name="Regression Petrol Consumption Model", verbose=True)
  print("MLflow Run completed with run_id {} and experiment_id {}".format(runID, experimentID))
  print("-" * 100)

# COMMAND ----------

# MAGIC %md ### Let's Explore the MLflow UI
# MAGIC 
# MAGIC * Add Notes & Tags
# MAGIC * Compare Runs pick two best runs
# MAGIC * Annotate with descriptions and tags
# MAGIC * Evaluate the best run
