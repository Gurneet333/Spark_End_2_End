# Databricks notebook source
# MAGIC %md
# MAGIC ## Movio Health Device User Classification
# MAGIC - What is the objective? - Motivate improving health
# MAGIC - Augment/replace?  Optimize? - Augment the raw data colletion with a classification of life style
# MAGIC - Where is the data coming from? - IOT fitness device
# MAGIC - What is in the Data (metadata of the source)?
# MAGIC  - active heartrate
# MAGIC  - resting heartrate
# MAGIC  - BMI
# MAGIC  - VO2
# MAGIC  - workout minutes
# MAGIC  - lifestyle
# MAGIC  - occupation
# MAGIC  - sex

# COMMAND ----------

# MAGIC %md
# MAGIC # End to End Application Example
# MAGIC - Data Engineering
# MAGIC - ML Model Development
# MAGIC - Application deployment
# MAGIC 
# MAGIC <img src="https://data-science-at-scale.s3.amazonaws.com/data-science-at-scale/images/DIA+Framework-3.png" style="width:40%;height:auto;">

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Engineering
# MAGIC - Enable access to the data
# MAGIC - Make it reliable for application
# MAGIC 
# MAGIC <img src="https://data-science-at-scale.s3.amazonaws.com/data-science-at-scale/images/delta_lake_pipeline_1.png" style="width:40%;height:auto;">

# COMMAND ----------

# General imports
import os
import pandas as pd
import numpy as np

from random import random, randint
from urllib.request import urlretrieve

# Data Engineering
from pyspark.sql.functions import mean, col
from pyspark.sql.types import _parse_datatype_string

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from mlflow import mlflow,log_metric, log_param, log_artifacts

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

%matplotlib inline

# COMMAND ----------

# MAGIC %run ./includes/utilities_configuration

# COMMAND ----------

## This routine reads the raw parquet files and 
## establishes delta tables with associated hive metastore table names
process_file(
  "health_profile_data.snappy.parquet",
  silverDailyPath,
  "e2e_health_profile_data"
)
process_file(
  "user_profile_data.snappy.parquet",
  dimUserPath,
  "e2e_health_user_profile_data"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read the user and health profile data from the silver delta tables

# COMMAND ----------

user_profile_df = spark.read.table("e2e_health_user_profile_data")
health_profile_df = spark.read.table("e2e_health_profile_data")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate a sample of the users

# COMMAND ----------

user_profile_sample_df = user_profile_df.sample(0.1)

user_profile_sample_df.groupby("lifestyle").count().show()

# COMMAND ----------

# TODO Join the user profile sample with the health data
health_profile_sample_df = (
  user_profile_sample_df
  .join(health_profile_df, "_id")
)

# check to be sure that sample count is consistent with the joined data size
assert 365*user_profile_sample_df.count() == health_profile_sample_df.count()

display(health_profile_sample_df)

# COMMAND ----------

# generate the aggregation of health data for each user in the dataset

health_tracker_sample_agg_df = (
    health_profile_sample_df.groupBy("_id")
    .agg(
        mean("BMI").alias("mean_BMI"),
        mean("active_heartrate").alias("mean_active_heartrate"),
        mean("resting_heartrate").alias("mean_resting_heartrate"),
        mean("VO2_max").alias("mean_VO2_max"),
        mean("workout_minutes").alias("mean_workout_minutes")
    )
)

display(health_tracker_sample_agg_df)

# COMMAND ----------

# Join with the user data and print the schema of the aggregated and joined data

health_tracker_augmented_df = (
  health_tracker_sample_agg_df
  .join(user_profile_df, "_id")
)

health_tracker_augmented_df.printSchema()

# COMMAND ----------

# Check the augmented data schema

augmented_schema = """
  mean_BMI double,
  mean_active_heartrate double,
  mean_resting_heartrate double,
  mean_VO2_max double,
  mean_workout_minutes double,
  female boolean,
  country string,
  occupation string,
  lifestyle string
"""

health_tracker_augmented_df = (health_tracker_augmented_df.select(["mean_BMI",
                                                                   "mean_active_heartrate",
                                                                   "mean_resting_heartrate",
                                                                   "mean_VO2_max",
                                                                   "mean_workout_minutes",
                                                                   "female",
                                                                   "country",
                                                                   "occupation",
                                                                   "lifestyle"
                                                                  ]))



assert health_tracker_augmented_df.schema == _parse_datatype_string(augmented_schema)

display(health_tracker_augmented_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write the augmented data to the delta table

# COMMAND ----------

(
  health_tracker_augmented_df.write
  .format("delta")
  .mode("overwrite")
  .save(goldPath + "health_tracker_augmented")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read the augmented (gold) data from the delata lake

# COMMAND ----------

# note: conversion to pandas in preparation for machine learning.

health_tracker_augmented_df = (
  spark.read
  .format("delta")
  .load(goldPath + "health_tracker_augmented")
  .toPandas()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Science
# MAGIC - Join and aggregate the data
# MAGIC - feature development
# MAGIC - Explore the data (EDA)
# MAGIC - Experiment with modeling techniques
# MAGIC - Tune
# MAGIC 
# MAGIC <img src="https://data-science-at-scale.s3.amazonaws.com/data-science-at-scale/images/mlflow-infrastructure-for-a-complete-machine-learning-life-cycle-4-638.jpg" style="width:auto;height:50%;">

# COMMAND ----------

# enumerate the lifestyles in the augmented dataset
lifestyles = health_tracker_augmented_df.lifestyle.unique()
lifestyles

# COMMAND ----------

# form the feature and target datasets from the augmented dataframe

features = health_tracker_augmented_df.drop("lifestyle", axis=1)
target = health_tracker_augmented_df[["lifestyle"]].copy()

# COMMAND ----------

# Take a look at the features (sample)
features.sample(10)

# COMMAND ----------

# illuminate the datatypes to form category and numeric feature lists
features.dtypes

# COMMAND ----------

features_numerical = features.select_dtypes(include=[float])
features_categorical = features.select_dtypes(exclude=[float])

# COMMAND ----------

# MAGIC %md
# MAGIC # Explore the Data before training

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distributions of numerical features

# COMMAND ----------

fig, ax = plt.subplots(1,5, figsize=(25,5))

for i, feature in enumerate(features_numerical):
  sns.histplot(features[feature], ax=ax[i])
  ax[i].set_xlim(0,250)
  
display(fig.show())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distributions of the numerical features after standardization

# COMMAND ----------

fig, ax = plt.subplots(1,5, figsize=(25,5))

for i, feature in enumerate(features_numerical):
  feature_series = features[feature]
  feature_scaled = (feature_series - feature_series.mean())/feature_series.std()
  sns.histplot(feature_scaled, ax=ax[i])
  ax[i].set_xlim(-5, 5)

display(fig.show())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distributions of the numberical features categorized by lifestyle

# COMMAND ----------

features_numerical

# COMMAND ----------

fig, ax = plt.subplots(1,5, figsize=(25,5))
for i, feature in enumerate(features_numerical):
  for lifestyle in lifestyles:
    subset = features[target["lifestyle"] == lifestyle]
    sns.histplot(subset[feature], ax=ax[i], label=lifestyle)
  ax[i].legend()

display(fig.show())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Explore mean resting heartrate across catigorical features

# COMMAND ----------

fig, ax = plt.subplots(1,3, figsize=(27,5))

for i, feature in enumerate(features_categorical):
  for value in features[feature].unique():
    subset = features[features[feature] == value]
    sns.histplot(subset["mean_resting_heartrate"], ax=ax[i], label=value)
  ax[i].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

display(fig.show())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Explore at target variable distribution over the categorical features

# COMMAND ----------

fig, ax = plt.subplots(1, 3, figsize=(27,5))
(
  health_tracker_augmented_df
  .groupby("female")
  .lifestyle.value_counts()
  .unstack(0)
  .plot(kind="bar", ax=ax[0]).legend(loc='center left',bbox_to_anchor=(1.0, 0.5))
)
(
  health_tracker_augmented_df
  .groupby("country")
  .lifestyle.value_counts()
  .unstack(0)
  .plot(kind="bar", ax=ax[1]).legend(loc='center left',bbox_to_anchor=(1.0, 0.5))
)
(
  health_tracker_augmented_df
  .groupby("occupation")
  .lifestyle.value_counts()
  .unstack(0)
  .plot(kind="bar", ax=ax[2]).legend(loc='center left',bbox_to_anchor=(1.0, 0.5))
)

display(fig.show())

# COMMAND ----------

pd.get_dummies(features_categorical)

# COMMAND ----------


ohe = OneHotEncoder(sparse=False, drop=None, handle_unknown='ignore')
ohe.fit_transform(features_categorical)

# COMMAND ----------

# MAGIC %md
# MAGIC # Building a Linear Customer Classfication Model

# COMMAND ----------

from sklearn.preprocessing import LabelEncoder

features = health_tracker_augmented_df.drop("lifestyle", axis=1)
target = health_tracker_augmented_df["lifestyle"]
le = LabelEncoder()
target = le.fit_transform(target)

# COMMAND ----------

# Perform Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, target)

# COMMAND ----------

# Split Data into Numerical and Categorical Sets
X_train_numerical = X_train.select_dtypes(exclude=["object"])
X_test_numerical = X_test.select_dtypes(exclude=["object"])
X_train_categorical = X_train.select_dtypes(include=["object"])
X_test_categorical = X_test.select_dtypes(include=["object"])

# COMMAND ----------

ss = StandardScaler()
ohe = OneHotEncoder(sparse=False, drop=None, handle_unknown='ignore')

# Create One-Hot Encoded Categorical DataFrames
X_train_ohe = pd.DataFrame(
  ohe.fit_transform(X_train_categorical),
  columns=ohe.get_feature_names(),
  index=X_train_numerical.index
)
X_test_ohe = pd.DataFrame(
  ohe.transform(X_test_categorical),
  columns=ohe.get_feature_names(),
  index=X_test_numerical.index
)

# COMMAND ----------

# Merge Numerical and One-Hot Encoded Categorical
X_train = X_train_numerical.merge(X_train_ohe, left_index=True, right_index=True)
X_test = X_test_numerical.merge(X_test_ohe, left_index=True, right_index=True)

# COMMAND ----------

# Standardize Data
X_train = pd.DataFrame(
  ss.fit_transform(X_train),
  index=X_train_ohe.index,
  columns=X_train.columns)
X_test = pd.DataFrame(
  ss.transform(X_test),
  index=X_test_ohe.index,
  columns=X_train.columns)

# COMMAND ----------

data = (X_train, X_test, y_train, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Grid-Searched Model Fitting
# MAGIC The following models were fit using a grid-searched, cross validation with the respective parameter dictionaries:
# MAGIC 
# MAGIC Regularization method on the logistic regression coefficients:

# COMMAND ----------

# MAGIC %md
# MAGIC ### Lasso - Least Absolute Shrinkage Selector Operator
# MAGIC - L = ∑( Ŷi– Yi)² + λ∑ |β|
# MAGIC - {'alpha' : logspace(-5,5,11)}

# COMMAND ----------

estimator = LogisticRegression(max_iter=10000)
param_grid = {
  'C' : np.logspace(-5,5,11),
  "penalty" : ['l2']
}
mlflow_run(experiment_id, estimator, param_grid, data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ridge 
# MAGIC - L = ∑( Ŷi– Yi)² + λ∑ β²
# MAGIC - {'alpha' : logspace(-5,5,11)}

# COMMAND ----------

estimator = LogisticRegression(max_iter=10000)
param_grid = {
  'C' : np.logspace(-5,5,11),
  "penalty" : ['l1'], "solver" : ['saga']
}
mlflow_run(experiment_id, estimator, param_grid, data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Elastic Net
# MAGIC - L = ∑( Ŷi– Yi)² + λ∑ β² + λ∑ |β|
# MAGIC - {'alpha' : logspace(-5,5,11), 'l1_ratio' : linspace(0,1,11)}

# COMMAND ----------

estimator = LogisticRegression(max_iter=100000)
param_grid = {
  'C' : np.logspace(-5,5,11),
  "penalty" : ['elasticnet'],
  'l1_ratio' : np.linspace(0,1,11),
  "solver" : ['saga']
}
mlflow_run(experiment_id, estimator, param_grid, data)

# COMMAND ----------

experiments = mlflow.search_runs()
prepare_coefs(experiments, le.classes_, X_train.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC # Productionization
# MAGIC - Package model for production
# MAGIC - Execute inference at Scale
# MAGIC - Monitor Models
# MAGIC - Operate Data Pipelines at Scale

# COMMAND ----------

# Pull this from the UI
logged_model = experiments[experiments["metrics.test acc"] == experiments["metrics.test acc"].max()].artifact_uri.values[0]+"/model"

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
accuracy_score(y_test,loaded_model.predict(X_test))

# COMMAND ----------

# MAGIC %md
# MAGIC <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img src="https://drive.google.com/uc?export=view&id=1ILbthTP3pUJMrbN0HawvChIXCnyM9-l1" alt="Data Science at Scale" style=" width: 250px; height: auto;"></a>
# MAGIC 
# MAGIC <a href="mailto:lpalum@gmail.com">lpalum at gmail</a>
