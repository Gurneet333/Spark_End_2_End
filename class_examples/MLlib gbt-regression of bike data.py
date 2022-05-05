# Databricks notebook source
# MAGIC %md # Regression with gradient-boosted trees and MLlib pipelines
# MAGIC 
# MAGIC This notebook uses a bike-sharing dataset to illustrate MLlib pipelines and the gradient-boosted trees machine learning algorithm. The challenge is to predict the number of bicycle rentals per hour based on the features available in the dataset such as day of the week, weather, season, and so on. Demand prediction is a common problem across businesses; good predictions allow a business or service to optimize inventory and to match supply and demand to make customers happy and maximize profitability. 

# COMMAND ----------

# MAGIC %md ## Load the dataset
# MAGIC The dataset is from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset) and is provided with Databricks Runtime. The dataset includes information about bicycle rentals from the Capital bikeshare system in 2011 and 2012. 
# MAGIC 
# MAGIC Load the data using the CSV datasource for Spark, which creates a [Spark DataFrame](http://spark.apache.org/docs/latest/sql-programming-guide.html).  

# COMMAND ----------

# Setup the partitions and turn on AQE
spark.conf.set("spark.sql.shuffle.partitions", "32")  # keep the size of shuffles to cores default shuffle partitions of 200 will create "thrash"
spark.conf.set("spark.sql.adaptive.enabled", "true") # Spark 3.0 AQE - coalescing post-shuffle partitions, converting sort-merge join to broadcast join, and skew join optimization

df = spark.read.csv("/databricks-datasets/bikeSharing/data-001/hour.csv", header="true", inferSchema="true")
# The following command caches the DataFrame in memory. This improves performance since subsequent calls to the DataFrame can read from memory instead of re-reading the data from disk.
df.cache()

# COMMAND ----------

# MAGIC %md #### Data description
# MAGIC 
# MAGIC The following columns are included in the dataset:
# MAGIC 
# MAGIC **Index column**:
# MAGIC * instant: record index
# MAGIC 
# MAGIC **Feature columns**:
# MAGIC * dteday: date
# MAGIC * season: season (1:spring, 2:summer, 3:fall, 4:winter)
# MAGIC * yr: year (0:2011, 1:2012)
# MAGIC * mnth: month (1 to 12)
# MAGIC * hr: hour (0 to 23)
# MAGIC * holiday: 1 if holiday, 0 otherwise
# MAGIC * weekday: day of the week (0 to 6)
# MAGIC * workingday: 0 if weekend or holiday, 1 otherwise
# MAGIC * weathersit: (1:clear, 2:mist or clouds, 3:light rain or snow, 4:heavy rain or snow)  
# MAGIC * temp: normalized temperature in Celsius  
# MAGIC * atemp: normalized feeling temperature in Celsius  
# MAGIC * hum: normalized humidity  
# MAGIC * windspeed: normalized wind speed  
# MAGIC 
# MAGIC **Label columns**:
# MAGIC * casual: count of casual users
# MAGIC * registered: count of registered users
# MAGIC * cnt: count of total rental bikes including both casual and registered

# COMMAND ----------

# MAGIC %md Call `display()` on a DataFrame to see a sample of the data. The first row shows that 16 people rented bikes between midnight and 1am on January 1, 2011. 

# COMMAND ----------

display(df)

# COMMAND ----------

print("The dataset has %d rows." % df.count())

# COMMAND ----------

# MAGIC %md ## Preprocess data
# MAGIC This dataset is well prepared for machine learning algorithms. The numeric input columns (temp, atemp, hum, and windspeed) are normalized, categorial values (season, yr, mnth, hr, holiday, weekday, workingday, weathersit) are converted to indices, and all of the columns except for the date (`dteday`) are numeric.
# MAGIC 
# MAGIC The goal is to predict the count of bike rentals (the `cnt` column). Reviewing the dataset, you can see that some columns contain duplicate information. For example, the `cnt` column equals the sum of the `casual` and `registered` columns. You should remove the `casual` and `registered` columns from the dataset. The index column `instant` is also not useful as a predictor.
# MAGIC 
# MAGIC You can also delete the column `dteday`, as this information is already included in the other date-related columns `yr`, `mnth`, and `weekday`. 

# COMMAND ----------

df = df.drop("instant").drop("dteday").drop("casual").drop("registered")
display(df)

# COMMAND ----------

# MAGIC %md Print the dataset schema to see the type of each column.

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %md #### Split data into training and test sets
# MAGIC 
# MAGIC Randomly split data into training and test sets. By doing this, you can train and tune the model using only the training subset, and then evaluate the model's performance on the test set to get a sense of how the model will perform on new data. 

# COMMAND ----------

# Split the dataset randomly into 70% for training and 30% for testing. Passing a seed for deterministic behavior
train, test = df.randomSplit([0.7, 0.3], seed = 0)
print("There are %d training examples and %d test examples." % (train.count(), test.count()))

# COMMAND ----------

# MAGIC %md #### Visualize the data
# MAGIC You can plot the data to explore it visually. The following plot shows the number of bicycle rentals during each hour of the day.  As you might expect, rentals are low during the night, and peak at commute hours.  
# MAGIC 
# MAGIC To create plots, call `display()` on a DataFrame in Databricks and click the plot icon below the table.
# MAGIC 
# MAGIC To create the plot shown, run the command in the following cell. The results appear in a table. From the drop-down menu below the table, select "Line". Click **Plot Options...**. In the dialog, drag `hr` to the **Keys** field, and drag `cnt` to the **Values** field. Also in the **Keys** field, click the "x" next to `<id>` to remove it. In the **Aggregation** drop down, select "AVG". 

# COMMAND ----------

display(train.select("hr", "cnt"))

# COMMAND ----------

# MAGIC %md ## Train the machine learning pipeline
# MAGIC 
# MAGIC Now that you have reviewed the data and prepared it as a DataFrame with numeric values, you're ready to train a model to predict future bike sharing rentals. 
# MAGIC 
# MAGIC Most MLlib algorithms require a single input column containing a vector of features and a single target column. The DataFrame currently has one column for each feature. MLlib provides functions to help you prepare the dataset in the required format.
# MAGIC 
# MAGIC MLlib pipelines combine multiple steps into a single workflow, making it easier to iterate as you develop the model. 
# MAGIC 
# MAGIC In this example, you create a pipeline using the following functions:
# MAGIC * `VectorAssembler`: Assembles the feature columns into a feature vector.
# MAGIC * `VectorIndexer`: Identifies columns that should be treated as categorical. This is done heuristically, identifying any column with a small number of distinct values as categorical. In this example, the following columns are considered categorical: `yr` (2 values), `season` (4 values), `holiday` (2 values), `workingday` (2 values), and `weathersit` (4 values).
# MAGIC * `GBTRegressor`: Uses the [Gradient-Boosted Trees (GBT)](https://spark.apache.org/docs/latest/ml-classification-regression.html#gradient-boosted-tree-classifier) algorithm to learn how to predict rental counts from the feature vectors.
# MAGIC * `CrossValidator`: The GBT algorithm has several hyperparameters. This notebook illustrates how to use [hyperparameter tuning in Spark](https://spark.apache.org/docs/latest/ml-tuning.html). This capability automatically tests a grid of hyperparameters and chooses the best resulting model.
# MAGIC 
# MAGIC For more information:  
# MAGIC [VectorAssembler](https://spark.apache.org/docs/latest/ml-features.html#vectorassembler)  
# MAGIC [VectorIndexer](https://spark.apache.org/docs/latest/ml-features.html#vectorindexer)  

# COMMAND ----------

# MAGIC %md The first step is to create the VectorAssembler and VectorIndexer steps. 

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, VectorIndexer

# Remove the target column from the input feature set.
featuresCols = df.columns
featuresCols.remove('cnt')

# vectorAssembler combines all feature columns into a single feature vector column, "rawFeatures".
vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol="rawFeatures")

# vectorIndexer identifies categorical features and indexes them, and creates a new column "features". 
vectorIndexer = VectorIndexer(inputCol="rawFeatures", outputCol="features", maxCategories=4)

# COMMAND ----------

# MAGIC %md Next, define the model.

# COMMAND ----------

from pyspark.ml.regression import GBTRegressor
# The next step is to define the model training stage of the pipeline. 
# The following command defines a GBTRegressor model that takes an input column "features" by default and learns to predict the labels in the "cnt" column. 
gbt = GBTRegressor(labelCol="cnt")

# COMMAND ----------

# MAGIC %md The third step is to wrap the model you just defined in a `CrossValidator` stage. `CrossValidator` calls the GBT algorithm with different hyperparameter settings. It trains multiple models and selects the best one, based on minimizing a specified metric. In this example, the metric is [root mean squared error (RMSE)](https://en.wikipedia.org/wiki/Root-mean-square_deviation).

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

# Define a grid of hyperparameters to test:
#  - maxDepth: maximum depth of each decision tree 
#  - maxIter: iterations, or the total number of trees 
paramGrid = ParamGridBuilder()\
  .addGrid(gbt.maxDepth, [2, 5])\
  .addGrid(gbt.maxIter, [10, 100])\
  .build()

# Define an evaluation metric.  The CrossValidator compares the true labels with predicted values for each combination of parameters, and calculates this value to determine the best model.
evaluator = RegressionEvaluator(metricName="rmse", labelCol=gbt.getLabelCol(), predictionCol=gbt.getPredictionCol())

# Declare the CrossValidator, which performs the model tuning.
cv = CrossValidator(estimator=gbt, evaluator=evaluator, estimatorParamMaps=paramGrid)

# COMMAND ----------

# MAGIC %md Create the pipeline.

# COMMAND ----------

from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[vectorAssembler, vectorIndexer, cv])

# COMMAND ----------

# MAGIC %md Train the pipeline.
# MAGIC 
# MAGIC Now that you have set up the workflow, you can train the pipeline with a single call.  
# MAGIC When you call `fit()`, the pipeline runs feature processing, model tuning, and training and returns a fitted pipeline with the best model it found.
# MAGIC This step takes several minutes.

# COMMAND ----------

import mlflow
import mlflow.spark

# turn on autologging
mlflow.spark.autolog()

with mlflow.start_run():
  pipelineModel = pipeline.fit(train)
  
  # Log the best model.
  mlflow.spark.log_model(spark_model=pipelineModel.stages[2].bestModel, artifact_path='best-model') 

# COMMAND ----------

# MAGIC %md ## Make predictions and evaluate results
# MAGIC 
# MAGIC The final step is to use the fitted model to make predictions on the test dataset and evaluate the model's performance. The model's performance on the test dataset provides an approximation of how it is likely to perform on new data. For example, if you had weather predictions for the next week, you could predict bike rentals expected during the next week.
# MAGIC 
# MAGIC Computing evaluation metrics is important for understanding the quality of predictions, as well as for comparing models and tuning parameters.

# COMMAND ----------

# MAGIC %md The `transform()` method of the pipeline model applies the full pipeline to the input dataset. The pipeline applies the feature processing steps to the dataset and then uses the fitted GBT model to make predictions. The pipeline returns a DataFrame with a new column `predictions`.

# COMMAND ----------

predictions = pipelineModel.transform(test)

# COMMAND ----------

display(predictions.select("cnt", "prediction", *featuresCols))

# COMMAND ----------

# MAGIC %md A common way to evaluate the performance of a regression model is the calculate the [root mean squared error (RMSE)](https://spark.apache.org/docs/latest/mllib-evaluation-metrics.html#regression-model-evaluation). The value is not very informative on its own, but you can use it to compare different models. `CrossValidator` determines the best model by selecting the one that minimizes RMSE. 

# COMMAND ----------

rmse = evaluator.evaluate(predictions)
print("RMSE on our test set: %g" % rmse)

# COMMAND ----------

# MAGIC %md You can also plot the results, as you did the original dataset. In this case, the hourly count of rentals shows a similar shape.

# COMMAND ----------

A = test.withColumnRenamed("hr","test_hr").select("test_hr","cnt")
B=A.join(predictions.select("hr", "prediction"), A.test_hr == predictions.hr, how="left").drop("test_hr")

# COMMAND ----------

B.toPandas().groupby("hr").mean().plot(figsize=(11,8))

# COMMAND ----------

# MAGIC %md It's also a good idea to examine the residuals, or the difference between the expected result and the predicted value. The residuals should be randomly distributed; if there are any patterns in the residuals, the model may not be capturing something important. In this case, the average residual is about 1, less than 1% of the average value of the `cnt` column. 

# COMMAND ----------

import pyspark.sql.functions as F
predictions_with_residuals = predictions.withColumn("residual", (F.col("cnt") - F.col("prediction")))
display(predictions_with_residuals.agg({'residual': 'mean'}))

# COMMAND ----------

# MAGIC %md Plot the residuals across the hours of the day to look for any patterns. In this example, there are no obvious correlations.

# COMMAND ----------

display(predictions_with_residuals.select("hr", "residual"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Now with SKlearn using Hyperopt and Spark Trials
# MAGIC Here are the steps in a Hyperopt workflow:
# MAGIC 
# MAGIC - Define a function to minimize.
# MAGIC - Define a search space over hyperparameters.
# MAGIC - Select a search algorithm.
# MAGIC - Run the tuning algorithm with Hyperopt fmin().
# MAGIC - Reference for the tuning process: https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
 
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials
 


# COMMAND ----------

X=df.toPandas().drop('cnt', axis=1)
y=df.toPandas()['cnt']

# COMMAND ----------

X.printSchema()

# COMMAND ----------

def objective(params):
    # Create a gradient boosting regression model
    gbt = GradientBoostingRegressor(n_estimators=int(params['n_estimators']),max_depth=int(params['max_depth']))
    
    # Use the cross-validation accuracy to compare the models' performance
    error = cross_val_score(gbt, X, y).mean()
    
    # Hyperopt tries to minimize the objective function. A higher accuracy value means a better model, so you must return the negative accuracy.
    return {'loss': error, 'status': STATUS_OK}

# COMMAND ----------

search_space = {'n_estimators': hp.quniform('n_estimators', 100,600,100), 'max_depth':  hp.quniform('max_depth', 2,7,1)}

spark_trials = SparkTrials(parallelism=12)

mlflow.sklearn.autolog()

with mlflow.start_run():
  best_params = fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=32, trials=spark_trials, rstate=np.random.RandomState(123))

# Print the best value found for number of estimators and the max depth of the tree
print("Best value found: ", best_params)


# COMMAND ----------

# MAGIC %md

# COMMAND ----------

import mlflow
from mlflow.models.signature import infer_signature


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

with mlflow.start_run():
  reg = GradientBoostingRegressor(max_depth=int(best_params['max_depth']), n_estimators=int(best_params['n_estimators']), random_state=0)
  reg.fit(X_train, y_train)
  mlflow.sklearn.log_model(reg, "sklean-gbt", input_example=X.head(), signature=infer_signature(X, y))  
  
reg.score(X_test, y_test)

# COMMAND ----------

tmp = pd.DataFrame([X_test['hr'].values, y_test.values]).transpose()
tmp.columns=['hr','y']
tmp['sk_gbt_pred']=reg.predict(X_test)
tmp.groupby('hr').mean().plot(figsize=(11,8))

# COMMAND ----------

tmp['residual']=tmp['y']-tmp['sk_gbt_pred']
tmp.groupby('hr').mean()['residual'].plot.bar(figsize=(11,8), title="avg residual {:4.2f}".format(tmp.groupby('hr').mean()['residual'].mean()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using the SKlearn model to predict - batch inference

# COMMAND ----------

# predictions is the spark DataFrame that holds the MLlib GBT results from the test DataFrame

# COMMAND ----------

predictions.show()

# COMMAND ----------

logged_model = 'runs:/5cc37f8fe36b48af8b38117a869103aa/model'

# Load model as a Spark UDF.
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model)

# grab the feature vector columns
columns = predictions.columns[:-4]

# Predict on a Spark DataFrame.
results = predictions.withColumn('sklearn_prediction', loaded_model(*columns))
results = results.withColumn('mllib_residual', (F.col("cnt") - F.col("prediction")))
results = results.withColumn('sklearn_residual', (F.col("cnt") - F.col("sklearn_prediction")))
results = results.withColumnRenamed("prediction", "mllib_prediction")


# COMMAND ----------

results.select("hr","cnt","mllib_prediction","sklearn_prediction").toPandas().groupby("hr").mean().plot(figsize=(11,8))

# COMMAND ----------

tmp = results.select("hr","mllib_residual","sklearn_residual").toPandas().groupby('hr').mean()
tmp.plot.bar(figsize=(11,8), title="Average MLLIB GBT residual {:4.2f}     Average SKlearn GBT residual {:4.2f}".format(tmp['mllib_residual'].mean(),tmp['sklearn_residual'].mean()))

# COMMAND ----------

# MAGIC %md #### Improving the model
# MAGIC Here are some suggestions for improving this model:
# MAGIC * The count of rentals is the sum of `registered` and `casual` rentals. These two counts may have different behavior, as frequent cyclists and casual cyclists may rent bikes for different reasons. Try training one GBT model for `registered` and one for `casual`, and then add their predictions together to get the full prediction.
# MAGIC * For efficiency, this notebook used only a few hyperparameter settings. You might be able to improve the model by testing more settings. A good start is to increase the number of trees by setting `maxIter=200`; this takes longer to train but might more accurate.
# MAGIC * This notebook used the dataset features as-is, but you might be able to improve performance with some feature engineering. For example, the weather might have more of an impact on the number of rentals on weekends and holidays than on workdays. You could try creating a new feature by combining those two columns.  MLlib provides a suite of feature transformers; find out more in the [ML guide](http://spark.apache.org/docs/latest/ml-features.html).
