# Databricks notebook source
from pyspark.sql import DataFrame
from pyspark.sql.types import *
from pyspark.sql import functions as F
from delta.tables import *
import random

import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# configureation of the spark environment appropriate for my cluster
spark.conf.set("spark.sql.shuffle.partitions", "16")  # Configure the size of shuffles the same as core count
spark.conf.set("spark.sql.adaptive.enabled", "true")  # Spark 3.0 AQE - coalescing post-shuffle partitions, converting sort-merge join to broadcast join, and skew join optimization

"""
Class that encapsulates the definition, training and testing of an ALS Music Recommender
using the facilites of MLflow for tracking and registration/packaging
"""
class MusicRecommender():
  """
  Build out the required plumbing and electrical work:
  - Get the delta lake version of the training data to be stored with the model (bread crumbs)
  - Set (create) the experiment name
  - Load and create training test splits
  - instantiate an alternate least squares estimator and its associated evaluation and tuning grid
  - create a cross validation configuration for the model training
  """
  def __init__(self, dataPath: str, modelName: str, minPlayCount: int = 1)->None:
    self.dataPath = dataPath
    self.modelName = modelName
    self.minPlayCount = minPlayCount
    
    # create an MLflow experiment for this model
    MY_EXPERIMENT = "/Users/" + dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get() + "/" + self.modelName + "-experiment/"
    mlflow.set_experiment(MY_EXPERIMENT)
    
    # split the data set into train, validation and test anc cache them
    # We'll hold out 60% for training, 20% of our data for validation, and leave 20% for testing
    self.raw_plays_df_with_int_ids = spark.read.format('delta').load(self.dataPath+"triplets")
    self.raw_plays_df_with_int_ids = self.raw_plays_df_with_int_ids.filter(self.raw_plays_df_with_int_ids.Plays >= self.minPlayCount).cache()
    self.metadata_df = spark.read.format('delta').load(self.dataPath+"metadata").cache()
    self.training_data_version = DeltaTable.forPath(spark, self.dataPath+"triplets").history().head(1)[0]['version']
    
    seed = 42
    (split_60_df, split_a_20_df, split_b_20_df) = self.raw_plays_df_with_int_ids.randomSplit([0.6, 0.2, 0.2], seed = seed)
    # Let's cache these datasets for performance
    self.training_df = split_60_df.cache()
    self.validation_df = split_a_20_df.cache()
    self.test_df = split_b_20_df.cache()
    
    # Initialize our ALS learner
    als = ALS()

    # Now set the parameters for the method
    als.setMaxIter(5)\
       .setSeed(seed)\
       .setItemCol("new_songId")\
       .setRatingCol("Plays")\
       .setUserCol("new_userId")\
       .setColdStartStrategy("drop")

    # Now let's compute an evaluation metric for our test dataset
    # We Create an RMSE evaluator using the label and predicted columns
    self.reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="Plays", metricName="rmse")

    # Setup an ALS hyperparameter tuning grid search
    grid = ParamGridBuilder() \
      .addGrid(als.maxIter, [5, 10, 15]) \
      .addGrid(als.regParam, [0.15, 0.2, 0.25]) \
      .addGrid(als.rank, [4, 8, 12, 16, 20]) \
      .build()
    
    """
    grid = ParamGridBuilder() \
      .addGrid(als.maxIter, [5]) \
      .addGrid(als.regParam, [0.25]) \
      .addGrid(als.rank, [16]) \
      .build()
    """

    # Create a cross validator, using the pipeline, evaluator, and parameter grid you created in previous steps.
    self.cv = CrossValidator(estimator=als, evaluator=self.reg_eval, estimatorParamMaps=grid, numFolds=3)

  """
  Train the ALS music recommendation using the training and validation set and the cross validation created
  at the time of instantiation.  Use MLflow to log the training results and push the best model from this
  training session to the MLflow registry at "Staging"
  """
  def train(self):
    # setup the schema for the model
    input_schema = Schema([
      ColSpec("integer", "new_songId"),
      ColSpec("integer", "new_userId"),
    ])
    output_schema = Schema([ColSpec("double")])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    
    with mlflow.start_run(run_name=self.modelName+"-run") as run:
      mlflow.set_tags({"group": 'admin', "class": "DSCC202-402"})
      mlflow.log_params({"user_rating_training_data_version": self.training_data_version,"minimum_play_count":self.minPlayCount})

      # Run the cross validation on the training dataset. The cv.fit() call returns the best model it found.
      cvModel = self.cv.fit(self.training_df)

      # Evaluate the best model's performance on the validation dataset and log the result.
      validation_metric = self.reg_eval.evaluate(cvModel.transform(self.validation_df))

      mlflow.log_metric('test_' + self.reg_eval.getMetricName(), validation_metric) 

      # Log the best model.
      mlflow.spark.log_model(spark_model=cvModel.bestModel, signature = signature,
                             artifact_path='als-model', registered_model_name=self.modelName)
      
    """
    - Capture the latest model version
    - archive any previous Staged version
    - Transition this version to Staging
    """
    client = MlflowClient()
    model_versions = []

    # Transition this model to staging and archive the current staging model if there is one
    for mv in client.search_model_versions(f"name='{self.modelName}'"):
      model_versions.append(dict(mv)['version'])
      if dict(mv)['current_stage'] == 'Staging':
        print("Archiving: {}".format(dict(mv)))
        # Archive the currently staged model
        client.transition_model_version_stage(
            name=self.modelName,
            version=dict(mv)['version'],
            stage="Archived"
        )
    client.transition_model_version_stage(
        name=self.modelName,
        version=model_versions[0],  # this model (current build)
        stage="Staging"
    )

  """
  Test the model in staging with the test dataset generated when this object (music recommender)
  was instantiated.
  """
  def test(self):
    # THIS SHOULD BE THE VERSION JUST TRANINED
    model = mlflow.spark.load_model('models:/'+self.modelName+'/Staging')
    # View the predictions
    test_predictions = model.transform(self.test_df)
    RMSE = self.reg_eval.evaluate(test_predictions)
    print("Staging Model Root-mean-square error on the test dataset = " + str(RMSE))  
  
  """
  Method takes a specific userId and returns the songs that they have listened to
  and a set of recommendations in rank order that they may like based on their
  listening history.
  """
  def recommend(self, userId: int)->(DataFrame,DataFrame):
    # generate a dataframe of songs that the user has previously listened to
    listened_songs = self.raw_plays_df_with_int_ids.filter(self.raw_plays_df_with_int_ids.new_userId == userId) \
                                              .join(self.metadata_df, 'songId') \
                                              .select('new_songId', 'artist_name', 'title','Plays') \

    # generate dataframe of unlistened songs
    unlistened_songs = self.raw_plays_df_with_int_ids.filter(~ self.raw_plays_df_with_int_ids['new_songId'].isin([song['new_songId'] for song in listened_songs.collect()])) \
                                                .select('new_songId').withColumn('new_userId', F.lit(userId)).distinct()

    # feed unlistened songs into model for a predicted Play count
    model = mlflow.spark.load_model('models:/'+self.modelName+'/Staging')
    predicted_listens = model.transform(unlistened_songs)

    return (listened_songs.select('artist_name','title','Plays').orderBy('Plays', ascending = False), predicted_listens.join(self.raw_plays_df_with_int_ids, 'new_songId') \
                     .join(self.metadata_df, 'songId') \
                     .select('artist_name', 'title', 'prediction') \
                     .distinct() \
                     .orderBy('prediction', ascending = False)) 

  """
  Generate a data frame that recommends a number of songs for each of the users in the dataset (model)
  """
  def recommendForUsers(self, numOfSongs: int) -> DataFrame:
    model = mlflow.spark.load_model('models:/'+self.modelName+'/Staging')
    return model.stages[0].recommendForAllUsers(numOfSongs)

  """
  Generate a data frame that recommends a number of users for each of the songs in the dataset (model)
  """
  def recommendForSongs(self, numOfUsers: int) -> DataFrame:
    model = mlflow.spark.load_model('models:/'+self.modelName+'/Staging')
    return model.stages[0].recommendForAllItems(numOfUsers)

# COMMAND ----------


