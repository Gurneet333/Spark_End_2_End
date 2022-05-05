# Databricks notebook source
# MAGIC %md
# MAGIC # Predicting Song Listens Using Apache Spark
# MAGIC One of the most common uses of big data is to predict what users want.  This allows Google to show you relevant ads, Amazon to recommend relevant products, and Netflix to recommend movies that you might like. Here I  demonstrate how Apache Spark can be used to recommend songs to a user.  We will start with some basic techniques, and then use the [Spark ML][sparkml] library's Alternating Least Squares method to make more sophisticated predictions.
# MAGIC 
# MAGIC I use triplets from 1 million users from the [million song dataset](http://labrosa.ee.columbia.edu/millionsong/pages/getting-dataset), along with metadata for the songs that includes information such as the song title, artist name and tempo. The triplets com in the form (User ID, Song ID, Number of Plays). 
# MAGIC [sparkml]: https://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html
# MAGIC 
# MAGIC [Building an Implicit Recommendation Engine with Spark](https://youtu.be/58OjaDH2FI0)
# MAGIC 
# MAGIC [Colaborative Filtering with Implict Feedback Datasets](http://yifanhu.net/PUB/cf.pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load and Cache
# MAGIC 
# MAGIC The Databricks File System (DBFS) sits on top of S3. We're going to be accessing this data a lot. Rather than read it over and over again from S3, we'll cache both
# MAGIC the song plays csv and the song metadata in memory.

# COMMAND ----------

# import libraries
from pyspark.sql.types import *
from pyspark.sql import functions as F6

# Load the data from the lake
raw_plays_df_with_int_ids= spark.read.format('delta').load("s3a://dscc202-datasets/songs/triplets")
songs2tracks_df = spark.read.format('delta').load("s3a://dscc202-datasets/songs/song2tracks")
metadata_df = spark.read.format('delta').load("s3a://dscc202-datasets/songs/metadata")


# get total unique users and songs
unique_users = raw_plays_df_with_int_ids.select('userId').distinct().count()
unique_songs = raw_plays_df_with_int_ids.select('songId').distinct().count()
print('Number of unique users: {0}'.format(unique_users))
print('Number of unique songs: {0}'.format(unique_songs))

# cache
raw_plays_df_with_int_ids.cache()
raw_plays_df_with_int_ids.printSchema()
# raw_plays_df_with_int_ids.show(5)

songs2tracks_df.cache()
songs2tracks_df.printSchema()
songs2tracks_df.show(5)

metadata_df.cache()
metadata_df.printSchema()
# metadata_df.show(5)

# COMMAND ----------

raw_plays_df_with_int_ids = sqlContext.sql("select * from G05_db.SilverTable_Wallets")
# raw_plays_df_with_int_ids = df.select("*").toPandas()

# COMMAND ----------

display(raw_plays_df_with_int_ids)

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql.functions import dense_rank
raw_plays_df_with_int_ids=raw_plays_df_with_int_ids.withColumn("new_songId",dense_rank().over(Window.orderBy("token_address")))
raw_plays_df_with_int_ids=raw_plays_df_with_int_ids.withColumn("new_userId",dense_rank().over(Window.orderBy("wallet_address")))
raw_plays_df_with_int_ids = raw_plays_df_with_int_ids.withColumnRenamed("token_address","songId")
raw_plays_df_with_int_ids = raw_plays_df_with_int_ids.withColumnRenamed("wallet_address","userId")
raw_plays_df_with_int_ids = raw_plays_df_with_int_ids.withColumnRenamed("transactions","Plays")

# COMMAND ----------

display(raw_plays_df_with_int_ids)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exploratory Data Analysis

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def prepareSubplot(xticks, yticks, figsize=(10.5, 6), hideLabels=False, gridColor='#999999',
                gridWidth=1.0, subplots=(1, 1)):
    """Template for generating the plot layout."""
    plt.close()
    fig, axList = plt.subplots(subplots[0], subplots[1], figsize=figsize, facecolor='white',
                               edgecolor='white')
    if not isinstance(axList, np.ndarray):
        axList = np.array([axList])
        
    for ax in axList.flatten():
        ax.axes.tick_params(labelcolor='#999999', labelsize='10')
        for axis, ticks in [(ax.get_xaxis(), xticks), (ax.get_yaxis(), yticks)]:
            axis.set_ticks_position('none')
            axis.set_ticks(ticks)
            axis.label.set_color('#999999')
            if hideLabels: axis.set_ticklabels([])
        ax.grid(color=gridColor, linewidth=gridWidth, linestyle='-')
        map(lambda position: ax.spines[position].set_visible(False), ['bottom', 'top', 'left', 'right'])
        
    if axList.size == 1:
        axList = axList[0]  # Just return a single axes object for a regular plot
    return fig, axList
    

from pyspark.sql import DataFrame
import inspect
def printDataFrames(verbose=False):
    frames = inspect.getouterframes(inspect.currentframe())
    notebookGlobals = frames[1][0].f_globals
    for k,v in notebookGlobals.items():
        if isinstance(v, DataFrame) and '_' not in k:
            print("{0}: {1}".format(k, v.columns)) if verbose else print("{0}".format(k))


def printLocalFunctions(verbose=False):
    frames = inspect.getouterframes(inspect.currentframe())
    notebookGlobals = frames[1][0].f_globals
    import types
    ourFunctions = [(k, v.__doc__) for k,v in notebookGlobals.items() if isinstance(v, types.FunctionType) and v.__module__ == '__main__']
    
    for k,v in ourFunctions:
        print("** {0} **".format(k))
        if verbose:
            print(v)
        

# COMMAND ----------

# count total entries
total_entries = raw_plays_df_with_int_ids.count()

# find percentage listens by number of songs played
number_listens = []
for i in range(10):
  number_listens.append(float(raw_plays_df_with_int_ids.filter(raw_plays_df_with_int_ids.Plays == i+1).count())/total_entries*100)

# create bar plot
bar_width = 0.7
colorMap = 'Set1'
cmap = cm.get_cmap(colorMap)

fig, ax = prepareSubplot(np.arange(0, 10, 1), np.arange(0, 80, 5))
plt.bar(np.linspace(1,10,10), number_listens, width=bar_width, color=cmap(0))
plt.xticks(np.linspace(1,10,10) + bar_width/2.0, np.linspace(1,10,10))
plt.xlabel('Number of Plays'); plt.ylabel('%')
plt.title('Percentage Number of Plays of Songs')
display(fig)


# COMMAND ----------

#find cumulative sum
cumsum_number_listens = np.cumsum(number_listens)
cumsum_number_listens = np.insert(cumsum_number_listens, 0, 0)
print(cumsum_number_listens)

fig, ax = prepareSubplot(np.arange(0, 10, 1), np.arange(0, 100, 10))
plt.plot(np.linspace(0,10,11), cumsum_number_listens, color=cmap(1))
plt.xticks(np.linspace(0,10,11), np.linspace(0,10,11))
plt.xlabel('Number of Plays'); plt.ylabel('Cumulative sum')
plt.title('Cumulative disribution of Number of Song Plays')
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC Almost 60% of songs are played once 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Recommendations
# MAGIC 
# MAGIC One way to recommend songs is to simply always recommend the songs with that are most listened to. In this part, I use Spark to find the name, total number of plays. I also filter those that have been listened to by more that 200 unique users to account for a broad range of appeal

# COMMAND ----------

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
df=sqlContext.sql("select * from G05_db.SilverTable_Wallets")
result_df = df.select("*").toPandas()
song_ids_with_total_listens = raw_plays_df_with_int_ids.groupBy('songId') \
                                                       .agg(F.count(raw_plays_df_with_int_ids.Plays).alias('User_Count'),
                                                            F.sum(raw_plays_df_with_int_ids.Plays).alias('Total_Plays')) \
                                                       .orderBy('Total_Plays', ascending = False)

print('song_ids_with_total_listens:',
song_ids_with_total_listens.show(3, truncate=False))

# Join with metadata to get artist and song title
song_names_with_plays_df = song_ids_with_total_listens.join(metadata_df, 'songId' ) \
                                                      .filter('User_Count >= 2') \
                                                      .select('artist_name', 'title', 'songId', 'User_Count','Total_Plays') \
                                                      .orderBy('Total_Plays', ascending = False)

print('song_names_with_plays_df:',
song_names_with_plays_df.show(20, truncate = False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Collaborative Filtering
# MAGIC 
# MAGIC [Collaborative filtering][collab] is a method of making automatic predictions (filtering) about the interests of a user by collecting preferences or taste information from many users (collaborating). The underlying assumption of the collaborative filtering approach is that if a person A has the same opinion as a person B on an issue, A is more likely to have B's opinion on a different issue x than to have the opinion on x of a person chosen randomly. More information about collaborative filtering can be found [here][collab2].
# MAGIC 
# MAGIC The image at the right (from [Wikipedia][collab]) shows an example of predicting of the user's rating using collaborative filtering. At first, people rate different items (like videos, images, games). After that, the system is making predictions about a user's rating for an item, which the user has not rated yet. These predictions are built upon the existing ratings of other users, who have similar ratings with the active user. For instance, in the image below the system has made a prediction, that the active user will not like the video.
# MAGIC 
# MAGIC We will apply he same concept but interchange the rating of an item with the number of plays since a larger number of plays would equate to a higher rating.
# MAGIC 
# MAGIC <br clear="all"/>
# MAGIC 
# MAGIC ----
# MAGIC 
# MAGIC For song recommendations, we start with a matrix whose entries are number of song plays by users.
# MAGIC 
# MAGIC Since not all users have listened to all songs, we do not know all of the entries in this matrix, which is precisely why we need collaborative filtering.  For each user, we have number of song plays for only a subset of the songs.  With collaborative filtering, the idea is to approximate the matrix by factorizing it as the product of two matrices: one that describes properties of each user, and one that describes properties of each song.
# MAGIC 
# MAGIC <br clear="all"/>
# MAGIC 
# MAGIC We want to select these two matrices such that the error for the users/songs pairs where we know the correct number of plays is minimized.  The [Alternating Least Squares][als] algorithm does this by first randomly filling the users matrix with values and then optimizing the value of the songs such that the error is minimized.  Then, it holds the songs matrix constant and optimizes the value of the user's matrix.  This alternation between which matrix to optimize is the reason for the "alternating" in the name.
# MAGIC 
# MAGIC Given a fixed set of user factors (i.e., values in the users matrix), we use the known number of plays to find the best values for the songs factors using the least squares optimization.  Then we "alternate" and pick the best user factors given fixed songs factors.
# MAGIC 
# MAGIC [collab]: https://en.wikipedia.org/?title=Collaborative_filtering
# MAGIC [collab2]: http://recommender-systems.org/collaborative-filtering/
# MAGIC [als]: https://en.wikiversity.org/wiki/Least-Squares_Method

# COMMAND ----------

# We'll hold out 60% for training, 20% of our data for validation, and leave 20% for testing
seed = 42
(split_60_df, split_a_20_df, split_b_20_df) = raw_plays_df_with_int_ids.randomSplit([0.6, 0.2, 0.2], seed = seed)

# Let's cache these datasets for performance
training_df = split_60_df.cache()
validation_df = split_a_20_df.cache()
test_df = split_b_20_df.cache()

print('Training: {0}, validation: {1}, test: {2}\n'.format(
  training_df.count(), validation_df.count(), test_df.count())
)
training_df.show(3)
validation_df.show(3)
test_df.show(3)

# COMMAND ----------

#Number of plays needs to be double type, not integers
validation_df = validation_df.withColumn("Plays", validation_df["Plays"].cast(DoubleType()))
validation_df.show(10)

# COMMAND ----------

from pyspark.sql.functions import col
display(validation_df.filter(col("Plays")==0))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Alternating Least Squares
# MAGIC 
# MAGIC In this part, we will use the Apache Spark ML Pipeline implementation of Alternating Least Squares, [ALS](http://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.recommendation.ALS). ALS takes a training dataset (DataFrame) and several parameters that control the model creation process. To determine the best values for the parameters, we will use ALS to train several models, and then we will select the best model and use the parameters from that model in the rest of the analysis.
# MAGIC 
# MAGIC The process we will use for determining the best model is as follows:
# MAGIC 1. Pick a set of model parameters. The most important parameter to model is the *rank*, which is the number of columns in the Users matrix or the number of rows in the Movies matrix (blue in the diagram above). In general, a lower rank will mean higher error on the training dataset, but a high rank may lead to [overfitting](https://en.wikipedia.org/wiki/Overfitting).  We will train models with ranks of 4, 8, 12 and 16 using the `training_df` dataset, as well as iterating over the regularization parameter. Because most of the number of songs are have only be listened to once, I expect a higher regularization parameter to be more effective.
# MAGIC 
# MAGIC 2. We set the appropriate parameters on the `ALS` object:
# MAGIC     * The "User" column will be set to the values in our `userId` DataFrame column.
# MAGIC     * The "Item" column will be set to the values in our `songId` DataFrame column.
# MAGIC     * The "Rating" column will be set to the values in our `Plays` DataFrame column.
# MAGIC     * We'll set the max number of iterations to be 5.
# MAGIC 
# MAGIC 3. Have the ALS output transformation (i.e., the result of [ALS.fit()](http://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.recommendation.ALS.fit)) produce a _new_ column
# MAGIC    called "prediction" that contains the predicted value.
# MAGIC 
# MAGIC 4. Create multiple models using [ALS.fit()](http://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.recommendation.ALS.fit), one for each of our rank values. We'll fit
# MAGIC    against the training data set (`training_df`).
# MAGIC 
# MAGIC 5. For each model, we'll run a prediction against our validation data set (`validation_df`) and check the error.
# MAGIC 
# MAGIC 6. We'll keep the model with the best error rate.
# MAGIC 
# MAGIC #### Why do my own cross-validation?
# MAGIC 
# MAGIC A challenge for collaborative filtering is how to provide ratings to a new user (a user who has not provided *any* ratings at all). Some recommendation systems choose to provide new users with a set of default ratings (e.g., an average value across all ratings), while others choose to provide no ratings for new users. Spark's ALS algorithm yields a NaN (`Not a Number`) value when asked to provide a rating for a new user.
# MAGIC 
# MAGIC Using the ML Pipeline's [CrossValidator](http://spark.apache.org/docs/1.6.2/api/python/pyspark.ml.html#pyspark.ml.tuning.CrossValidator) with ALS is thus problematic, because cross validation involves dividing the training data into a set of folds (e.g., three sets) and then using those folds for testing and evaluating the parameters during the parameter grid search process. It is likely that some of the folds will contain users that are not in the other folds, and, as a result, ALS produces NaN values for those new users. When the CrossValidator uses the Evaluator (RMSE) to compute an error metric, the RMSE algorithm will return NaN. This will make *all* of the parameters in the parameter grid appear to be equally good (or bad).
# MAGIC 
# MAGIC A discussion on the issue is shown [Spark JIRA 14489](https://issues.apache.org/jira/browse/SPARK-14489). There are proposed workarounds of having ALS provide default values or having RMSE drop NaN values. Both introduce potential issues. We have chosen to have RMSE drop NaN values. While this does not solve the underlying issue of ALS not predicting a value for a new user, it does provide some evaluation value. We manually implement the parameter grid search process using a for loop (below) and remove the NaN values before using RMSE.  Fixed in [Spark JIRA 12896](https://github.com/apache/spark/pull/12896)  Introduces a parameter coldStartStrategy that allows NaN values to be dropped.
# MAGIC 
# MAGIC For a production application, tradeoffs can be considered in how to handle new users.

# COMMAND ----------

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Let's initialize our ALS learner
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
reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="Plays", metricName="rmse")

grid = ParamGridBuilder() \
  .addGrid(als.maxIter, [10]) \
  .addGrid(als.regParam, [0.15, 0.2, 0.25]) \
  .addGrid(als.rank, [4, 8, 12, 16]) \
  .build()

# Create a cross validator, using the pipeline, evaluator, and parameter grid you created in previous steps.
cv = CrossValidator(estimator=als, evaluator=reg_eval, estimatorParamMaps=grid, numFolds=3)



tolerance = 0.03
ranks = [4, 8, 12, 16]
regParams = [0.15, 0.2, 0.25]
errors = [[0]*len(ranks)]*len(regParams)
models = [[0]*len(ranks)]*len(regParams)
err = 0
min_error = float('inf')
best_rank = -1
i = 0
for regParam in regParams:
  j = 0
  for rank in ranks:
    # Set the rank here:
    als.setParams(rank = rank, regParam = regParam)
    # Create the model with these parameters.
    model = als.fit(training_df)
    # Run the model to create a prediction. Predict against the validation_df.
    predict_df = model.transform(validation_df)

    # Remove NaN values from prediction (due to SPARK-14489)
    predicted_plays_df = predict_df.filter(predict_df.prediction != float('nan'))
    predicted_plays_df = predicted_plays_df.withColumn("prediction", F.abs(F.round(predicted_plays_df["prediction"],0)))
    # Run the previously created RMSE evaluator, reg_eval, on the predicted_ratings_df DataFrame
    error = reg_eval.evaluate(predicted_plays_df)
    errors[i][j] = error
    models[i][j] = model
    print( 'For rank %s, regularization parameter %s the RMSE is %s' % (rank, regParam, error))
    if error < min_error:
      min_error = error
      best_params = [i,j]
    j += 1
  i += 1

als.setRegParam(regParams[best_params[0]])
als.setRank(ranks[best_params[1]])
print( 'The best model was trained with regularization parameter %s' % regParams[best_params[0]])
print( 'The best model was trained with rank %s' % ranks[best_params[1]])
my_model = models[best_params[0]][best_params[1]]

# COMMAND ----------

#Example of predicted plays
predicted_plays_df.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC Because we're only looking for the nonzero plays, one way to increase the accuracy would be to convert predicted listens that are zero to one. However this would not really help the model overall, since our ultimate goal is for song recommendation and if the predicted number of plays is zero that is okay, this means that the user does o want to listen to the song.  

# COMMAND ----------

# MAGIC %md
# MAGIC ### Testing The Model
# MAGIC 
# MAGIC So far, we used the `training_df` and `validation_df` datasets to select the best model.  Since we used these two datasets to determine what model is best, we cannot use them to test how good the model is; otherwise, we would be very vulnerable to [overfitting](https://en.wikipedia.org/wiki/Overfitting).  To decide how good the model is, we need to use the `test_df` dataset.  We will use the best rank, and best regularization parameter in the list `best_params` previously determined to create a model for predicting songs for the test dataset and then we will compute the RMSE.

# COMMAND ----------

# In ML Pipelines, this next step has a bug that produces unwanted NaN values. We
# have to filter them out. See https://issues.apache.org/jira/browse/SPARK-14489

test_df = test_df.withColumn("Plays", test_df["Plays"].cast(DoubleType()))
predict_df = my_model.transform(test_df)

# Remove NaN values from prediction (due to SPARK-14489)
predicted_test_df = predict_df.filter(predict_df.prediction != float('nan'))

# Round floats to whole numbers
predicted_test_df = predicted_test_df.withColumn("prediction", F.abs(F.round(predicted_test_df["prediction"],0)))
# Run the previously created RMSE evaluator, reg_eval, on the predicted_test_df DataFrame
test_RMSE = reg_eval.evaluate(predicted_test_df)

print('The model had a RMSE on the test set of {0}'.format(test_RMSE))

# COMMAND ----------

test_df_p=test_df.toPandas()

# COMMAND ----------

training_df_p=training_df.toPandas()

# COMMAND ----------

import pandas as pd
l1=list(pd.unique(training_df_p['userId']))
l2=list(pd.unique(test_df_p['userId']))

l3=list(set(l2) - set(l1))
len(l3)

# COMMAND ----------

pd_test_df=predicted_test_df.toPandas()
l4=list(pd.unique(pd_test_df['userId']))

# COMMAND ----------

l5=list(set(l2) - set(l4))
len(l5)

# COMMAND ----------

test_df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Comparing the Model
# MAGIC 
# MAGIC Looking at the RMSE for the results predicted by the model versus the values in the test set is one way to evalute the quality of our model. Another way to evaluate the model is to evaluate the error from a test set where every rating is the average number of plays from the training set.

# COMMAND ----------

avg_plays_df = training_df.groupBy().avg('Plays').select(F.round('avg(Plays)'))

avg_plays_df.show(3)
# Extract the average rating value. (This is row 0, column 0.)
training_avg_plays = avg_plays_df.collect()[0][0]

print('The average number of plays in the dataset is {0}'.format(training_avg_plays))

# Add a column with the average rating
test_for_avg_df = test_df.withColumn('prediction', F.lit(training_avg_plays))

# Run the previously created RMSE evaluator, reg_eval, on the test_for_avg_df DataFrame
test_avg_RMSE = reg_eval.evaluate(test_for_avg_df)

print("The RMSE on the average set is {0}".format(test_avg_RMSE))

# COMMAND ----------

# MAGIC %md
# MAGIC Our model performs slightly better than simply predicting 3 songs for all in the test case.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prediction for a User
# MAGIC 
# MAGIC We can make a prediction for a user. To do this we make a list of all the songs that the user has not already listened to, and feed it into to model. The model will generate it the number of predicted plays, which we can arange in descending order.

# COMMAND ----------

raw_plays_df_with_int_ids.show()

# COMMAND ----------

raw_plays_df_with_int_ids.filter(raw_plays_df_with_int_ids.userId=='0xffff89dc98f926429dbc16b28b0033cdc119cdd7').select('new_userId').toPandas().iloc[0,0]

# COMMAND ----------

raw_plays_df_with_int_ids.filter(raw_plays_df_with_int_ids.new_songId == 191).show()

# COMMAND ----------

UserID = 1007200
listened_songs = raw_plays_df_with_int_ids.filter(raw_plays_df_with_int_ids.new_userId == UserID) \
                                          .select('new_songId')
                                   
listened_songs.show()
# generate list of listened songs
# listened_songs_list = []
# for song in listened_songs.collect():
#   listened_songs_list.append(song['new_songId'])

# print('Songs user has listened to:')
# listened_songs.select('artist_name', 'title').show()

# # generate dataframe of unlistened songs
# unlistened_songs = raw_plays_df_with_int_ids.filter(~ raw_plays_df_with_int_ids['new_songId'].isin(listened_songs_list)) \
#                                             .select('new_songId').withColumn('new_userId', F.lit(UserID)).distinct()

# # feed unlistened songs into model
# predicted_listens = my_model.transform(unlistened_songs)

# # remove NaNs
# predicted_listens = predicted_listens.filter(predicted_listens['prediction'] != float('nan'))

# # print output
# print('Predicted Songs:')
# predicted_listens.show()
# # predicted_listens.join(raw_plays_df_with_int_ids, 'new_songId') \
# #                  .join(metadata_df, 'songId') \
# #                  .select('artist_name', 'title', 'prediction') \
# #                  .distinct() \
# #                  .orderBy('prediction', ascending = False) \
#                  .show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC 
# MAGIC To conclude, we have used alternating least squares model to predict song listens of given users. We find that there are many songs that have only been listened to once, so we have compared models that include and exclude those songs that have been listened to once. While the model including these songs returns a slightly lower RMSE on the test dataset, the model excluding them may give better recommendations, since the training set includes more songs may be more reflective of their listeneing profiles (since they listen to them more).
# MAGIC 
# MAGIC Importantly due to Sparks efficiency to scale, this model can scale to many more users. For example Spotify has around 30 million active monthly users, which is only a magnitude more than the number of users considered in this dataset.
