# Databricks notebook source
# MAGIC %md ### MLflow Project and Model Example
# MAGIC 
# MAGIC [MLflow Keras Project Example](https://github.com/dmatrix/mlflow-workshop-project-expamle-1)
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><td>
# MAGIC     <img src="https://raw.githubusercontent.com/dmatrix/mlflow-workshop-part-2/master/images/models_1.png"
# MAGIC          alt="Bank Note " width="400">
# MAGIC   </td></tr>
# MAGIC   <tr><td> Save a Keras Model Flavor and load as both Keras Native Flavor and PyFunc Flavor</td></tr>
# MAGIC   <tr><td>
# MAGIC     <img src="https://raw.githubusercontent.com/dmatrix/mlflow-workshop-part-2/master/images/models_2.png"
# MAGIC          alt="Bank Note " width="400">
# MAGIC   </td></tr>
# MAGIC </table>

# COMMAND ----------

# MAGIC %pip install keras

# COMMAND ----------

import warnings
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import mlflow.keras
import mlflow.pyfunc

# COMMAND ----------

warnings.filterwarnings("ignore", category=DeprecationWarning)
print(f"mlflow version={mlflow.__version__};keras version={keras.__version__};tensorlow version={tf.__version__}")

# COMMAND ----------

# MAGIC %md [Source](https://androidkt.com/linear-regression-model-in-keras/)
# MAGIC Modified and extended for this tutorial
# MAGIC 
# MAGIC Problem: Build a simple Linear NN Model that predicts Celsius temperaturers from training data with Fahrenheit degree
# MAGIC Project Example: This is converted into a [project example](https://github.com/dmatrix/mlflow-workshop-project-expamle-1)
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><td>
# MAGIC     <img src="https://raw.githubusercontent.com/dmatrix/mlflow-workshop-part-2/master/images/temperature-conversion.png"
# MAGIC          alt="Bank Note " width="600">
# MAGIC   </td></tr>
# MAGIC </table>

# COMMAND ----------

# MAGIC %md Generate our X, y, and predict data

# COMMAND ----------

def f2c(f):
  return (f - 32) * 5.0/9.0

def gen_data(start, stop, step):
    X_fahrenheit = np.arange(start, stop, step, dtype=float)
    
    # Randomize the input
    np.random.shuffle(X_fahrenheit)
    y_celsius = np.array(np.array([f2c(f) for f in X_fahrenheit]))

    # generate inference Fahrenheit data to predict Celius 
    predict_data =[]
    [predict_data.append(t) for t in range (212, 170, -5)]
    
    return (X_fahrenheit, y_celsius, predict_data)

# COMMAND ----------

# MAGIC %md #### Define Inference Functions

# COMMAND ----------

def predict_keras_model(uri, data):
  model = mlflow.keras.load_model(uri)
  return [(f"(F={f}, C={model.predict([f])[0]})") for f in data]

def predict_pyfunc_model(uri, data):
  # Given Fahrenheit -> Predict Celcius
  # Create a pandas DataFrame with Fahrenheit unseen values
  # Get the Celius prediction
  pyfunc_model = mlflow.pyfunc.load_model(uri)
  df = pd.DataFrame(np.array(data))
  return pyfunc_model.predict(df)

# COMMAND ----------

# MAGIC %md #### Build a Keras Dense NN model

# COMMAND ----------

# Define the model
def baseline_model():
   model = keras.Sequential([
      keras.layers.Dense(64, activation='relu', input_shape=[1]),
      keras.layers.Dense(64, activation='relu'),
      keras.layers.Dense(1)
   ])

   optimizer = keras.optimizers.RMSprop(0.001)

   # Compile the model
   model.compile(loss='mean_squared_error',
                 optimizer=optimizer,
                 metrics=['mean_absolute_error', 'mean_squared_error'])
   return model

# COMMAND ----------

# MAGIC %md Capture run metrics using MLflow API

# COMMAND ----------

# MAGIC %md With `model.<flavor>.autolog()`, you don't need to log parameter, metrics, models etc.
# MAGIC autlogging is a convenient method to make it easier to use MLflow Fluent APIs, hence making your code eaiser to read.
# MAGIC   * Autologgin Features for [Model Flavors](https://mlflow.org/docs/latest/tracking.html#automatic-logging)

# COMMAND ----------

def mlflow_run(params, X, y, run_name="Keras Linear Regression"):

   # Start MLflow run and log everyting...
   with mlflow.start_run(run_name=run_name) as run:
      
      run_id = run.info.run_uuid
      exp_id = run.info.experiment_id
      
      model = baseline_model()
      # single line of MLflow Fluent API obviates the need to log
      # individual parameters, metrics, model, artifacts etc...
      # https://mlflow.org/docs/latest/python_api/mlflow.keras.html#mlflow.keras.autolog
      mlflow.keras.autolog()
      model.fit(X, y, batch_size=params['batch_size'], epochs=params['epochs'])

      return (exp_id, run_id)

# COMMAND ----------

# MAGIC %md #### Train the Keras model

# COMMAND ----------

# MAGIC %md Generate X, y, and predict_data

# COMMAND ----------

(X, y, predict_data) = gen_data(-212, 10512, 2)

# COMMAND ----------

params = {'batch_size': 10,'epochs': 100}
(exp_id, run_id) = mlflow_run(params, X, y)
print(f"Finished Experiment id={exp_id} and run id = {run_id}")

# COMMAND ----------

# MAGIC %md ### Check the MLflow UI
# MAGIC  * check the Model file
# MAGIC  * check the Conda.yaml file
# MAGIC  * check the metrics

# COMMAND ----------

# MAGIC %md ### Load the Keras Model Flavor: Native Flavor 

# COMMAND ----------

# Load this Keras Model Flavor as a Keras native model flavor and make a prediction
model_uri = f"runs:/{run_id}/model"
print(f"Loading the Keras Model={model_uri} as Keras Model")
predictions = predict_keras_model(model_uri, predict_data)
print(predictions)

# COMMAND ----------

# MAGIC %md ### Load the Keras Model Flavor as a PyFunc Flavor 

# COMMAND ----------

# Load this Keras Model Flavor as a pyfunc model flavor and make a prediction
pyfunc_uri = f"runs:/{run_id}/model"
print(f"Loading the Keras Model={pyfunc_uri} as Pyfunc Model")
predictions = predict_pyfunc_model(pyfunc_uri, predict_data)
print(predictions)

# COMMAND ----------

# MAGIC %md ### HOMEWORK Excercise
# MAGIC 
# MAGIC Using what we have learning in this session:
# MAGIC   * Can you decrease the MSE?
# MAGIC   * Increase the size of input data
# MAGIC   * Change the batch size and number of epochs
# MAGIC       * Run at least three experiments with different parameters: number of epochs, batches
# MAGIC   * Compare the runs and metrics
# MAGIC   
# MAGIC Were you able lower the MSE?
