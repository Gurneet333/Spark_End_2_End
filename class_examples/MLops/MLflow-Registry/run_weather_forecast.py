# Databricks notebook source
# MAGIC %md
# MAGIC # Machine learning application: Forecasting wind power. Using alternative energy for social & enviromental Good
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><td>
# MAGIC     <img src="https://github.com/dmatrix/mlflow-workshop-part-3/raw/master/images/wind_farm.jpg"
# MAGIC          alt="Keras NN Model as Logistic regression"  width="800">
# MAGIC   </td></tr>
# MAGIC </table>
# MAGIC 
# MAGIC In this notebook, we will use the MLflow Model Registry to build a machine learning application that forecasts the daily power output of a [wind farm](https://en.wikipedia.org/wiki/Wind_farm). 
# MAGIC 
# MAGIC Wind farm power output depends on weather conditions: generally, more energy is produced at higher wind speeds. Accordingly, the machine learning models used in the notebook predict power output based on weather forecasts with three features: `wind direction`, `wind speed`, and `air temperature`.
# MAGIC 
# MAGIC * This notebook uses altered data from the [National WIND Toolkit dataset](https://www.nrel.gov/grid/wind-toolkit.html) provided by NREL, which is publicly available and cited as follows:*
# MAGIC 
# MAGIC * Draxl, C., B.M. Hodge, A. Clifton, and J. McCaa. 2015. Overview and Meteorological Validation of the Wind Integration National Dataset Toolkit (Technical Report, NREL/TP-5000-61740). Golden, CO: National Renewable Energy Laboratory.*
# MAGIC 
# MAGIC * Draxl, C., B.M. Hodge, A. Clifton, and J. McCaa. 2015. "The Wind Integration National Dataset (WIND) Toolkit." Applied Energy 151: 355366.*
# MAGIC 
# MAGIC * Lieberman-Cribbin, W., C. Draxl, and A. Clifton. 2014. Guide to Using the WIND Toolkit Validation Code (Technical Report, NREL/TP-5000-62595). Golden, CO: National Renewable Energy Laboratory.*
# MAGIC 
# MAGIC * King, J., A. Clifton, and B.M. Hodge. 2014. Validation of Power Output for the WIND Toolkit (Technical Report, NREL/TP-5D00-61714). Golden, CO: National Renewable Energy Laboratory.*
# MAGIC 
# MAGIC Googl's Deep publised a [AI for Social Good: 7 Inspiring Examples](https://www.springboard.com/blog/ai-for-good/) blog. One of example was
# MAGIC how Wind Farms can predict expected power ouput based on wind conditions and temperature, hence mitigating the burden from consuming
# MAGIC energy from fossil fuels. 
# MAGIC 
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><td>
# MAGIC     <img src="https://github.com/dmatrix/ds4g-workshop/raw/master/notebooks/images/deepmind_system-windpower.gif"
# MAGIC          alt="Deep Mind ML Wind Power"  width="400">
# MAGIC     <img src="https://github.com/dmatrix/ds4g-workshop/raw/master/notebooks/images/machine_learning-value_wind_energy.max-1000x1000.png"
# MAGIC          alt="Deep Mind ML Wind Power"  width="400">
# MAGIC   </td></tr>
# MAGIC </table>

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

import mlflow
print(mlflow.__version__)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run some class and utility notebooks 
# MAGIC 
# MAGIC This allows us to use some Python model classes and utility functions

# COMMAND ----------

# MAGIC %run ./rfr_class

# COMMAND ----------

# MAGIC %run ./utils_class

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load our training data

# COMMAND ----------

# Load and print dataset
csv_path = "/dbfs/FileStore/shared_uploads/T1.csv"

# Use column 0 (date) as the index
wind_farm_data = Utils.load_data(csv_path, index_col=0)
wind_farm_data.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get Validation data

# COMMAND ----------

X_train, y_train = Utils.get_training_data(wind_farm_data)
val_x, val_y = Utils.get_validation_data(wind_farm_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Initialize a set of hyperparameters for the training and try three runs

# COMMAND ----------

# Initialize our model hyperparameters
params_list = [{"n_estimators": 100},
               {"n_estimators": 200},
               {"n_estimators": 300}]

# COMMAND ----------

# Train, fit and register our model and iterate over few different tuning parameters
# Use sqlite:///mlruns.db as the local store for tracking and model registery

# mlflow.set_tracking_uri("sqlite:///mlruns.db")

model_name = "PowerForecastingModel"
for params in params_list:
  rfr = RFRModel.new_instance(params)
  print("Using paramerts={}".format(params))
  runID = rfr.mlflow_run(X_train, y_train, val_x, val_y, model_name, register=True)
  print("MLflow run_id={} completed with MSE={} and RMSE={}".format(runID, rfr.mse, rfr.rsme))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Let's Examine the MLflow UI
# MAGIC 
# MAGIC 1. From your local host where your started jupyter lab start the mlflow ui
# MAGIC 2. **mlflow ui --backend-store-uri sqlite:///mlruns.db**
# MAGIC 3. Go to http://127.0.0.1:5000 in your browser
# MAGIC 4. Let's examine some models and start comparing their metrics
# MAGIC 5. Register three versions of the models

# COMMAND ----------

#!mlflow ui --backend-store-uri sqlite:///mlruns.db

# COMMAND ----------

# MAGIC %md
# MAGIC # Integrating Model Registry with CI/CD Forecasting Application
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><td>
# MAGIC     <img src="https://github.com/dmatrix/mlflow-workshop-part-3/raw/master/images/forecast_app.png"
# MAGIC          alt="Keras NN Model as Logistic regression"  width="800">
# MAGIC   </td></tr>
# MAGIC </table>
# MAGIC 
# MAGIC 1. Use the model registry fetch different versions of the model
# MAGIC 2. Score the model
# MAGIC 3. Select the best scored model
# MAGIC 4. Promote model to production, after testing

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define a helper function to load PyFunc model from the registry
# MAGIC <table>
# MAGIC   <tr><td> Save a Keras Model Flavor and load as PyFunc Flavor</td></tr>
# MAGIC   <tr><td>
# MAGIC     <img src="https://raw.githubusercontent.com/dmatrix/mlflow-workshop-part-2/master/images/models_2.png"
# MAGIC          alt="" width="600">
# MAGIC   </td></tr>
# MAGIC </table>

# COMMAND ----------

def score_model(data, model_uri):
    model = mlflow.pyfunc.load_model(model_uri)
    return model.predict(data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load scoring data

# COMMAND ----------

# Load the score data
score_path = "./data/score_windfarm_data.csv"
score_df = Utils.load_data(csv_path, index_col=0)
score_df.head()

# COMMAND ----------

# Drop the power column since we are predicting that value
actual_power = pd.DataFrame(score_df["LV ActivePower (kW)"].values, columns=['LV ActivePower (kW)'])
score = score_df.drop("LV ActivePower (kW)", axis=1)

# COMMAND ----------

# Formulate the model URI to fetch from the model registery
model_uri = "models:/{}/{}".format(model_name, 1)

# Predict the Power output 
pred_1 = pd.DataFrame(score_model(score, model_uri), columns=["predicted_1"])
pred_1

# COMMAND ----------

# MAGIC %md
# MAGIC #### Combine with the actual power

# COMMAND ----------

actual_power["predicted_1"] = pred_1["predicted_1"]
actual_power

# COMMAND ----------

# Formulate the model URI to fetch from the model registery
model_uri = "models:/{}/{}".format(model_name, 2)

# Predict the Power output
pred_2 = pd.DataFrame(score_model(score, model_uri), columns=["predicted_2"])
pred_2

# COMMAND ----------

actual_power["predicted_2"] = pred_2["predicted_2"]
actual_power

# COMMAND ----------

# Formulate the model URI to fetch from the model registery
model_uri = "models:/{}/{}".format(model_name, 3)

# Formulate the model URI to fetch from the model registery
pred_3 = pd.DataFrame(score_model(score, model_uri), columns=["predicted_3"])
pred_3

# COMMAND ----------

# MAGIC %md
# MAGIC ### Combine the values into a single pandas DataFrame 

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC 
# MAGIC actual_power.plot.line()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Homework - Challenge 
# MAGIC 
# MAGIC 1. Can you improve the model with different hyperparameters to get better RSME
# MAGIC 2. Register the model and score it
# MAGIC 3. Make stage transitions
# MAGIC 4. Load the "Production model"
# MAGIC 5. Score the production model

# COMMAND ----------

model_uri

# COMMAND ----------

# Formulate the model URI to fetch from the model registery
model_uri = "models:/{}/{}".format(model_name, 'staging')

# Formulate the model URI to fetch from the model registery
pred_3 = pd.DataFrame(score_model(score, model_uri), columns=["predicted_Stage"])
pred_3
