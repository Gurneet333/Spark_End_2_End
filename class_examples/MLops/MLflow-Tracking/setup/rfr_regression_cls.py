# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC Random Forest class definition for the `RFRModel`

# COMMAND ----------

import os
import numpy as np
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


class RFRModel():
    """
    General class for Scikit-Learn RandomForestRegressor
    """
    # class wide variables common to all instances
    # that keep track of cumulative estimators and rmse
    # so we can plot the results to see how it fairs w.r.t
    # rmse
    rmse = []
    estimators = []

    def __init__(self, params={}):
        """
        Constructor for the RandomForestRegressor
        :param params: dictionary to RandomForestRegressor
        """
        # unpack dictionary as kwargs to the class constructor
        self._rf = RandomForestRegressor(**params)
        self._params = params

    @classmethod
    def new_instance(cls, params={}):
        return cls(params)

    @property
    def model(self):
        """
        Getter to return the model created
        :return: handle or instance of the RandomForestReqgressor
        """
        return self._rf

    @property
    def params(self):
      """
      Getter for model parameters 
      """
      return self._params
      
    def mlflow_run(self, df, run_name="Regression Base Model", verbose=False):
        """
        This method trains, computes metrics, and logs all metrics, parameters,
        and artifacts for the current run using the MLflow APIs
        :param df: pandas dataFrame
        :param r_name: Name of the run as logged by MLflow
        :return: MLflow Tuple (ExperimentID, runID)
        """

        with mlflow.start_run(run_name=run_name) as run:

            # get current run and experiment id
            runID = run.info.run_uuid
            experimentID = run.info.experiment_id
            
            # extract all feature independent attributes
            X = df.iloc[:, 0:4].values
            # extract all the values of last columns, dependent variables,
            # which is what we want to predict as our values, the petrol consumption
            y = df.iloc[:, 4].values

            # create train and test data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            # Feature Scaling, though for RF is not necessary.
            # z = (X - u)/ s, where u is the mean, s the standard deviation
            # get the handle to the transformer
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            # train and predict
            self._rf.fit(X_train, y_train)
            y_pred = self._rf.predict(X_test)

            # Log model and params using the MLflow APIs
            mlflow.sklearn.log_model(self.model, "random-forest-reg-model")
            mlflow.log_params(self.params)

            # compute  regression evaluation metrics 
            mae = metrics.mean_absolute_error(y_test, y_pred)
            mse = metrics.mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Log metrics
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)

            # update global class instance variable with values
            self.rmse.append(rmse)
            self.estimators.append(self.params["n_estimators"])

            # plot graphs and save as artifacts
            (fig, ax) = Utils.plot_graphs(self.estimators, self.rmse, "Random Forest Estimators", "Root Mean Square", "Root Mean Square vs Estimators")

            # create temporary artifact file name and log artifact
            temp_file_name = Utils.get_temporary_directory_path("rmse_estimators-", ".png")
            temp_name = temp_file_name.name
            try:
                fig.savefig(temp_name)
                mlflow.log_artifact(temp_name, "rmse_estimators_plots")
            finally:
                temp_file_name.close()  # Delete the temp file

            # print some data if verbose mode
            if verbose: 
              print("-" * 100)
              print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))
              print("Estimator trees        :", self.params["n_estimators"])
              print('Mean Absolute Error    :', mae)
              print('Mean Squared Error     :', mse)
              print('Root Mean Squared Error:', rmse)
            
            return (experimentID, runID)

# COMMAND ----------

displayHTML("""
<div>Declared RandomForestRegress Model Class with methods:</div>
<li>Declared <b style="color:green">model()</b> returns existing instance of Random Forest Model</li>
<li>Declared <b style="color:green">new_instance(params={...}</b></li> returns a new instance of RandomForestClassifierModel</li> 
<li>Declared <b style="color:green"> mlflow_run(DataFrame, run_name="name")</b> returns experiment_ID, run_ID</li>

<br/>
""")
