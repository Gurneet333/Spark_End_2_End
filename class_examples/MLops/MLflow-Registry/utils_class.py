# Databricks notebook source
import pandas as pd
import tempfile
import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

class Utils:
   @staticmethod
   def load_data(path, index_col=0):
      df = pd.read_csv(path, index_col=0)
      df.index = pd.to_datetime(df.index)
      return df

   @staticmethod
   def get_training_data(df):
      training_data = pd.DataFrame(df.loc["01 01 2018 00:00":"07 01 2018 00:00"])
      X = training_data.drop(columns="LV ActivePower (kW)")
      y = training_data["LV ActivePower (kW)"]
      return X, y

   @staticmethod
   def get_validation_data(df):
      validation_data = pd.DataFrame(df.loc["07 01 2018 00:00":"12 01 2018 00:00"])
      X = validation_data.drop(columns="LV ActivePower (kW)")
      y = validation_data["LV ActivePower (kW)"]
      return X, y

   @staticmethod
   def get_temporary_directory_path(prefix, suffix):
      """
      Get a temporary directory and files for artifacts
      :param prefix: name of the file
      :param suffix: .csv, .txt, .png etc
      :return: object to tempfile.
      """

      temp = tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix)
      return temp

   @staticmethod
   def print_pandas_dataset(d, n=5):
      """
      Given a Pandas dataFrame show the dimensions sizes
      :param d: Pandas dataFrame
      :return: None
      """
      print("rows = %d; columns=%d" % (d.shape[0], d.shape[1]))
      print(d.head(n))
