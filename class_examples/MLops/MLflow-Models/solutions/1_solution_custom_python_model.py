# Databricks notebook source
import mlflow.pyfunc

# COMMAND ----------

class FahrenheitToCelius(mlflow.pyfunc.PythonModel):
  
  def __init__(self):
    pass
  
  def _f2c(self, f):
    return (f - 32) * 5 / 9
  
  def predict(self, context, model_input):
    return model_input.apply(lambda f: self._f2c(f))

# COMMAND ----------

# Construct and save the model
model_path = "f_2_c_model2"
f2c_model = FahrenheitToCelius()
mlflow.pyfunc.save_model(path=model_path, python_model=f2c_model)

# COMMAND ----------

# Load the model in `python_function` format
loaded_model = mlflow.pyfunc.load_model(model_path)

# COMMAND ----------

import pandas as pd
model_input = pd.DataFrame([212, 205, 100])
model_output = loaded_model.predict(model_input)
model_output
