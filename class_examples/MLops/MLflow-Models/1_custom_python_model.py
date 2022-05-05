# Databricks notebook source
# MAGIC %md ### Custom Python Models
# MAGIC 
# MAGIC The mlflow.pyfunc module provides [save_model()](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.save_model) and [log_model()](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.log_model) utilities for creating MLflow Models with the `python_function` flavor that contain user-specified code and _artifact_ (file) dependencies. These artifact dependencies may include serialized models produced by any Python ML library.
# MAGIC 
# MAGIC Because these custom models contain the [python_function flavor](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#module-mlflow.pyfunc), they can be deployed to any of MLflow’s supported production environments, such as SageMaker, AzureML, or local REST endpoints.
# MAGIC 
# MAGIC The following examples demonstrate how you can use the mlflow.pyfunc module to create custom Python models. For additional information about model customization with MLflow’s python_function utilities, see the [python_function custom models documentation](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#pyfunc-create-custom).
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><td>
# MAGIC     <img src="https://github.com/dmatrix/olt-mlflow/raw/master/models/images/mlflow-models-python-model.png"
# MAGIC          alt="Sentiment Analysis with Vader " width="600">
# MAGIC   </td></tr>
# MAGIC </table>

# COMMAND ----------

import mlflow.pyfunc

# COMMAND ----------

# DBTITLE 1,Define PythonModel class
# MAGIC %md [PythonModel](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.PyFuncModel) is an abstract class

# COMMAND ----------

class AddN(mlflow.pyfunc.PythonModel):
  # Constructor
  def __init__(self, n):
        self.n = n
      
  # Implement its abstract method; this is your custom logic to
  # preprocess or post process your input
  def predict(self, context, model_input):
        return model_input.apply(lambda column: column + self.n)

# COMMAND ----------

# Construct and save the model
model_path = "add_n_model1"
add5_model = AddN(n=5)
mlflow.pyfunc.save_model(path=model_path, python_model=add5_model)

# COMMAND ----------

# Load the model in `python_function` format
loaded_model = mlflow.pyfunc.load_model(model_path)

# COMMAND ----------

# Evaluate the model
import pandas as pd
model_input = pd.DataFrame([range(10)])
model_output = loaded_model.predict(model_input)
assert model_output.equals(pd.DataFrame([range(5, 15)]))
