# Databricks notebook source
# MAGIC %pip install vaderSentiment

# COMMAND ----------

# MAGIC %md ### VaderSentiment Python Package
# MAGIC 
# MAGIC You can read the orignal paper by authors [here](http://comp.social.gatech.edu/papers/icwsm14.vader.hutto.pdf).

# COMMAND ----------

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import mlflow.pyfunc

# COMMAND ----------

# Define some input text

INPUT_TEXTS = [{'text': "This is a bad ass movie. You got to see it! :-)"},
               {'text': "Ricky Gervais is smart, witty, and creative!!!!!! :D"},
               {'text': "LOL, this guy fell off a chair while sleeping and snoring in a meeting"},
               {'text': "Men shoots himself while trying to steal a dog, OMG"},
               {'text': "Yay!! Another good phone interview. I nailed it!!"},
               {'text': "This is INSANE! I can't believe it. How could you do such a horrible thing?"}]

# COMMAND ----------

# MAGIC %md ### Define a SocialMediaAnalyserModel
# MAGIC 
# MAGIC This is a subclass of [PythonModel](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.PythonModel)

# COMMAND ----------

class SocialMediaAnalyserModel(mlflow.pyfunc.PythonModel):

   def __init__(self):
      """
      Constructor for our Cusomized PyFunc Model Class
      """
      # Initialize an instance of vader analyser
      self._analyser = SentimentIntensityAnalyzer()

   def _score(self, text):
    """
    Private function to analyse the scores. It invokes model's 
    param: text to analyse
    return: sentiment analyses scores
    """
    scores = self._analyser.polarity_scores(text)
    return scores

   def predict(self, context, model_input):
    """
    Implement the predict function required for PythonModel
    """
    model_output = model_input.apply(lambda col: self._score(col))
    return model_output

# COMMAND ----------

def mlflow_run():
  
  # Save the conda environment for this model. 
  conda_env = {
    'channels': ['defaults', 'conda-forge'],
    'dependencies': [
        'python=3.7.6',
        'pip'],
    'pip': [
      'mlflow',
      'cloudpickle==1.3.0',
      'vaderSentiment==3.3.2'
    ],
    'name': 'mlflow-env'
  }
  
  # Model name and create an instance of PyFuncModel
  model_path = "vader"
  vader_model = SocialMediaAnalyserModel()
  with mlflow.start_run(run_name="Vader Sentiment Analysis") as run:
    # Log MLflow entities
    mlflow.pyfunc.log_model(model_path, python_model=vader_model, conda_env=conda_env)
    mlflow.log_param("algorithm", "VADER")
    mlflow.log_param("total_sentiments", len(INPUT_TEXTS))

    # Load back the model as a pyfunc_model for scoring
    model_uri = f"runs:/{run.info.run_uuid}/{model_path}"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    
    # Use predict function to score output from the model
    for i in range(len(INPUT_TEXTS)):
       text = INPUT_TEXTS[i]['text']
       mlflow.log_param(f"text_{i}", text)
      
      # Use predict function to score output from the model
       model_input = pd.DataFrame([text])
       scores = loaded_model.predict(model_input)
       print(f"<{text}> ------- {str(scores[0])}>")
       for index, value in scores.items():
          [mlflow.log_metric(f"{key}_{i}", value) for key, value in value.items()]
          
    return run.info.run_id

# COMMAND ----------

# MAGIC %md Log parameters, metrics, and model with MLflow APIs

# COMMAND ----------

 mlflow_run()

# COMMAND ----------

# MAGIC %md #### Excercise Assignment
# MAGIC 
# MAGIC  * Create some text input
# MAGIC  * load the model as a PyFuncModel
# MAGIC  * Invoke its predict method with each text
# MAGIC  * Do the scores change if you remove emojis and chat speak" or "text speak?
# MAGIC  * Print the scores

# COMMAND ----------

# MAGIC %md ### Solution

# COMMAND ----------

# MAGIC %md #### Create Test Data

# COMMAND ----------

# Define some input text

TEST_TEXTS = [
               {'text': "Ricky Gervais could not be any funnier and brilliant! smart :D"},
               {'text': "LOL, this guy snores so loud, he could wake up the dead from the graves!"},
               {'text': "Men shoots himself while trying to steal a gun, OMG. How dumb is that?"}
              ]
TEST_TEXTS_NO_EMOJIS = [
               {'text': "Ricky Gervais could not be any funnier and brilliant! smart"},
               {'text': "This guy snores so loud, he could wake up the dead from the graves!"},
               {'text': "Men shoots himself while trying to steal a gun. How dumb is that?"}
               ]

# COMMAND ----------

# MAGIC %md #### Load PyfuncModel

# COMMAND ----------

# Load back the model as a pyfunc_model for scoring
run_id = "<INSERT_YOUR_RUNID_HERE>"
model_path ="vader"
model_uri = f"runs:/{run_id}/{model_path}"
pyfunc_model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

# MAGIC %md ### Score the model

# COMMAND ----------

# Use predict function to score output from the model
def score_model(data, model):
  for i in range(len(data)):
     text = data[i]['text']
     # Use predict function to score output from the model
     model_input = pd.DataFrame([text])
     scores = model.predict(model_input)
     print(f"<{text}> ------- {str(scores[0])}>")

# COMMAND ----------

score_model(TEST_TEXTS, pyfunc_model)

# COMMAND ----------

score_model(TEST_TEXTS_NO_EMOJIS, pyfunc_model)

# COMMAND ----------


