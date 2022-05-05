# Databricks notebook source
# MAGIC %md #### Sentiment Analysis
# MAGIC 
# MAGIC We want to do sentiment analysis by using [VaderSentiment ML framework](https://medium.com/analytics-vidhya/simplifying-social-media-sentiment-analysis-using-vader-in-python-f9e6ec6fc52f) not supported as an MLflow Flavor.
# MAGIC The goal of sentiment analysis is to "gauge the attitude, sentiments, evaluations, attitudes and emotions of a speaker/writer based on the computational treatment of subjectivity in a text."
# MAGIC 
# MAGIC VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media.
# MAGIC 
# MAGIC VADER has a lot of advantages over traditional methods of Sentiment Analysis, including:
# MAGIC 
# MAGIC 
# MAGIC  * It works exceedingly well on social media type text, yet readily generalizes to multiple domains
# MAGIC  * It doesn’t require any training data but is constructed from a generalizable, valence-based, human-curated gold standard sentiment lexicon
# MAGIC  * It is fast enough to be used online with streaming data, and
# MAGIC  * It does not severely suffer from a speed-performance tradeoff.
# MAGIC 
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><td>
# MAGIC     <img src="https://github.com/dmatrix/olt-mlflow/raw/master/models/images/sentiment_analysis.jpg"
# MAGIC          alt="Sentiment Analysis with Vader " width="600">
# MAGIC   </td></tr>
# MAGIC </table>
# MAGIC 
# MAGIC [image source](https://medium.com/analytics-vidhya/sentiment-analysis-with-vader-label-the-unlabeled-data-8dd785225166)

# COMMAND ----------

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

# MAGIC %md #### Homework Assignment
# MAGIC 
# MAGIC  * Create some text input
# MAGIC  * load the model as a PyFuncModel
# MAGIC  * Invoke its predict method with each text
# MAGIC  * Do the scores change if you remove emojis and chat speak" or "text speak?
# MAGIC  * Print the scores

# COMMAND ----------

# MAGIC %md ### Homework Challenge
# MAGIC  * Try using a new library such as NLTK
# MAGIC  * Can you write a PythonModel?
# MAGIC  * Use MLflow to log parameters and metrics.
# MAGIC  
# MAGIC  Resources:
# MAGIC   * [Basic Sentence Analysis with NLTK](https://towardsdatascience.com/basic-binary-sentiment-analysis-using-nltk-c94ba17ae386)
# MAGIC   * [Python Tutorials on NLTK](https://www.youtube.com/watch?v=FLZvOKSCkxY&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL)
