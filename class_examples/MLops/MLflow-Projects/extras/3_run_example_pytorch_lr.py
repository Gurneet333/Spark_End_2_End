# Databricks notebook source
# MAGIC %md ### PyTorch Linear Regressoin NN Example 
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><td>
# MAGIC     <img src="https://raw.githubusercontent.com/dmatrix/mlflow-workshop-project-expamle-1/master/images/temperature-conversion.png"
# MAGIC          alt="Bank Note " width="400">
# MAGIC   </td></tr>
# MAGIC </table>

# COMMAND ----------

import os
import numpy as np
import random
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# COMMAND ----------

class FahrenheitTemperatures(Dataset):
    def __init__(self, start=-50, stop=300, size=2000):
        super(FahrenheitTemperatures, self).__init__()
        
        # Intialize local variables and scale the data
        np.random.seed(42)
        f_temp = np.random.randint(start, high=stop, size=size).reshape(-1, 1)
        c_temp = np.array([self._f2c(f) for f in f_temp]).reshape(-1, 1)
        
        # convert to tensors
        self.X = torch.from_numpy(f_temp).float()
        self.y = torch.from_numpy(c_temp).float()
        self._samples = self.X.shape[0]
        
    def __getitem__(self, index):
        # support indexing such that dataset[i] can be used to get i-th sample
        # implement this python function for indexing
        return self.X[index], self.y[index] 
        
    def __len__(self):
        # we can call len(dataset) to return the size, so this can be used
        # as an iterator
        return self._samples
    
    def _f2c(self, f) -> float:
        return (f - 32) * 5.0/9.0
    
    @property
    def samples(self):
        return self._samples

# COMMAND ----------

class LinearNN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearNN, self).__init__()
        
        # Input, output, and hidden size paramaters
        self.input_size = input_size
        self.output_size = output_size
        
        # Build the NN architecture
        self.l1 = torch.nn.Linear(input_size, output_size)
        
    def forward(self, x):
        out = self.l1(x)
        return out

# COMMAND ----------

# Let's now access our dataset
dataset = FahrenheitTemperatures()
first_dataset = dataset[0]
features, labels = first_dataset
samples = dataset.samples
print('Fahrenheit: {:.4f}'.format(features[0]))
print('Celcius   : {:.4f}'.format(labels[0]))
print('Samples   : {:.2f}'.format(samples))

# COMMAND ----------

data_loader = DataLoader(dataset=dataset, batch_size=5, shuffle=True)

# COMMAND ----------

# our model, loss function, and optimizer
model = LinearNN(1, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)

# COMMAND ----------

# Training loop
epochs= 1500
for epoch in range(epochs):
    for i, (inputs, labels) in enumerate(data_loader):
        
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(inputs)
        
        # Compute and print loss
        loss = criterion(y_pred, labels)
        
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        print('Epoch: {}/{}, loss: {:.4f}, Weight: {:.4f}, bias: {:.4f}'
              .format(epoch + 1., epochs, loss.data.item(), model.l1.weight.item(), model.l1.bias.item()))

# COMMAND ----------

def f2c(f):
  return (f - 32) * 5.0/9.0

f_temp = np.arange(212, 185, -5)
for f in f_temp:
  print('F: {} ->C {:.2f}'.format(f, f2c(f)))

# COMMAND ----------

from sklearn import preprocessing

with torch.no_grad():
    f_temp = np.arange(212, 185, -5).reshape(-1, 1)
    y_pred = model(torch.from_numpy(f_temp).float())
    print(y_pred)

# COMMAND ----------

with torch.no_grad():
  for f in [32, 212, 100, 90, -50, -32]:
    c_pred = model(torch.tensor([f]).float())
    print('F: {} ->Converted C: {:.2f},  Predicted C: {:.2f}'.format(f, f2c(f), c_pred.item()))

# COMMAND ----------

# MAGIC %md ### Use our learned weights and bias in the function to predict.
# MAGIC 
# MAGIC `y = mX +c`
# MAGIC 
# MAGIC `y = weight*X + bias`

# COMMAND ----------

def linear_func(X, m, c):
  y = m*X + c
  return y

for f in [32, 212, 100, 90, -50, -32]:
    print('F: {} ->Predicted C: {:.2f}'.format(f, linear_func(f, model.l1.weight.item(),model.l1.bias.item())))

# COMMAND ----------

# MAGIC %md ### Homework Excercise:
# MAGIC * Convert it into an MLflow Project
# MAGIC  * Use [MLflow Keras Project Example](https://github.com/dmatrix/mlflow-workshop-project-expamle-1) as a reference
# MAGIC * Load the model as PyFunc and predict
# MAGIC * Use the `2_run_example_keras_lr` as a template

# COMMAND ----------


