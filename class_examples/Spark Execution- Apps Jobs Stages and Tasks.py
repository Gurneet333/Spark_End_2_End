# Databricks notebook source
# MAGIC %md
# MAGIC <img src="https://data-science-at-scale.s3.amazonaws.com/data-science-at-scale/images/app_job_stage_task/1.png" width="640">

# COMMAND ----------

# MAGIC %md
# MAGIC <p>Consider this illustration of a spark cluster with a driver (all clusters have one) where your applications runs 
# MAGIC and 4 worker/executors and 8 cores.  Note: that the numbers of workers and cores are configurable based on the size of cluster you are working with.
# MAGIC It could be 100s/1000s down to 1 worker coresident with the driver like the Databricks Community edition.</p><br>
# MAGIC <img src="https://data-science-at-scale.s3.amazonaws.com/data-science-at-scale/images/app_job_stage_task/2.png" width="640">
# MAGIC <p>When we are running in the context of a notebook, each cell of the notebook can comprise one or more jobs, stages and tasks that are 
# MAGIC   scheduled to run on the workers of the cluster [Physcial Job Planning].  </p>
# MAGIC <img src="https://data-science-at-scale.s3.amazonaws.com/data-science-at-scale/images/app_job_stage_task/3.png" width="800">
# MAGIC   
# MAGIC <p>Consider the following definitions:</p>
# MAGIC - **Job**
# MAGIC  - The work required to compute an RDD
# MAGIC - **Stage**
# MAGIC  - A series of work within a job to produce one or  more pipelined RDD’s
# MAGIC - **Tasks**
# MAGIC  - A unit of work within a stage, corresponding to one  RDD partition
# MAGIC - **Shuffle**
# MAGIC  - The transfer of data between stages
# MAGIC 
# MAGIC <img src="https://data-science-at-scale.s3.amazonaws.com/data-science-at-scale/images/app_job_stage_task/4.png" width="640">
# MAGIC 
# MAGIC 
# MAGIC In the next cell we can try a simple example job that includes a set of *transforms*:
# MAGIC - **parallelize** (create an RDD)
# MAGIC - **union** (combine RDD)
# MAGIC - **groupBy** (group by first letter of the word)
# MAGIC - **filter**     (remove the first letter of )
# MAGIC 
# MAGIC followed by a single *action*:
# MAGIC - **collect** 
# MAGIC 
# MAGIC This illustration outlines the steps that spark follows to build and schedule this kind of job<br>
# MAGIC <img src="https://data-science-at-scale.s3.amazonaws.com/data-science-at-scale/images/app_job_stage_task/5.png" width="800">
# MAGIC 
# MAGIC Shuffling is worth a specific mention here.  There are two kinds of shuffling between stages of the DAG.  Wide shuffling (grouping) is more expensive that the narrow shuffling (filtering):
# MAGIC <table broder=0><tr><td>
# MAGIC   <img src="https://data-science-at-scale.s3.amazonaws.com/data-science-at-scale/images/app_job_stage_task/6.png" width="400">
# MAGIC   </td>
# MAGIC   <td><img src="https://data-science-at-scale.s3.amazonaws.com/data-science-at-scale/images/app_job_stage_task/7.png" width="400">
# MAGIC   </td></tr></table>

# COMMAND ----------

# DBTITLE 1,Example of a simple spark job
# creating RDD x with fruits and names
a = spark.sparkContext.parallelize(["Apple", "Orange", "Pineapple", "Kiwi", "Banana", "Grape",  "Date", "Pomeganate", "Mango"], 3)
b = spark.sparkContext.parallelize(["Allan", "Oliver", "Paula", "Karen", "James", "Cory", "Christine", "Jackeline", "Juan"], 3)

# take the union of the two RDD
x = a.union(b)

# Applying groupBy operation on x
y = x.groupBy(lambda name: name[0])

# filter out names that start with J and P
z = y.filter(lambda name: name[0]!='J' and name[0]!='P')

for t in z.collect():
    print((t[0],[i for i in t[1]]))
 

#TODO After you run this cell click on the "view" link next to the Job to see the spark UI that outlines how the job was planned and scheduled on the cluster

# COMMAND ----------

# MAGIC %md
# MAGIC ## Multi-stage pipelines
# MAGIC Let's look at a little more complicated example of a multi-stage Job that reads in the works of Sherlock Holmes and counts the occurance of each word in the text.
# MAGIC <p>We are going to look at two different ways to perform this job.<p>
# MAGIC   - Resilient Distributed Data (RDD) based job (low level spark api)<br>
# MAGIC   <img src="https://data-science-at-scale.s3.amazonaws.com/data-science-at-scale/images/app_job_stage_task/10.png" width="800">
# MAGIC   - Dataframe (DF) based job (higher level api that we will use primarily in ths class)

# COMMAND ----------

# DBTITLE 1,Read the sherlock.txt file 
mount_name = "dscc202-datasets"
s3_bucket = "s3a://dscc202-datasets/"
try:
  dbutils.fs.mount(s3_bucket, "/mnt/%s" % mount_name)
except:
  dbutils.fs.unmount("/mnt/%s" % mount_name)
  dbutils.fs.mount(s3_bucket, "/mnt/%s" % mount_name)

display(dbutils.fs.ls("/mnt/%s" % mount_name)  )

# COMMAND ----------

# DBTITLE 1,Read the file into a RDD
# Read in the works of sherlock holmes
lines = spark.read.text("/mnt/dscc202-datasets/misc/sherlock.txt").rdd.map(lambda r: r[0])

# COMMAND ----------

# DBTITLE 1,Take a look at the first 10 lines of the file
lines.take(10)

# COMMAND ----------

# DBTITLE 1,Perform the RDD based job
"""
Steps
=====
- split each line in the file up into individual words (separated by spaces)
- filter out words that are less than 1 character
- create a tuple for each word that looks like (word, 1)
- for each unique word add up all of the occurances (reduce)
- sort the results in descending order
- take a look at the highest 10 occuring words
"""
lines.flatMap(lambda x: x.split(' ')) \
              .filter(lambda x: len(x) >= 1) \
              .map(lambda x: (x, 1)) \
              .reduceByKey(lambda a,b: a+b) \
              .sortBy(lambda x: x[1], False) \
              .take(10)

# COMMAND ----------

# DBTITLE 1,Read the sherlock.txt file into a Dataframe
from pyspark.sql.functions import *
lines = spark.read.text("/mnt/dscc202-datasets/misc/sherlock.txt")
lines.take(10)

# Note the data type of each Row and value as opposed to the list above

# COMMAND ----------

"""
Steps
=====
- split each line in the file up into a new column called words separated by spaces and explode into new rows of the dataframe
- filter out words that are less than 1 character
- groupby unique words 
- count the occurance of each unique word --> creates a new column called count
- sort the results in descending order
- take a look at the highest 10 occuring words
"""
lines.withColumn('word', explode(split(col('value'), ' ')))\
    .filter("word != ''")\
    .groupBy('word')\
    .count()\
    .sort('count', ascending=False)\
    .show(10)
