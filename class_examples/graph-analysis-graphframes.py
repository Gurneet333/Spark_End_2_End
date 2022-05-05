# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Graph Analysis with GraphFrames
# MAGIC This notebook goes over basic graph analysis using [the GraphFrames package available on spark-packages.org](http://spark-packages.org/package/graphframes/graphframes). The goal of this notebook is to show you how to use GraphFrames to perform graph analysis. You're going to be doing this with Bay area bike share data from [Kaggle](https://www.kaggle.com/benhamner/sf-bay-area-bike-share/downloads/sf-bay-area-bike-share.zip).
# MAGIC 
# MAGIC #### Graph Theory and Graph Processing
# MAGIC Graph processing is an important aspect of analysis that applies to a lot of use cases. Fundamentally graph theory and processing are about defining relationships between different nodes and edges. Nodes or vertices are the units while edges are the relationships that are defined between those. This works great for social network analysis and running algorithms like [PageRank](https://en.wikipedia.org/wiki/PageRank) to better understand and weigh relationships.
# MAGIC 
# MAGIC Some business use cases could be to look at central people in social networks [who is most popular in a group of friends], importance of papers in bibliographic networks [which papers are most referenced], and of course ranking web pages!
# MAGIC 
# MAGIC #### Graphs and Bike Trip Data
# MAGIC As mentioned, in this example you'll be using Bay area bike share data. The way you're going to orient your analysis is by making every vertex a station and each trip will become an edge connecting two stations. This creates a *directed* graph.
# MAGIC 
# MAGIC **Further Reference:**
# MAGIC * [Graph Theory on Wikipedia](https://en.wikipedia.org/wiki/Graph_theory)
# MAGIC * [PageRank on Wikipedia](https://en.wikipedia.org/wiki/PageRank)
# MAGIC 
# MAGIC #### **Table of Contents**
# MAGIC * **Create DataFames**
# MAGIC * **Imports**
# MAGIC * **Building the Graph**
# MAGIC * **PageRank**
# MAGIC * **Trips from Station to Station**
# MAGIC * **In Degrees and Out Degrees**

# COMMAND ----------

# MAGIC %md
# MAGIC ### Mount Datasets

# COMMAND ----------

mount_name = "dscc202-datasets"
s3_bucket = "s3a://dscc202-datasets/"
try:
  dbutils.fs.mount(s3_bucket, "/mnt/%s" % mount_name)
except:
  dbutils.fs.unmount("/mnt/%s" % mount_name)
  dbutils.fs.mount(s3_bucket, "/mnt/%s" % mount_name)

# COMMAND ----------

# MAGIC %md ### Create DataFrames

# COMMAND ----------

bikeStations = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/mnt/dscc202-datasets/sf_bikes/station.csv")
tripData = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/mnt/dscc202-datasets/sf_bikes/trip.csv")

# COMMAND ----------

display(bikeStations)

# COMMAND ----------

display(tripData)

# COMMAND ----------

# MAGIC %md It can often times be helpful to look at the exact schema to ensure that you have the right types associated with the right columns.

# COMMAND ----------

bikeStations.printSchema()
tripData.printSchema()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Imports
# MAGIC You're going to need to import several things before you can continue. You're going to import a variety of SQL functions that are going to make working with DataFrames much easier and you're going to import everything that you're going to need from GraphFrames.

# COMMAND ----------

from pyspark.sql.functions import *
from graphframes import *

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build the Graph
# MAGIC Now that you've imported your data, you're going to need to build your graph. To do so you're going to do two things. You are going to build the structure of the vertices (or nodes) and you're going to build the structure of the edges. What's awesome about GraphFrames is that this process is incredibly simple. All that you need to do get the distinct **id** values in the Vertices table and rename the start and end stations to **src** and **dst** respectively for your edges tables. These are required conventions for vertices and edges in GraphFrames.

# COMMAND ----------

stationVertices = bikeStations.distinct()

tripEdges = (tripData
  .withColumnRenamed("start_station_name", "src")
  .withColumnRenamed("end_station_name", "dst"))

# COMMAND ----------

display(stationVertices)

# COMMAND ----------

display(tripEdges)

# COMMAND ----------

# MAGIC %md Now you can build your graph. 
# MAGIC 
# MAGIC You're also going to cache the input DataFrames to your graph.

# COMMAND ----------

stationGraph = GraphFrame(stationVertices, tripEdges)

tripEdges.cache()
stationVertices.cache()

# COMMAND ----------

print("Total Number of Stations: {}".format(stationGraph.vertices.count()))
print("Total Number of Trips in Graph: {}".format(stationGraph.edges.count()))
print("Total Number of Trips in Original Data: {}".format(tripData.count()))  # sanity check

# COMMAND ----------

# MAGIC %md
# MAGIC ### Trips From Station to Station
# MAGIC One question you might ask is what are the most common destinations in the dataset from location to location. You can do this by performing a grouping operator and adding the edge counts together. This will yield a new graph except each edge will now be the sum of all of the semantically same edges. Think about it this way: you have a number of trips that are the exact same from station A to station B, you just want to count those up!
# MAGIC 
# MAGIC In the below query you'll see that you're going to grab the station to station trips that are most common and print out the top 10.

# COMMAND ----------

topTrips = (stationGraph
  .edges
  .groupBy("src", "dst")
  .count()
  .orderBy(desc("count"))
  .limit(10))

display(topTrips)

# COMMAND ----------

# MAGIC %md You can see above that a given vertex being a Caltrain station seems to be significant! This makes sense as these are natural connectors and likely one of the most popular uses of these bike share programs to get you from A to B in a way that you don't need a car!

# COMMAND ----------

# MAGIC %md 
# MAGIC ### In Degrees and Out Degrees
# MAGIC Remember that in this instance you've got a directed graph. That means that your trips are directional - from one location to another. Therefore you get access to a wealth of analysis that you can use. You can find the number of trips that go into a specific station and leave from a specific station.
# MAGIC 
# MAGIC Naturally you can sort this information and find the stations with lots of inbound and outbound trips! Check out this definition of [Vertex Degrees](http://mathworld.wolfram.com/VertexDegree.html) for more information.
# MAGIC 
# MAGIC Now that you've defined that process, go ahead and find the stations that have lots of inbound and outbound traffic.

# COMMAND ----------

inDeg = stationGraph.inDegrees
display(inDeg.orderBy(desc("inDegree")).limit(5))

# COMMAND ----------

outDeg = stationGraph.outDegrees
display(outDeg.orderBy(desc("outDegree")).limit(5))

# COMMAND ----------

# MAGIC %md One interesting follow up question you could ask is what is the station with the highest ratio of in degrees but fewest out degrees. As in, what station acts as almost a pure trip sink. A station where trips end at but rarely start from.

# COMMAND ----------

display(inDeg)

# COMMAND ----------

display(inDeg.withColumnRenamed("id", "in_id"))

# COMMAND ----------

inDeg = inDeg.withColumnRenamed("id", "in_id")
degreeRatio = inDeg.join(outDeg, inDeg.in_id == outDeg.id, how='left').drop("in_id").select("id", (col("inDegree")/col("outDegree")).alias("degreeRatio"))
degreeRatio.cache()
display(degreeRatio.orderBy(desc("degreeRatio")).limit(10))

# COMMAND ----------

# MAGIC %md 
# MAGIC You can do something similar by getting the stations with the lowest in degrees to out degrees ratios, meaning that trips start from that station but don't end there as often. This is essentially the opposite of what you have above.

# COMMAND ----------

display(degreeRatio.orderBy(asc("degreeRatio")).limit(10))

# COMMAND ----------

# MAGIC %md The conclusions of what you get from the above analysis should be relatively straightforward. If you have a higher value, that means many more trips come into that station than out, and a lower value means that many more trips leave from that station than come into it!
# MAGIC 
# MAGIC Hopefully you've gotten some value out of this notebook! Graph stuctures are everywhere once you start looking for them and hopefully GraphFrames will make analyzing them easy!

# COMMAND ----------

display(stationGraph.edges)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Some flight data ...

# COMMAND ----------

df = spark.sql("select * from dscc202_db.bronze_air_traffic limit 100000")
display(df) 

# COMMAND ----------

# Create Vertices (airports) and Edges (flights)

tripVertices = df.withColumnRenamed("origin", "id").select("id").distinct()

tripEdges = df.select(col("origin_airport_seq_id").alias("tripId"),col("dep_delay").alias("delay"), col("origin").alias("src"), col("dest").alias("dst"), split(col("dest_city_name"), ',')[0].alias("city_dst"), col("dest_state_abr").alias("state_dst"))

# COMMAND ----------

tripGraph = GraphFrame(tripVertices,tripEdges)

# COMMAND ----------

print(f"Airports: {tripGraph.vertices.count()}")

print(f"Trips: {tripGraph.edges.count()}")

# COMMAND ----------

# For planes leaving SFO what are the average departure delays to a specific destination state
display(tripGraph.edges\
.filter("src = 'SFO' and delay > 0")\
.groupBy("src", "state_dst", "dst")\
.avg("delay")\
.sort(desc("avg(delay)")))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Motifs - Domain Specific Language for selecting within a graphframe using node and edge criteria
# MAGIC [Motif Docs](# https://graphframes.github.io/graphframes/docs/_site/user-guide.html#motif-finding)

# COMMAND ----------

# Find all of the nodes in the graph where a to c through b
motifs = tripGraph.find("(a)-[ab]->(b); (b)-[bc]->(c)").filter("(b.id = 'SFO') and (c.id = 'JFK') and (ab.delay > 500 or bc.delay > 500) and bc.tripid > ab.tripid and bc.tripid > ab.tripid + 10000")

display(motifs)

# COMMAND ----------

# Determining Airport ranking of importance using pageRank
ranks = tripGraph.pageRank(resetProbability=0.15, maxIter=5)

display(ranks.vertices.orderBy(ranks.vertices.pagerank.desc()).limit(50))

# COMMAND ----------

filteredPaths = tripGraph.bfs(fromExpr = "id = 'SFO'", toExpr = "id = 'BUF'", maxPathLength = 1)

display(filteredPaths)

# COMMAND ----------

filteredPaths = tripGraph.bfs(fromExpr = "id = 'SFO'", toExpr = "id = 'BUF'", maxPathLength = 2)

display(filteredPaths)
