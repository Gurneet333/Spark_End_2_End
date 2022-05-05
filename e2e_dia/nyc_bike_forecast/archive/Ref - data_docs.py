# Databricks notebook source
# MAGIC %run ./includes/configuration

# COMMAND ----------

# MAGIC %run ./includes/utilities

# COMMAND ----------

# MAGIC %md
# MAGIC ##Weather API Fields
# MAGIC - hourly.dt Time of historical data, Unix, UTC
# MAGIC - hourly.temp Temperature. Units – default: kelvin, metric: Celsius, imperial: Fahrenheit. How to change units used
# MAGIC - hourly.feels_like Temperature. This accounts for the human perception of weather. Units – default: kelvin, metric: Celsius, imperial: Fahrenheit.
# MAGIC - hourly.pressure Atmospheric pressure on the sea level, hPa
# MAGIC - hourly.humidity Humidity, %
# MAGIC - hourly.dew_point Atmospheric temperature (varying according to pressure and humidity) below which water droplets begin to condense and dew can form. Units – default: kelvin, metric: Celsius, imperial: Fahrenheit.
# MAGIC - hourly.clouds Cloudiness, %
# MAGIC - hourly.visibility Average visibility, metres
# MAGIC - hourly.wind_speed Wind speed. Wind speed. Units – default: metre/sec, metric: metre/sec, imperial: miles/hour. How to change units used
# MAGIC - hourly.wind_gust (where available) Wind gust. Units – default: metre/sec, metric: metre/sec, imperial: miles/hour. How to change units used
# MAGIC - chourly.wind_deg Wind direction, degrees (meteorological)
# MAGIC - hourly.rain (where available) Precipitation volume, mm
# MAGIC - hourly.snow (where available) Snow volume, mm
# MAGIC - hourly.weather
# MAGIC   - hourly.weather.id Weather condition id
# MAGIC   - hourly.weather.main Group of weather parameters (Rain, Snow, Extreme etc.)
# MAGIC   - hourly.weather.description Weather condition within the group (full list of weather conditions). Get the output in your language
# MAGIC   - hourly.weather.icon Weather icon id. How to get icons

# COMMAND ----------

40.641311, 	-73.778139 
40.776927, 	-73.873966 
42.365613, 	-71.009560 
39.055351, 	-84.654504
42.208569, 	-83.355562 
33.641533, 	-84.444516 
44.880732, 	-93.204932 
40.784128, 	-111.980470 
47.439278, 	-122.315614 
33.941589, 	-118.408530 
39.856096, 	-104.673738
29.986136, 	-95.336586
37.614871, 	-122.389852
40.7129, -74.006

# COMMAND ----------

r=requests.get(WEATHER_FORECAST_JSON)

# COMMAND ----------

df1= pd.json_normalize(json.loads(r.text))
df2= pd.json_normalize(json.loads(r.text)['hourly'])
df3= pd.json_normalize(df2['weather'].apply(lambda x: x[0]))

pd.concat(df1,df2, axis=1)
#df= pd.concat([pd.DataFrame(json_dict), pd.DataFrame(list(json_dict['nested_array_to_expand']))], axis=1).drop('nested_array_to_expand', 1)


# COMMAND ----------

df1

# COMMAND ----------

df2.drop('weather', axis=1).join(df3)

# COMMAND ----------

df=df1.drop("hourly", axis=1).join(df2.drop('weather', axis=1)).join(df3)

# COMMAND ----------

  if len(r.text)>0:
    tzo = pd.json_normalize(json.loads(r.text))['timezone_offset'].values[0]
    df=pd.json_normalize(json.loads(r.text), max_level=4)
    df['time']=df['dt'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S',  time.gmtime(x+tzo)))

# COMMAND ----------

pd.json_normalize(df['weather'].loc[])

# COMMAND ----------



# COMMAND ----------

#Read the weather data for New York from S3 (temp and description)
import databricks.koalas as ks
import numpy as np

weather_dimensions=[('humidity.csv','humidity_pct'), ('temperature.csv','temp_f'), ('pressure.csv','pressure_hpa'), ('weather_description.csv','desc'), ('wind_speed.csv','wind_speed_mps'), ('wind_direction.csv','wind_dir')]

def read_file(filename):
  spark.sparkContext.addFile(create_s3_path('weather-data-feed')+filename)
  return spark.read.csv("file://"+SparkFiles.get(filename), header=True, inferSchema= True)

dfs = [read_file(x[0]) for x in weather_dimensions]
# this allows us to form the dataframe by joining one dimension at a time
ks.set_option('compute.ops_on_diff_frames', True)
# read the first dimension
kdf = ks.melt(dfs[0].to_koalas(), id_vars=['datetime'], value_vars=['New York']).drop("variable").rename(columns={'value':weather_dimensions[0][1]}).astype({'datetime': np.datetime64})
# join the rest
for d,df in zip(weather_dimensions[1:],dfs[1:]):
  name = d[1]
  kdf[name]=ks.melt(df.to_koalas(), id_vars=['datetime'], value_vars=['New York'])['value']
#Convert kelvin to F (K − 273.15) × 9/5 + 32 
kdf['temp_f']=((kdf['temp_f']-273.15) * (9/5)) + 32 

DELTA_TABLE_NY_WEATHER = BASE_DELTA_TABLE + "BRONZE_NYC_WEATHER"

# delete the old delta table
dbutils.fs.rm(DELTA_TABLE_NY_WEATHER, recurse=True)

# Write a new delta table
kdf.to_delta(DELTA_TABLE_NY_WEATHER, mode='overwrite')

drop_make("e2e_nyc_weather", DELTA_TABLE_NY_WEATHER)
