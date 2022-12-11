# %% [markdown]
# # DATA603 Big Data Processing Project 
# Group 3: Pooja Kangokar Pranesh, Yun-Zih Chen, Elizabeth Cardosa
# 
# The goal of this project is leverage big data technologies to train a model using the UCI ML Drug Review dataset to predict the star rating of drug based on the sentiment of the review. This model will then perform inference in a streaming manner on ‘real-time’ reviews coming in. This application can then be used to help potential customers understand the overall sentiment towards a drug and if it might be useful for them. 
# 

# %%
# from google.colab import drive
# drive.mount('/content/drive')

# %%
working_folder = "gs://spark-drug-analysis/Model_Train/"

# # Install Libraries and Dependencies
import pyspark.pandas as ps
import pandas as pd

# %%
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark import SparkContext
# Create a Spark Session
spark = SparkSession.builder.appName("drug-analysis").getOrCreate()
# Check Spark Session Information
spark

# %%
sc = SparkContext.getOrCreate();

# %% [markdown]
# # Read-in Dataset
# 

# %% [markdown]
# ## Dataset: https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29
# 

# %% [markdown]
# The dataset provides patient reviews on specific drugs along with related conditions and a 10 star patient rating reflecting overall patient satisfaction. The data was obtained by crawling online pharmaceutical review sites. The intention was to study
# 
# - sentiment analysis of drug experience over multiple facets, i.e. sentiments learned on specific aspects such as effectiveness and side effects,
# - the transferability of models among domains, i.e. conditions, and
# - the transferability of models among different data sources (see 'Drug Review Dataset (Druglib.com)').
# 
# The data is split into a train (75%) a test (25%) partition (see publication) and stored in two .tsv (tab-separated-values) files, respectively.
# 
# Attribute Information:
# 
# 1. drugName (categorical): name of drug
# 2. condition (categorical): name of condition
# 3. review (text): patient review
# 4. rating (numerical): 10 star patient rating
# 5. date (date): date of review entry
# 6. usefulCount (numerical): number of users who found review useful
# 

# %% [markdown]
# Important notes:
# 
# When using this dataset, you agree that you
# 1. only use the data for research purposes
# 2. don't use the data for any commerical purposes
# 3. don't distribute the data to anyone else
# 4. cite us
# 
# Felix Gräßer, Surya Kallumadi, Hagen Malberg, and Sebastian Zaunseder. 2018. Aspect-Based Sentiment Analysis of Drug Reviews Applying Cross-Domain and Cross-Data Learning. In Proceedings of the 2018 International Conference on Digital Health (DH '18). ACM, New York, NY, USA, 121-125. DOI: [Web Link] 

# %% [markdown]
# ## Load in Test Data

# %%
# Read in training data file
customschema = StructType([
  StructField("UniqueID", IntegerType(), True)
  ,StructField("drugName", StringType(), True)
  ,StructField("condition", StringType(), True)
  ,StructField("review", StringType(), True)
  ,StructField("rating", DoubleType(), True)
  ,StructField("date", StringType(), True)
  ,StructField("usefulCount", IntegerType(), True)
  ,StructField("sentiment", DoubleType(), True)
  ])

# %%
df_test = spark.read.format("csv")\
           .option("delimiter", "|")\
           .option("header", "true")\
           .option("quote", "\"")\
           .option("escape", "\"")\
           .option("multiLine","true")\
           .option("quoteMode","ALL")\
           .option("mode","PERMISSIVE")\
           .option("ignoreLeadingWhiteSpace","true")\
           .option("ignoreTrailingWhiteSpace","true")\
           .option("parserLib","UNIVOCITY")\
           .schema(customschema)\
           .load(working_folder + "drug_reviews_with_sentiment_test.csv")

# %%
df_test.count()

# %%
df_test.show(5)

# %% [markdown]
# ## Load in and Explore Training Data

# %%
# Read in training data file
customschema = StructType([
  StructField("UniqueID", IntegerType(), True)
  ,StructField("drugName", StringType(), True)
  ,StructField("condition", StringType(), True)
  ,StructField("review", StringType(), True)
  ,StructField("rating", DoubleType(), True)
  ,StructField("date", StringType(), True)
  ,StructField("usefulCount", IntegerType(), True)
  ,StructField("sentiment", DoubleType(), True)
  ])

df = spark.read.format("csv")\
           .option("delimiter", "|")\
           .option("header", "true")\
           .option("escape", "\"")\
           .option("multiLine","true")\
           .option("quoteMode","ALL")\
           .option("mode","PERMISSIVE")\
           .option("ignoreLeadingWhiteSpace","true")\
           .option("ignoreTrailingWhiteSpace","true")\
           .option("parserLib","UNIVOCITY")\
           .schema(customschema)\
           .load(working_folder + "drug_reviews_with_sentiment_train.csv")

# %%
df.count()

# %%
df.show(5)

# %%
df.select('sentiment').summary().show()

# %% [markdown]
# ## TODO: Train model to predict star rating based off of the 'condition', 'usefulCount', and 'sentiment' with 'rating' as the target

# %%
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

# %%
df_train = df.drop('date', 'document', 'token', 'class')

# %%
df_train.show()

# %%
target = 'rating'
numeric_cols = ['usefulCount','sentiment']
#categorical_cols = ['condition']

# %%
# Use String Indexer to convert categorical values to a numeric index
#stringIndex = StringIndexer(inputCols=categorical_cols, handleInvalid='skip', outputCols=[x + "_idx" for x in categorical_cols])
#stringIndex_model = stringIndex.fit(df_train)
#train_df = stringIndex_model.transform(df_train).drop(*categorical_cols)

# %%
# Assemble the inputs into the format needed for the model
assemblerInputs = numeric_cols 
vectorAssembler = VectorAssembler(inputCols= assemblerInputs, outputCol="features")
train_df = vectorAssembler.transform(df_train).select('features', target)

# %%
train_df.show(5)

# %%
rf = RandomForestRegressor(labelCol=target)

# %%
pipeline_rf = Pipeline(stages= [vectorAssembler, rf]) 

# %%
# Fit Random Forest Model with pipeline
rf_pipelineModel = pipeline_rf.fit(df_train)

# %%
train_preds = rf_pipelineModel.transform(df_train)

# %%
# Select example rows to display.
train_preds.show(5)

# %%
evaluator = RegressionEvaluator(labelCol=target, predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(train_preds)
print("Root Mean Squared Error (RMSE) on train data = %g" % rmse)

# %%
# Drop unimportant columns for model 
df_test = df_test.drop('date', 'document', 'token', 'class')
# Drop rows with missing values
df_test = df_test.dropna()

# %%
## Drop rows where condition contains irrelevant strings
df_test = df_test.where(~df_test.condition.contains("</span>"))

# %%
df_test.count()

# %%
df_test.show(5)

# %%
df_test = df_test.drop('date', 'document', 'token', 'class')

# %%
test_preds = rf_pipelineModel.transform(df_test)

# %%
# Select example rows to display.
test_preds.show(5)

# %%
evaluator = RegressionEvaluator(labelCol=target, predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(test_preds)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

# %%
# https://www.sparkitecture.io/machine-learning/model-saving-and-loading

# %%
rf_pipelineModel.write().overwrite().save(working_folder+"pipeline_saved_model")

print("saved the model")



