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
working_folder = "gs://spark-drug-analysis/"

# %% [markdown]
# # Install Libraries and Dependencies

# %%
# # Install PySpark and Spark NLP
# ! pip install -qq pyspark==3.2.1 spark-nlp findspark 

# %%
# !wget http://setup.johnsnowlabs.com/colab.sh -O - | bash

# %%
import pyspark.pandas as ps
import pandas as pd
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark import SparkContext
from sparknlp.pretrained import PretrainedPipeline
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *

spark = sparknlp.start()

print("Spark NLP version: {}".format(sparknlp.version()))
print("Apache Spark version: {}".format(spark.version))

sc = SparkContext.getOrCreate();

# %%
"""# Import SparkSession
from pyspark.sql import SparkSession
# Create a Spark Session
spark = SparkSession.builder.master("local[*]").getOrCreate()
# Check Spark Session Information
spark"""

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
  ])

# %%
df_test = spark.read.format("csv")\
           .option("delimiter", "\t")\
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
           .load(working_folder + "drugsComTest_raw.tsv")

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
  ])

df = spark.read.format("csv")\
           .option("delimiter", "\t")\
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
           .load(working_folder + "drugsComTrain_raw.tsv")

# %%
df.count()

# %%
df.show(5)

# %% [markdown]
# ### Clean Training Dataset

# %%
# Remove rows with null columns
df = df.dropna()
df_test = df_test.dropna()

# %%
df.count()

# %%
df_test.count()

# %%
# Drop conditions with </span> tag
df = df.where(~df.condition.contains("</span>"))

# %%
df_test = df_test.where(~df_test.condition.contains("</span>"))

# %%
df.count()

# %%
df_test.count()

# %%
df.groupby('rating').count().orderBy("rating", ascending=False).show()

# %%
# Average Star Rating by Condition
df.groupBy("condition").agg({'rating':'avg', 'condition':'count'}).orderBy("count(condition)",ascending=False).show()

# %%
# Average Star Rating by Drug Name 
df.groupBy("drugName").agg({'rating':'avg', 'drugName':'count'}).orderBy("count(drugName)",ascending=False).show()

# %%
pd_df_train = df.toPandas()

# %%
pd_df_test = df_test.toPandas()

# %% [markdown]
# # Use TextBlob to Extract Sentiments
# John Snow Labs sentiment models do not provide us with a continuous sentiment score, but simply postive or negative labels. Also, due to lazy evaluation in Spark the inference transformations blow up the size of the dataframe causing it to be unworkable on our UMBC Colab instances. Because of this, we have opted to use the TextBlob library to obtain the sentiment polarity and Swifter to parallelize the inference operation to train our final model. 

# %%
#!pip install swifter -qq

# %%
from textblob import TextBlob
#import swifter

# %%
def get_sentiment(text):
  return TextBlob(text).sentiment.polarity

# %%
pd_df_train['sentiment'] = pd_df_train['review'].apply(get_sentiment)

# %%
pd_df_train.head()

# %%
pd_df_train.sentiment.describe()

# %%
pd_df_test['sentiment'] = pd_df_test['review'].apply(get_sentiment)

# %%
pd_df_test.head()

# %%
import csv

# %%
pd_df_train.to_csv(working_folder + "drug_reviews_with_sentiment_train.csv", index=False, sep='|', quoting=csv.QUOTE_MINIMAL)
pd_df_test.to_csv(working_folder + "drug_reviews_with_sentiment_test.csv", index=False, sep='|', quoting=csv.QUOTE_MINIMAL)

# %% [markdown]
# # Use John Snow Labs pretrained sentiment models pipeline
# 

# %% [markdown]
# https://nlp.johnsnowlabs.com/
# 
# Medium Article: 
# https://medium.com/analytics-vidhya/sentiment-analysis-with-sparknlp-couldnt-be-easier-2a8ea3b728a0
# 
# John Snow Labs Reference Notebook: 
# https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/quick_start_google_colab.ipynb#scrollTo=tyMMD_upEfIa
# 
# This model using BioBERT would potentially perform better, but it is not free-tier:
# https://nlp.johnsnowlabs.com/2022/07/28/bert_sequence_classifier_drug_reviews_webmd_en_3_0.html
# 

# %% [markdown]
# ### Use Twitter Sentiment Analysis Model: analyze_sentimentdl_use_twitter 
# 
# Model: https://nlp.johnsnowlabs.com/2021/01/18/sentimentdl_use_twitter_en.html
# 
# Universal Sentence Encoder: https://nlp.johnsnowlabs.com/2020/04/17/tfhub_use.html

# # %%
# """pipeline = PretrainedPipeline('analyze_sentimentdl_use_twitter', 'en')
# pipeline.model.stages
# # rename the text column as 'text', pipeline expects 'text' 
# df_result = pipeline.transform(df.withColumnRenamed("review", "text"))
# # Extract results from the "sentiments" column
# df_twitter_sentiments = df_result.withColumn("sentiment", explode('sentiment.result')).drop(*['document','sentence_embeddings'])"""

# # %% [markdown]
# # 
# # A vast majority of the reviews are negative
# # 
# # 
# # +---------+------+<br>
# # |sentiment| count|<br>
# # +---------+------+<br>
# # | positive| 31299|<br>
# # |  neutral|  6568|<br>
# # | negative|123430|<br>
# # +---------+------+
# # 
# # 

# # %%
# # took 20 minutes to run
# #df_twitter_sentiments.groupBy('sentiment').count().show()

# # %% [markdown]
# # ### Use RoBERTa Sentiment Classifier: roberta_classifier_autotrain_sentiment_polarity_918130222
# # 
# # Model: https://nlp.johnsnowlabs.com/2022/09/19/roberta_classifier_autotrain_sentiment_polarity_918130222_en.html
# # 
# # HuggingFace: https://huggingface.co/docs/transformers/model_doc/roberta
# # 
# # Breakdown how pretrained pipeline works under the hood: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SENTIMENT_EN.ipynb

# # %%
# # documentAssembler = DocumentAssembler() \
# #         .setInputCol("review") \
# #         .setOutputCol("document")

# # tokenizer = Tokenizer() \
# #     .setInputCols("document") \
# #     .setOutputCol("token")

# # seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_autotrain_sentiment_polarity_918130222","en") \
# #     .setInputCols(["document", "token"]) \
# #     .setOutputCol("sentiment")

# # %%
# #nlp_pipeline = Pipeline(stages=[documentAssembler, tokenizer, seq_classifier])

# # %%
# #df_train = nlp_pipeline.fit(df).transform(df)

# # %%
# df_train = df_train.withColumn("sentiment", explode('sentiment.result')).drop('document','token','class')

# # %%
# df_train.show()

# # %%
# df_train = df_train.withColumn("sentiment", df_train["sentiment"].cast(DoubleType()))

# # %%
# df_train.count()

# # %%
# import py4j.protocol  
# from py4j.protocol import Py4JJavaError  
# from py4j.java_gateway import JavaObject  
# from py4j.java_collections import JavaArray, JavaList

# from pyspark import RDD, SparkContext  
# from pyspark.serializers import PickleSerializer, AutoBatchedSerializer


# # Helper function to convert python object to Java objects
# def _to_java_object_rdd(rdd):  
#     """ Return a JavaRDD of Object by unpickling
#     It will convert each Python object into Java object by Pyrolite, whenever the
#     RDD is serialized in batch or not.
#     """
#     rdd = rdd._reserialize(AutoBatchedSerializer(PickleSerializer()))
#     return rdd.ctx._jvm.org.apache.spark.mllib.api.python.SerDe.pythonToJava(rdd._jrdd, True)

# # First you have to convert it to an RDD 
# JavaObj = _to_java_object_rdd(small_df.rdd)

# # Now we can run the estimator
# sc._jvm.org.apache.spark.util.SizeEstimator.estimate(JavaObj)

# # %%
# # First you have to convert it to an RDD 
# JavaObj = _to_java_object_rdd(df_train.rdd)

# # Now we can run the estimator
# sc._jvm.org.apache.spark.util.SizeEstimator.estimate(JavaObj)

# # %%
# #df_train.write.mode('overwrite').parquet(working_folder + "drug_reviews_with_sentiment_train.parquet")

# # %%
# # Drop rows with missing values
# df_test = df_test.dropna()

# # %%
# ## Drop rows where condition contains irrelevant strings
# df_test = df_test.where(~df_test.condition.contains("</span>"))

# # %%
# df_test.count()

# # %%
# df_test = nlp_pipeline.fit(df_test).transform(df_test)

# # %%
# df_test = df_test.withColumn("sentiment", explode('class.result')).drop(*['token','class','document'])

# # %%
# df_test.show()

# # %%
# # Write complete dataframe to disk
# #df_test.write.csv(working_folder + "drug_reviews_with_sentiment_test.csv")
# # Write complete dataframe to disk
# #pd_df_test = df_test.toPandas()
# #pd_df_test.to_csv(working_folder + "drug_reviews_with_sentiment_test.csv")
# df_test.write.mode('overwrite').parquet(working_folder + "drug_reviews_with_sentiment_test.parquet")


