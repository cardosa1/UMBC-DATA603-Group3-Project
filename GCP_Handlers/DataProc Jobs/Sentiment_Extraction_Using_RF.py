from datetime import datetime
import subprocess
import pandas as pd

from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark import SparkContext


from pyspark.ml import PipelineModel

# Create a Spark Session
spark = SparkSession.builder.appName("drug-analysis").getOrCreate()
# Check Spark Session Information
spark
sc = SparkContext.getOrCreate();


today = datetime.now()
working_folder = "gs://spark-drug-analysis/"
path = working_folder+'new_data/'
result = subprocess.run(['gsutil', 'ls', path+'*.csv'], stdout=subprocess.PIPE)

all_dat = pd.DataFrame()
for file in result.stdout.splitlines():
	dat = pd.read_csv(file.decode("utf-8").strip())
	all_dat = all_dat.append(dat, ignore_index=True)

print(f'all_dat = {all_dat}')

new_data = all_dat[['usefulCount', 'sentiment']]
new_data_spark = spark.createDataFrame(new_data)

print(f'new_data_spark = {new_data_spark.show()}')

test_pipelineModel = PipelineModel.load(working_folder+'Model_Train/pipeline_saved_model')

test_preds = test_pipelineModel.transform(new_data_spark)

# save predictions to gs cloud
pandas_pred_df = test_preds.toPandas()
pandas_pred_df.to_csv(working_folder+'predicted_data'+'/CF_Rating_prediction'+'_'+str(today)+'.csv')

pandas_pred_df.head()

print("Saved the predictions to gs")