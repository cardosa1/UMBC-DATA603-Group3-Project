#!/bin/bash

# Run the command to execute the pyspark job
gcloud dataproc jobs submit pyspark Sentiment_Extraction_Using_RF.py --cluster spark-drug-analysis-dataproc --region us-central1

# if above run was succesfull then do the following:
if [ $? -eq 0 ]; then
    gsutil mv gs://spark-drug-analysis/new_data/*.csv gs://spark-drug-analysis/archive_data/
else
    echo FAIL
fi