# Drug Sentiment Analysis

This Project is trained on the UCIML Drug Review Dataset. The dataset provides patient reviews on specific drugs along with related conditions and a 10-star patient rating system reflecting overall patient satisfaction. The data was obtained by crawling online pharmaceutical review sites. This data was published in a study on sentiment analysis of drug experience over multiple facets, ex. sentiments learned on specific aspects such as effectiveness and side effects.

## Built With

PySpark - Random Forest, TextBlob

Google Cloud - Pub/Sub, Cloud Functions, Cloud Storage, DataProc, Crontab and Spark Serverless

## Environment Setup

<img width="799" alt="image" src="https://user-images.githubusercontent.com/98969137/206949951-03ee2681-d322-4445-ad38-d0dbcd86ef70.png">

1. Step 1: Create Pub/Sub Topic on GCP
2. Step 2: Create the schematic format of incoming data. Ex: {"UniqueID":1235, "drugName": "Dolo", "condition": "Fever", "review": "Worst didn't work", "rating": 0, "date": "10-30-2022", "usefulCount": 10}
3. Step 3: Create a Cloud Function with the above Pub/Sub Topic as the Trigger.
4. Step 4: Create a DataProc Cluster to schedule the Rating_Prediction Job.

