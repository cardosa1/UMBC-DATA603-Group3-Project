# Drug Sentiment Analysis

This Project is trained on the UCIML Drug Review Dataset. The dataset provides patient reviews on specific drugs along with related conditions and a 10-star patient rating system reflecting overall patient satisfaction. The data was obtained by crawling online pharmaceutical review sites. This data was published in a study on sentiment analysis of drug experience over multiple facets, ex. sentiments learned on specific aspects such as effectiveness and side effects.

## Built With

PySpark - Random Forest, TextBlob

Google Cloud - Pub/Sub, Cloud Functions, Cloud Storage, DataProc, Crontab and Spark Serverless

## Environment Setup

<img width="799" alt="image" src="https://user-images.githubusercontent.com/98969137/206949951-03ee2681-d322-4445-ad38-d0dbcd86ef70.png">

1. Step 1: Creating Pub/Sub Topic on GCP <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;gcloud pubsub topics create DRUG-REVIEW-TOPIC   --schema=drug-reviews  --project=Drug-Analysis

2. Step 2: Create the schematic format of incoming data.<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;           gcloud pubsub schemas create DRUG-REVIEWS --type=AVRO --definition=SCHEMA_DEFINITION (json file)<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;            Ex: {"UniqueID":1235, "drugName": "Dolo", "condition": "Fever", "review": "Worst didn't work", "rating": 0, "date": "10-30-2022", "usefulCount": 10} 

3. Step 3: Create a Cloud Function with the above Pub/Sub Topic as the Trigger. <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;           Use cloud_function.py in our code folder structure
4. Step 4: Spark serverless batch creation for Model training <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;           Create batch using model_training.py file

5. Step 5: Create a DataProc Cluster to schedule the Rating_Prediction Job.<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Following gcloud command cerates dataproc cluster with 2 worker nodes of 200GB disk size and 1 master node with boot disk<br /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;of size 100GB.<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;gcloud dataproc clusters create spark-drug-analysis-dataproc --enable-component-gateway --region us-central1 --subnet default<br /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--zone us-central1-b --master-machine-type n1-standard-4 --master-boot-disk-size 100 --num-workers 2 --worker-machine-type &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;n2d-standard-4 --worker-boot-disk-size 200 --image-version 2.0-rocky8 --optional-components JUPYTER,ZOOKEEPER --scopes <br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'https://www.googleapis.com/auth/cloud-platform' --project newagent-bba27<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Schedule the job using crontab <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Crontab -e {Contents: 30 6 30 * * shell_invoke.sh >> /var/logs/cron.log 2>&1}



## Code Run

1. Sentiment_Extraction Cloud Function is trigged automatically and dumps the new data with sentiment details to cloud storage.
3. A spark serverless batch is created to run the Model Training and to save the trained model to cloud storage.
2. Using the pre-trained model DataProc Schedules a CronJob with shell script that runs the Rating_Exatration.py file on new streamed data.

## Collaborators

| Names                  | Year   | Major         |
| :---:                  | :---:  | :---:         |
| Pooja Kangokar Pranesh | 2022   | Data Science  |
| Elizabeth Cardosa      | 2022   | Data Science  |
| Yun-Zih Chen           | 2022   | Data Science  |


