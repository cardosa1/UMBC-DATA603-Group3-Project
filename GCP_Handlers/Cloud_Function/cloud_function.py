#{"UniqueID":1234, "drugName": "Dolo", "condition": "Fever", "review": "Fantastic", "rating": 10, "date": "11-30-2022", "usefulCount": 100}

import base64
from textblob import TextBlob
import json
import pandas as pd
import csv
from datetime import datetime

def get_sentiment(text):
  return TextBlob(text).sentiment.polarity

def hello_pubsub(event, context):
    """Triggered from a message on a Cloud Pub/Sub topic.
    Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
    """
    today = datetime.now()
  
    print("Current Date and Time :", today)
    print("Current Month :", today.month)
    print("Current Year :", today.year)

    working_folder = "gs://spark-drug-analysis/"

    pubsub_message = json.loads(base64.b64decode(event['data']))
    print(pubsub_message)
    
    review = pubsub_message['review']
    UniqueID = pubsub_message['UniqueID']
    drugName = pubsub_message['drugName']
    condition = pubsub_message['condition']
    date = pubsub_message['date']
    usefulCount = pubsub_message['usefulCount']

    # calculate sentiment based on review
    sentiment = get_sentiment(review)

    # save data to gs bucket
    data = [review, UniqueID, drugName, condition, date, usefulCount, sentiment]
    df = pd.DataFrame([data], columns = ['review', 'UniqueID', 'drugName', 'condition', 'date', 'usefulCount', 'sentiment'])
    df.to_csv(working_folder+str(today.month)+'-'+str(today.year)+'/CF_sentiment_extraction'+'_'+str(today)+'.csv')
    
    print("created file and loaded data")