# Following gcloud commands create pub/sub topic

gcloud pubsub topics create TOPIC_ID  --schema=SCHEMA_ID  --project=TOPIC_PROJECT

# This command is used to create schema that will be used by the pub/sub topic
# For our project SCHEMA_DEFINATION refer to Schema_avro.json

gcloud pubsub schemas create SCHEMA_ID --type=SCHEMA_TYPE --definition=SCHEMA_DEFINITION