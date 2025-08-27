# This script pushes model to S3 bucket for now. To add on training code later on
import boto3
import os

# Replace with your S3 bucket name and desired object key
bucket_name = 'hdb-resale-model'
object_key = 'champion_model.pkl' # Path within your S3 bucket
local_file_path = 'models/champion_model.pkl'

# Create an S3 client
s3 = boto3.client('s3',
        aws_access_key_id=os.environ.get('AWS_ACESSS_KEY_ID'),
        aws_secret_access_key='AWS_SECRET_ACCESS_KEY',
        region_name='us-east-1'
        )

try:
    s3.upload_file(local_file_path, bucket_name, object_key)
    print(f"Successfully uploaded {local_file_path} to s3://{bucket_name}/{object_key}")
    
except Exception as e:
    print(f"Error uploading file: {e}")
