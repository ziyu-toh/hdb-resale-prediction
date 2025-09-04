import json
import boto3
import pandas as pd
import urllib3
from io import StringIO

s3 = boto3.client("s3")
def lambda_handler(event, context):
    bucket_name = "hdb-resale-pred-raw"

    dataset_id = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
    url = "https://data.gov.sg/api/action/datastore_search"

    # create a PoolManager for making requests
    http = urllib3.PoolManager()

    # send GET request with query parameters
    response = http.request("GET", url, fields={"resource_id": dataset_id, "limit": 10})

    # decode bytes to string and parse JSON
    data = json.loads(response.data.decode("utf-8"))

    # convert to dataframe
    df = pd.json_normalize(data["result"]["records"])

    # Save to bucket as csv
    s3.put_object(Bucket=bucket_name, 
                  Key="FromJan2017onwards.csv",
                  Body=df.to_csv(index=False))