import json
import boto3
import urllib3
import urllib.request

def connect_to_hdb_s3():
    dataset_id = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
    url = f"https://api-open.data.gov.sg/v1/public/api/datasets/{dataset_id}/initiate-download"

    # create a PoolManager for making requests
    http = urllib3.PoolManager()

    # send GET request with query parameters. Initiates the download, will return a 
    print("Getting download URL for the dataset")
    try:
        response = http.request("GET", url)
        return response
        
    except Exception as e:
        print("Error occurred while making the request:", e)
        raise

def get_hdb_s3_link(response):
    # decode bytes to string and parse JSON
    data = json.loads(response.data.decode("utf-8"))

    # extract download URL from JSON response
    download_url = data["data"]["url"]
    
    return download_url

def get_data_from_hdb_url(download_url):
    # Download file from source bucket to tmp folder
    print("Downloading file from source bucket...")
    try:
        urllib.request.urlretrieve(download_url, 
                                    "/tmp/temp_file.csv")
    except Exception as e:
        print("Error occurred while downloading the file:", e)
        raise

def upload_data_to_s3(bucket_name, file_key):
    # Time to upload the data - Initialize S3 client
    s3 = boto3.client('s3')

    # Upload file to destination bucket
    print("Uploading file from tmp folder to destination bucket...")
    try:
        s3.upload_file("/tmp/temp_file.csv", bucket_name, file_key)
    
    except Exception as e:
        print("Error occurred while uploading the file:", e)
        raise
    
    print("File uploaded successfully.")

def lambda_handler(event, context):
    bucket_name = "hdb-resale-pred-raw"
    file_key = "FromJan2017onwards.csv"
    
    response = connect_to_hdb_s3()
    download_url = get_hdb_s3_link(response)
    get_data_from_hdb_url(download_url)
    upload_data_to_s3(bucket_name, file_key)
    
 