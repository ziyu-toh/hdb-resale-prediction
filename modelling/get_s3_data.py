import json
import boto3
import urllib3
import urllib.request

bucket_name = "hdb-resale-pred-raw"
file_key = "FromJan2017onwards.csv"

dataset_id = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
url = f"https://api-open.data.gov.sg/v1/public/api/datasets/{dataset_id}/initiate-download"

# create a PoolManager for making requests
http = urllib3.PoolManager()

# send GET request with query parameters. Initiates the download, will return a 
print("Getting download URL for the dataset")
response = http.request("GET", url)

# decode bytes to string and parse JSON
data = json.loads(response.data.decode("utf-8"))

# extract download URL from JSON response
download_url = data["data"]["url"]

# Extract source bucket and key from obtained download URL
source_bucket = download_url.split("/", 4)[-2] # table-downloads-ingest.data.gov.sg
source_key = download_url.split("/", 4)[-1]

# Download file from source bucket
print("Downloading file from source bucket...")
urllib.request.urlretrieve("https://s3.ap-southeast-1.amazonaws.com/table-downloads-ingest.data.gov.sg/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/6f8109f7bce05c219b3825a999cc7f3a02cbc19fe536138a5eaf86bfe6d8711f.csv?AWSAccessKeyId=ASIAU7LWPY2WEOGEEWJV&Expires=1757340883&Signature=5fNhByF2%2BgUzYP79485VttC%2BquQ%3D&X-Amzn-Trace-Id=Root%3D1-68bed6c2-7effeac906e14a536097b462%3BParent%3D07b0d9c5b7e125b3%3BSampled%3D0%3BLineage%3D1%3Ab9934a3d%3A0&response-content-disposition=attachment%3B%20filename%3D%22ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv%22&x-amz-security-token=IQoJb3JpZ2luX2VjEFUaDmFwLXNvdXRoZWFzdC0xIkYwRAIgO7GC%2BYSgeHxxm6LCw7yg5qgzYUtJIK0gzlfXNQ4cCzUCIB13duqpLdqXqjgtPUvC72M%2B3Ot6OJh3ToH4OrxY7JsoKq8DCL7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBBoMMzQyMjM1MjY4NzgwIgwyTk2STii3Cnf3z8MqgwMQL38XggT1AnwlJIy9EAoD%2F4gvXRJ%2F89QjOcd%2BWS8ELUUWoJis%2FsTudwZ%2Fu%2BSMUKRtNDJcgyW7fWrXpcQHVWZW5IHPN77PjvWosfRGlRmObvPcj0eZXNPJNAFRgVAr454s1JB0UqPx9e%2BPJzYnYaZhGyryOawncLL%2B5qu1ADKcz%2FBarP%2FU9q%2FTVmsEO%2BmPO%2FrC4F0U2%2BOHfRdXkEPOha4xgaxPRmZsspf3UeNYJ5JV5rryYmw0OPn%2FYyj4v6JjNh6%2Bwe2Ya9r5I%2FdO0%2FKWzTFoUpVKgjx2d73Ij8vGJ%2BowpDNa153SSv2L764SJcSGjhdRRIZMH%2Bg%2BlBbHBoR8g2F398Xv%2F2iuYysNWdjOjQIp%2Bc71y%2BBQUcuWuW%2FfSYg9YInmZHkfToj%2Bm6PeQJQy%2BtoQ42LDFwP4EF4MtCr2CyGLxEdjeCkT4UMLVvX9O8PliO74GZhTSnW7VQcQWuEWG7LZQ5sz6itvYRjo0xpVwU7MCkGYysz2%2FzjVOq%2FPSGxoakgkzF8w4KP7xQY6ngH2LQ3h84jZFCYuGcW953kj2axJU9fQfizUEN%2BXhVWE29qeWUlrw5IDuCMaERgOYc%2B4h%2BUg9rT5LMF5SIHXFHs%2Fl6tExzgmJ3HfPvp%2FtINYWpb0URFtzq2b5wA9whcASKDAMHu1%2B3Hk0A0yOA01SJXJ8PcMdaF9bA9zijmH8pkgVDz3cEH6NKYRO3QH9vw9stdlU941oBUQ8FGG09neJw%3D%3D", 
                            "/tmp/temp_file.csv")

# Time to upload the data - Initialize S3 client
s3 = boto3.client('s3')

# Upload file to destination bucket
print("Uploading file to destination bucket...")
s3.upload_file("temp_file.csv", bucket_name, file_key)

print("File uploaded successfully.")