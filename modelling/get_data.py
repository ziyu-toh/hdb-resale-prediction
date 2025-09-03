# Test for AWS
import urllib3
import json
import pandas as pd

dataset_id = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
url = "https://data.gov.sg/api/action/datastore_search"

# create a PoolManager for making requests
http = urllib3.PoolManager()

# send GET request with query parameters
response = http.request("GET", url, fields={"resource_id": dataset_id, "limit": 10})

# decode bytes to string and parse JSON
data = json.loads(response.data.decode("utf-8"))

print(data["result"]["records"])

# convert to dataframe
df = pd.json_normalize(data["result"]["records"])
print(df.shape)