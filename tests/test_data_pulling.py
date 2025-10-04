import pandas as pd
import numpy as np
import os

from modelling.data_pulling.get_s3_data import connect_to_hdb_s3, get_hdb_s3_link, get_data_from_hdb_url, upload_data_to_s3

def test_connect_to_hdb_s3():
    response = connect_to_hdb_s3()
    assert response.status in [200, 201], f"Expected status 200 or 201 but got {response.status}"
    assert response.data, "No data received in response"

def test_get_hdb_s3_link():
    response = connect_to_hdb_s3()
    download_url = get_hdb_s3_link(response)
    assert download_url is not None

def test_get_data_from_hdb_url():
    response = connect_to_hdb_s3()
    download_url = get_hdb_s3_link(response)
    get_data_from_hdb_url(download_url)
    assert os.path.exists("/tmp/temp_file.csv")