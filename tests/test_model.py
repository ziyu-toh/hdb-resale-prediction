import numpy as np
import joblib
import pandas as pd
import boto3

def test_prediction():
    # Load data
    sample_data = pd.DataFrame({"town":["ANG MO KIO"],
                                "flat_type":["2 ROOM"],
                                "flat_model_revised":["Improved"],
                                "flat_age_years":[46],
                                "floor_area_sqm":[44.0],
                                "days_from_earliest_data":[4323],
                                "storey_range_grouped":["1-15"]})
    
    # Load model from s3 by downloading to tmp folder first
    s3_input = boto3.resource('s3', region_name='ap-southeast-1')
    input_bucket = s3_input.Bucket('hdb-resale-best-model')
    input_bucket.download_file('champion_model.joblib', '/tmp/champion_model.joblib')
    loaded_model = joblib.load('/tmp/champion_model.joblib')

    # Prediction
    pred = loaded_model.predict(sample_data)
    print("Prediction:", pred)

    assert loaded_model is not None, "No model found"
    assert pred is not None, "No predictions made"
    assert isinstance(pred[0], np.float64), "Prediction is not a float"
    