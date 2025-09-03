from app.fastapi.main import app, loaded_model
from fastapi.testclient import TestClient
import numpy as np
import polars as pl

sample_json = {"town":"Ang Mo Kio",
                "flat_type":"2 Room",
                "flat_model_revised":"Improved",
                "flat_age_years":46,
                "floor_area_sqm":44.0,
                "days_from_earliest_data":4323,
                "storey_range_grouped":"1-15"}

def test_prediction():
    # Load data
    sample_data = pl.DataFrame([sample_json])
    
    # Prediction
    pred = loaded_model.predict(sample_data)

    assert loaded_model is not None
    assert pred is not None
    assert isinstance(pred[0], np.float64)

def test_read_main():
    client = TestClient(app)
    response = client.get("/")
    
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World"}
    
def test_app_prediction():
    client = TestClient(app)
    response = client.post(
        "/predict",
        json=sample_json,
    )
    assert response.status_code == 200
    assert response.json() == {"Prediction": response.json()["Prediction"]}