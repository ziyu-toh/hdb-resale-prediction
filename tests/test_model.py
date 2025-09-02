import numpy as np
import pickle
import pandas as pd

def test_prediction():
    # Load data
    sample_data = pd.DataFrame({"town":["ANG MO KIO"],
                                "flat_type":["2 ROOM"],
                                "flat_model_revised":["Improved"],
                                "flat_age_years":[46],
                                "floor_area_sqm":[44.0],
                                "days_from_earliest_data":[4323],
                                "storey_range_grouped":["0-15"]})
    
    with open('models/champion_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Prediction
    pred = model.predict(sample_data)

    assert model is not None, "No model found"
    assert pred is not None, "No predictions made"
    assert isinstance(pred[0], np.float64), "Prediction is not a float"
    