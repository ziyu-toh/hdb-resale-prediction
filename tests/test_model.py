import numpy as np
import pickle
import pandas as pd

def test_prediction():
    # Load data
    test_df = pd.read_parquet('data/processed/test.parquet')
    
    with open('models/champion_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Prediction
    pred = model.predict(test_df.drop(columns=['resale_price']).iloc[:1])

    assert model is not None, "No model found"
    assert pred is not None, "No predictions made"
    assert isinstance(pred[0], np.float64), "Prediction is not a float"
    