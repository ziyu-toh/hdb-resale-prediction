import mlflow
import pickle
import pandas as pd

# Specify best ID
BEST_RUN_ID = "runs:/1f85486f1d7a49109dc6074f9c0b2421/model"

# Load data
test_df = pd.read_parquet('data/processed/test.parquet')

# Registering best model to MLflow Model Registry
mlflow.register_model(
    model_uri=BEST_RUN_ID, # run_id is from the experiment, not the model
    name="HDBResalePricePrediction",
)

# Reloading best model, saving to models directory for deployment
loaded_model = mlflow.sklearn.load_model(BEST_RUN_ID)
print("Predictions: ", loaded_model.predict(test_df.drop(columns=['resale_price']).iloc[:5]))

with open('models/champion_model.pkl', 'wb') as f:
    pickle.dump(loaded_model, f)