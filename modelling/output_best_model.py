import mlflow
import pickle
import pandas as pd

# Specify best ID: To check on MLFlow
BEST_RUN_ID = "runs:/d796c6f22a124971aa3ebce48b7916b8/model"

# Registering best model to MLflow Model Registry
mlflow.register_model(
    model_uri=BEST_RUN_ID, # run_id is from the experiment, not the model
    name="HDBResalePricePrediction"
)

# Reloading best model, saving to models directory for deployment
loaded_model = mlflow.sklearn.load_model(BEST_RUN_ID)

with open('models/champion_model.pkl', 'wb') as f:
    pickle.dump(loaded_model, f)
    
with open('app/fastapi/models/champion_model.pkl', 'wb') as f:
    pickle.dump(loaded_model, f)
    