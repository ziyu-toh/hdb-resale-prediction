import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load model from s3 by downloading to tmp folder first. 
# #Once dockerised, it has to be connected to the file path on docker
loaded_model = joblib.load('./app/fastapi_app/champion_model/champion_model.joblib')

# Define input data model
class InputData(BaseModel):
    flat_age_years: int
    floor_area_sqm: float
    days_from_earliest_data: int
    flat_type: str
    flat_model_revised: str
    town: str
    storey_range_grouped: str
    
# Initialize FastAPI app
app = FastAPI()

@app.get("/")
async def root():
    return {"msg": "Hello World"} 

@app.post("/predict")
async def predict(input_data: InputData):
    input_df = pd.DataFrame([input_data.model_dump()])
    
    # Prediction
    try:
        prediction = loaded_model.predict(input_df)
        print("Prediction:", prediction)
        
    except Exception as e:
        print("Error during prediction:", e)
        
    return {"Prediction": prediction[0]}