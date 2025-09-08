import pickle
import polars as pl
from fastapi import FastAPI
from pydantic import BaseModel

# Load champion model. Once dockerised, it has to be connected to the file path on docker
# . refers to current WD, specified on Dockerfile
with open('./fastapi_app/models/champion_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

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
    input_df = pl.DataFrame([input_data])
    
    # Prediction
    try:
        prediction = loaded_model.predict(input_df)
        print("Prediction:", prediction)
        
    except Exception as e:
        print("Error during prediction:", e)
        
    return {"Prediction": prediction[0]}