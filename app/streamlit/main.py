# app.py (Streamlit frontend)
import streamlit as st
import requests
import yaml
from datetime import datetime, date

st.title("HDB Resale Price Predictor")

# Load preset data values
with open("./app/config.yaml", "r") as file:
    config = yaml.safe_load(file)
    
# Define days from earliest data point (1st March 2012) at the point of running the app
days_from_earliest_data = (datetime.now().date() - date(2012, 3, 1)).days

# Specify buttons in streamlit
flat_age_years = st.number_input("Flat Age (Years)", value=0)
floor_area_sqm = st.number_input("Floor Area (sqm)", value=0.0)
flat_type = st.selectbox("Flat Type", config["data_flat_types"])
storey_range_grouped = st.selectbox("Storey Range", config["data_storey_range"])
flat_model_revised = st.selectbox("Flat Model (Revised)", config["data_flat_models"])
town = st.selectbox("Town", config["data_towns"])

if st.button("Submit"):
    fastapi_url = "http://fastapi:80/predict" 
    payload = {
        "flat_age_years": flat_age_years,
        "floor_area_sqm": floor_area_sqm,
        "days_from_earliest_data": days_from_earliest_data,
        "flat_type": flat_type,
        "flat_model_revised": flat_model_revised,
        "town": town,
        "storey_range_grouped": storey_range_grouped
    }

    try:
        response = requests.post(fastapi_url, json=payload)
        
        if response.status_code == 200:
            st.error(f"FastAPI response: {response.json()}")
       
        else:
            st.error(f"Error from FastAPI: {response.status_code} - {response.text}")
    
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to FastAPI backend. Ensure it is running.")
        
