# app.py (Streamlit frontend)
import streamlit as st
import requests

st.title("HDB Resale Price Predictor")

flat_age_years = st.number_input("Flat Age (Years)", value=0)
floor_area_sqm = st.number_input("Floor Area (sqm)", value=0.0)
days_from_earliest_data = st.number_input("Days from Earliest Data", value=0)
flat_type = st.selectbox("Flat Type", ["1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM"])
storey_range_grouped = st.selectbox("Storey Range", ["1-15", "16-30", "31+"])
flat_model_revised = st.text_input("Flat Model (Revised)")
town = st.text_input("Town")

if st.button("Submit"):
    fastapi_url = "http://127.0.0.1/predict" # "http://54.169.110.104:80/predict" 
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
        
