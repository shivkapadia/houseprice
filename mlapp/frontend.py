import streamlit as st
import requests
import pandas as pd
import json

# FastAPI URL (make sure it matches the URL of your FastAPI server)
BASE_URL = "http://localhost:8000"

st.title("House Price Prediction")

# Upload CSV file
st.header("Upload CSV File")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())

    if st.button("Submit Data"):
        # Convert DataFrame to JSON
        data_json = df.to_dict(orient="records")
        response = requests.post(f"{BASE_URL}/upload/", files={"file": uploaded_file})
        result = response.json()
        st.write("Mean Squared Error:", result["mean_squared_error"])

# Enter data manually
st.header("Enter Data Manually")
data_input = st.text_area("Enter JSON data", placeholder='[{"feature1": value1, "feature2": value2, ...}]')
if st.button("Submit Manual Data"):
    try:
        data_json = json.loads(data_input)
        response = requests.post(f"{BASE_URL}/manual/", json={"data": data_json})
        result = response.json()
        st.write("Mean Squared Error:", result["mean_squared_error"])
    except json.JSONDecodeError:
        st.error("Invalid JSON format. Please correct it and try again.")

# Save or load model
st.header("Save/Load Model")
if st.button("Save Model"):
    response = requests.post(f"{BASE_URL}/save-model/")
    result = response.json()
    st.write(result["message"])

if st.button("Load Model"):
    response = requests.post(f"{BASE_URL}/load-model/")
    result = response.json()
    st.write(result["message"])
