# app.py

import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("rf_model.pkl")

st.title("Salary Purchase Prediction App")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
salary = st.number_input("Salary", min_value=10000, max_value=200000, value=50000)

# Create input dataframe
input_df = pd.DataFrame([[age, salary]], columns=["Age", "Salary"])  # Match model

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("✅ Likely to make a purchase.")
    else:
        st.warning("❌ Not likely to make a purchase.")
