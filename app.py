import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the model and columns
model = pickle.load(open('yield_model.pkl', 'rb'))
model_columns = pickle.load(open('model_columns.pkl', 'rb'))

st.title("🌾 Malawi Crop Yield Predictor")
st.markdown("Predict yield based on climate and soil conditions.")

# Create input fields for the user
temp = st.number_input("Annual Mean Temperature", value=210)
precip = st.number_input("Annual Precipitation (mm)", value=1000)
hhsize = st.slider("Household Size", 1, 15, 5)
soil = st.selectbox("Soil Type", ["Sandy", "Clay", "Loamy"])

# Prepare data for prediction
input_data = pd.DataFrame([[temp, precip, hhsize]], columns=['annual_mean_temperature', 'annual_precipitation_mm', 'hhsize'])
# (Note: In a full app, you would add all features and match the One-Hot encoding)

if st.button("Predict Yield"):
    prediction_log = model.predict(input_data) # Assuming log model
    prediction = np.expm1(prediction_log)
    st.success(f"Estimated Yield: {prediction[0]:.2f} kg/ha")