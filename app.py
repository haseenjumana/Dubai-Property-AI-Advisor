import pandas as pd
import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🏠 Dubai Real Estate Price Predictor")

st.markdown("Enter the property details below:")

area = st.number_input("Area (sq ft)", min_value=200, max_value=10000, value=1000)
bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=2)
bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=2)

location = st.selectbox("Location", ["Dubai Marina", "Downtown Dubai", "Deira", "Bur Dubai", "JLT"])

if st.button("Predict Price"):
    input_data = pd.DataFrame([[location, area, bedrooms, bathrooms]],
                              columns=["location", "area", "bedrooms", "bathrooms"])
    
    st.write("📊 Input Data Preview:")
    st.write(input_data)

    try:
        prediction = model.predict(input_data)[0]
        st.success(f"Estimated Price: AED {int(prediction):,}")
    except Exception as e:
        st.error(f"🚨 Prediction failed: {e}")


