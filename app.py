import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

model = joblib.load(PROJECT_ROOT / "models" / "best_model.pkl")
poly = joblib.load(PROJECT_ROOT / "models" / "polynomial_features.pkl")

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗"
)

st.title("🚗 Car Price Prediction")
st.write("Enter the car details below to predict its selling price.")

present_price = st.number_input("Present Price (Lakh ₹)", min_value=0.0, value=5.0)
kms_driven = st.number_input("Kilometers Driven", min_value=0, value=30000)

fuel = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "CNG"]
)

seller = st.selectbox(
    "Seller Type",
    ["Dealer", "Individual"]
)

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic"]
)

owner = st.number_input("Previous Owners", min_value=0, max_value=5, value=0)

car_age = st.number_input("Car Age", min_value=0, max_value=30, value=5)

fuel_map = {
    "Petrol":0,
    "Diesel":1,
    "CNG":2
}

seller_map = {
    "Dealer":0,
    "Individual":1
}

transmission_map = {
    "Manual":0,
    "Automatic":1
}

if st.button("Predict Price"):

    df = pd.DataFrame([{
        "Present_Price":present_price,
        "Kms_Driven":kms_driven,
        "Fuel_Type":fuel_map[fuel],
        "Seller_Type":seller_map[seller],
        "Transmission":transmission_map[transmission],
        "Owner":owner,
        "Car_Age":car_age
    }])

    df_poly = poly.transform(df)

    prediction = model.predict(df_poly)

    st.success(f"Predicted Selling Price: ₹ {prediction[0]:.2f} Lakh")