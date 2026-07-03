import joblib
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Load saved model and polynomial transformer
model = joblib.load(PROJECT_ROOT / "models" / "best_model.pkl")
poly = joblib.load(PROJECT_ROOT / "models" / "polynomial_features.pkl")

print("=" * 50)
print("CAR PRICE PREDICTION")
print("=" * 50)

present_price = float(input("Present Price (in lakh): "))
kms_driven = int(input("Kilometers Driven: "))
fuel_type = int(input("Fuel Type (Petrol=0, Diesel=1, CNG=2): "))
seller_type = int(input("Seller Type (Dealer=0, Individual=1): "))
transmission = int(input("Transmission (Manual=0, Automatic=1): "))
owner = int(input("Number of Previous Owners: "))
car_age = int(input("Car Age (years): "))

input_df = pd.DataFrame([{
    "Present_Price": present_price,
    "Kms_Driven": kms_driven,
    "Fuel_Type": fuel_type,
    "Seller_Type": seller_type,
    "Transmission": transmission,
    "Owner": owner,
    "Car_Age": car_age
}])

input_poly = poly.transform(input_df)

prediction = model.predict(input_poly)

print("\n" + "=" * 50)
print(f"Predicted Selling Price: ₹{prediction[0]:.2f} Lakh")
print("=" * 50)