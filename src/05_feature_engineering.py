# ==========================================
# Step 6 - Feature Engineering
# ==========================================

import pandas as pd
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "featured_data.csv"

# Load dataset
df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# ------------------------------------------
# Create Car Age
# ------------------------------------------

CURRENT_YEAR = 2025

df["Car_Age"] = CURRENT_YEAR - df["Year"]

# ------------------------------------------
# Remove Car_Name
# ------------------------------------------

df.drop("Car_Name", axis=1, inplace=True)

# ------------------------------------------
# Encode Fuel_Type
# ------------------------------------------

df["Fuel_Type"] = df["Fuel_Type"].map({
    "Petrol":0,
    "Diesel":1,
    "CNG":2
})

# ------------------------------------------
# Encode Seller_Type
# ------------------------------------------

df["Seller_Type"] = df["Seller_Type"].map({
    "Dealer":0,
    "Individual":1
})

# ------------------------------------------
# Encode Transmission
# ------------------------------------------

df["Transmission"] = df["Transmission"].map({
    "Manual":0,
    "Automatic":1
})

# ------------------------------------------
# Remove Year Column
# ------------------------------------------

df.drop("Year", axis=1, inplace=True)

# ------------------------------------------
# Save dataset
# ------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)

print("\nFeature Engineering Completed Successfully")

print("\nFinal Shape")

print(df.shape)

print("\nColumns")

print(df.columns.tolist())

print("\nSaved To")

print(OUTPUT_FILE)