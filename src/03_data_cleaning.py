# ==========================================
# Car Price Prediction using Polynomial Regression
# Step 4 - Data Cleaning
# ==========================================

import pandas as pd
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# File paths
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "car data.csv"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"

# Load dataset
df = pd.read_csv(RAW_DATA)

print("=" * 60)
print("DATA CLEANING")
print("=" * 60)

# -----------------------------
# Shape before cleaning
# -----------------------------
print("\nShape Before Cleaning:")
print(df.shape)

# -----------------------------
# Missing values
# -----------------------------
print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# Remove duplicate rows
# -----------------------------
duplicates = df.duplicated().sum()
print(f"\nDuplicate Rows Found: {duplicates}")

if duplicates > 0:
    df = df.drop_duplicates()
    print("Duplicate rows removed.")
else:
    print("No duplicate rows found.")

# -----------------------------
# Shape after cleaning
# -----------------------------
print("\nShape After Cleaning:")
print(df.shape)

# -----------------------------
# Save cleaned dataset
# -----------------------------
df.to_csv(PROCESSED_DATA, index=False)

print("\nCleaned dataset saved successfully!")
print(PROCESSED_DATA)