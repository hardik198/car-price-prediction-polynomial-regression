# ==========================================
# Car Price Prediction using Polynomial Regression
# Step 2 - Load and Explore Dataset
# ==========================================

import pandas as pd
from pathlib import Path



# Project root folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset path
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "car data.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 60)

# Shape
print("\nDataset Shape")
print(df.shape)

# Columns
print("\nColumn Names")
print(df.columns.tolist())

# First 5 rows
print("\nFirst 5 Rows")
print(df.head())

# Last 5 rows
print("\nLast 5 Rows")
print(df.tail())

# Data types
print("\nData Types")
print(df.dtypes)

# Dataset information
print("\nDataset Info")
df.info()

# Statistical summary
print("\nStatistical Summary")
print(df.describe(include="all"))