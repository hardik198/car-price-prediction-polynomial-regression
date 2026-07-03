# ==========================================
# Car Price Prediction using Polynomial Regression
# Step 5 - Exploratory Data Analysis (EDA)
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# File paths
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
IMAGE_PATH = PROJECT_ROOT / "images"

# Create images folder if it doesn't exist
IMAGE_PATH.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

# ==========================================
# 1. Dataset Shape
# ==========================================
print("=" * 60)
print("DATASET SHAPE")
print("=" * 60)
print(df.shape)

# ==========================================
# 2. Selling Price Distribution
# ==========================================
plt.figure(figsize=(8, 5))
plt.hist(df["Selling_Price"], bins=20)
plt.title("Selling Price Distribution")
plt.xlabel("Selling Price")
plt.ylabel("Count")

plt.savefig(IMAGE_PATH / "selling_price_distribution.png")
plt.close()

print("✓ Selling Price Distribution saved.")

# ==========================================
# 3. Fuel Type Count
# ==========================================
plt.figure(figsize=(6, 5))
df["Fuel_Type"].value_counts().plot(kind="bar")

plt.title("Fuel Type")
plt.xlabel("Fuel")
plt.ylabel("Count")

plt.savefig(IMAGE_PATH / "fuel_type.png")
plt.close()

print("✓ Fuel Type graph saved.")

# ==========================================
# 4. Transmission Count
# ==========================================
plt.figure(figsize=(6, 5))
df["Transmission"].value_counts().plot(kind="bar")

plt.title("Transmission")
plt.xlabel("Transmission")
plt.ylabel("Count")

plt.savefig(IMAGE_PATH / "transmission.png")
plt.close()

print("✓ Transmission graph saved.")