# ==========================================
# Step 7 - Prepare Data for Model Training
# ==========================================

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ------------------------------------------
# Project Paths
# ------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "featured_data.csv"

# ------------------------------------------
# Load Dataset
# ------------------------------------------

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("MODEL TRAINING PREPARATION")
print("=" * 60)

# ------------------------------------------
# Features and Target
# ------------------------------------------

X = df.drop("Selling_Price", axis=1)

y = df["Selling_Price"]

print("\nFeature Shape")
print(X.shape)

print("\nTarget Shape")
print(y.shape)

# ------------------------------------------
# Train Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTrain Shape")
print(X_train.shape)

print("\nTest Shape")
print(X_test.shape)

print("\nPreparation Completed Successfully.")